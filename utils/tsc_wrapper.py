'''
@Author: WANG Maonan
@Date: 2023-09-08 15:49:30
@Description: 处理 TSCHub ENV 中的 state, reward
+ state: 5 个时刻的每一个 movement 的 queue length
+ reward: 路口总的 waiting time
@LastEditTime: 2023-09-13 17:01:02
'''
import numpy as np
import gymnasium as gym
from gymnasium.core import Env
from collections import deque
from typing import Any, SupportsFloat, Tuple, Dict, List

class OccupancyList:
    def __init__(self) -> None:
        self.elements = []

    def add_element(self, element) -> None:
        if isinstance(element, list):
            if all(isinstance(e, float) for e in element):
                self.elements.append(element)
            else:
                raise ValueError("列表中的元素必须是浮点数类型")
        else:
            raise TypeError("添加的元素必须是列表类型")

    def clear_elements(self) -> None:
        self.elements = []

    def calculate_average(self) -> float:
        """计算一段时间的平均 occupancy
        """
        arr = np.array(self.elements)
        averages = np.mean(arr, axis=0, dtype=np.float32)/100
        self.clear_elements() # 清空列表
        return averages


class TSCEnvWrapper(gym.Wrapper):
    """TSC Env Wrapper for single junction with tls_id
    """
    def __init__(self, env: Env, tls_id:str, max_states:int=5) -> None:
        super().__init__(env)
        self.tls_id = tls_id # 单路口的 id
        self.states = deque([self._get_initial_state()] * max_states, maxlen=max_states)
        self.movement_ids = None
        self.phase_num = None # phase 数量
        self.llm_static_information = None # Static information, (1). Intersection Geometry; (2). Signal Phases Structure
        self.phase2movements = None
        
        #Dynamic Information
        self.state = None # 当前的 state
        self.last_state = None # 上一时刻的 state
        self.occupancy = OccupancyList()
    
    def transform_occ_data(self, occ:List[float]) -> Dict[str, float]:
        """将 avg_occupancy 与每一个 movement id 对应起来

        Args:
            occ (List[float]): _description_

        Returns:
            Dict[str, float]: _description_
        """
        output_dict = {}
        for movement_id, value in zip(self.movement_ids, occ):
            if 'r' in movement_id:
                continue
            output_dict[movement_id] = f"{value*100}%" # 转换为占有率
        return output_dict
    
    # ###########################
    # Custom Tools for TSC Agent
    # ###########################
    
    def get_available_actions(self) -> List[int]:
        """获得控制信号灯可以做的动作
        """
        tls_available_actions = list(range(self.phase_num))
        return tls_available_actions
    
    def get_current_occupancy(self):
        return self.state
    
    def get_previous_occupancy(self):
        return self.last_state
    
    def _get_initial_state(self) -> List[int]:
        # 返回初始状态，这里假设所有状态都为 0
        return [0]*12
    
    def get_state(self):
        return np.array(self.states, dtype=np.float32)
    
    @property
    def action_space(self):
        return gym.spaces.Discrete(4)
    
    @property
    def observation_space(self):
        obs_space = gym.spaces.Box(
            low=np.zeros((5,12)),
            high=np.ones((5,12)),
            shape=(5,12)
        )
        return obs_space
    
    # Wrapper
    def state_wrapper(self, state):
        """返回当前每个 movement 的 occupancy
        """
        occupancy = state['tls'][self.tls_id]['last_step_occupancy']
        can_perform_action = state['tls'][self.tls_id]['can_perform_action']
        return occupancy, can_perform_action
    
    def reward_wrapper(self, states) -> float:
        """返回整个路口的排队长度的平均值
        """
        total_waiting_time = 0
        for _, veh_info in states['vehicle'].items():
            total_waiting_time += veh_info['waiting_time']
        return -total_waiting_time
    
    def info_wrapper(self, infos, occupancy):
        """在 info 中加入每个 phase 的占有率
        """
        movement_occ = {key: value for key, value in zip(self.movement_ids, occupancy)}
        phase_occ = {}
        for phase_index, phase_movements in self.phase2movements.items():
            phase_occ[phase_index] = sum([movement_occ['_'.join(phase.split('--'))] for phase in phase_movements])
        
        infos['phase_occ'] = phase_occ
        return infos

    def reset(self, seed=1) -> Tuple[Any, Dict[str, Any]]:
        """reset 时初始化 (1) 静态信息; (2) 动态信息
        """
        state =  self.env.reset()
        # 初始化路口静态信息
        self.movement_ids = state['tls'][self.tls_id]['movement_ids']
        self.phase2movements = state['tls'][self.tls_id]['phase2movements']
        # 处理路口动态信息
        occupancy, _ = self.state_wrapper(state=state)
        self.states.append(occupancy)
        state = self.get_state()
        return state, {'step_time':0}
    

    def step(self, action: int) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        can_perform_action = False
        while not can_perform_action:
            action = {self.tls_id: action} # 构建单路口 action 的动作
            states, rewards, truncated, dones, infos = super().step(action) # 与环境交互
            occupancy, can_perform_action = self.state_wrapper(state=states) # 处理每一帧的数据
            # 记录每一时刻的数据
            self.occupancy.add_element(occupancy)
        
        # 处理好的时序的 state
        avg_occupancy = self.occupancy.calculate_average()
        rewards = self.reward_wrapper(states=states) # 计算 vehicle waiting time
        infos = self.info_wrapper(infos, occupancy=avg_occupancy) # info 里面包含每个 phase 的排队
        self.states.append(avg_occupancy)
        state = self.get_state() # 得到 state

        return state, rewards, truncated, dones, infos
    
    def close(self) -> None:
        return super().close()