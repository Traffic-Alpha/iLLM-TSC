'''
@Author: WANG Maonan
@Date: 2023-09-08 15:49:30
@Description: 处理 TSCHub ENV 中的 state, reward
+ state: 5 个时刻的每一个 movement 的 queue length
+ reward: 路口总的 waiting time
LastEditTime: 2024-07-11 13:38:26
'''
import numpy as np
import gymnasium as gym
import random
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
    def __init__(self, env: Env, tls_id:str, max_states:int=1) -> None:
        super().__init__(env)
        self.tls_id = tls_id # 单路口的 id
        self.movement_ids = None
        self.phase_num = None # phase 数量
        self.llm_static_information = None # Static information, (1). Intersection Geometry; (2). Signal Phases Structure
        self.phase2movements = None
        
        #Dynamic Information
        self.state = None # 当前的 state
        self.last_state = None # 上一时刻的 state
        self.occupancy = OccupancyList()
    

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
            low=-1, 
            high=3,
            shape=(8,5)
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
        n = 1
        for _, veh_info in states['vehicle'].items():
            total_waiting_time += veh_info['waiting_time']
            n += 1
        return -(total_waiting_time/n)
    ''''''
    def info_wrapper(self, infos, states):
        """在 info 中加入每个 phase 的占有率
        """
        #movement_occ = {key: value for key, value in zip(self.movement_ids, occupancy)}
        phase_occ = {}
        phase_movement_occ={}
        phase_vehicles_num={}
        movement_ids=states['tls'][self.tls_id]['movement_ids']
        for phase_index, phase_movements in self.phase2movements.items():
            phase_movements=phase_movements.copy()

            sum_temp=0
            for phase_movement in phase_movements:
                #phase_movement=phase_movement.replace('--','_')
                index=movement_ids.index(phase_movement)
                sum_temp+=states['tls'][self.tls_id]['last_step_occupancy'][index]
                phase_movement_occ[states['tls'][self.tls_id]['movement_ids'][index]]=str(states['tls'][self.tls_id]['last_step_occupancy'][index])+'%'
                phase_vehicles_num[states['tls'][self.tls_id]['movement_ids'][index]]=int(states['tls'][self.tls_id]['jam_length_meters'][index])
            phase_occ[phase_index] = sum_temp
        infos['phase_occ'] = phase_occ
        infos['movement_occ']= phase_movement_occ
        infos['phase2movements']= states['tls'][self.tls_id]['phase2movements']
        infos['jam_length_meters']= phase_vehicles_num
        infos['last_step_vehicle_id_list']=states['tls'][self.tls_id]['last_step_vehicle_id_list']
        infos['movement_ids']=states['tls'][self.tls_id]['movement_ids']
        infos['information_missing'] = False
        infos['missing_id'] = None
        return infos

    def reset(self, seed=1) -> Tuple[Any, Dict[str, Any]]:
        """reset 时初始化 (1) 静态信息; (2) 动态信息
        """
        state =  self.env.reset()
        # 初始化路口静态信息
        self.movement_ids = state['tls'][self.tls_id]['movement_ids']
        self.phase2movements = state['tls'][self.tls_id]['phase2movements']
        obs=self._process_obs(state=state)
        # 处理路口动态信息
        occupancy, _ = self.state_wrapper(state=state)
        return obs, {'step_time':0}
    

    def step(self, action: int) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
       
        """将 step 的结果提取, 从 dict 提取为 list
        """
        can_perform_action = False
        while not can_perform_action:
            action = {self.tls_id: action} # 构建单路口 action 的动作
            states, rewards, truncated, dones, infos = super().step(action) # 与环境交互
            ccupancy, can_perform_action = self.state_wrapper(state=states) # 处理每一帧的数据
        
        #action = {self.tls_id: action} # 构建单路口 action 的动作 
        
        #states, rewards, truncated, dones, infos = super().step(action) # 与环境交互
        # 处理 obs
            
        observation = self._process_obs(state=states) 
        process_reward = self.reward_wrapper(states=states)
        #occupancy, can_perform_action = self.state_wrapper(state=states) # 处理每一帧的数据 
        infos = self.info_wrapper(infos, states=states)
        # 20% probability of missing information at the intersection
        flag = random.randint(0, 5)
        if ( flag % 5 == 0):
            observation[0:1,:] = -1 #observation[0,1] 信息缺失
            infos['information_missing'] = True
            infos['missing_id'] = 'E0--s'
            infos['movement_occ']['E0--s'] = '-1'
            infos['jam_length_meters']['E0--s'] = '-1'

        return observation, process_reward, truncated, dones, infos

    
    def _process_obs(self, state):
        """处理 observation, 将 dict 转换为 array.
        - 每个 movement 的 state 包含以下的部分, state 包含以下几个部分, 
            :[flow, mean_occupancy, max_occupancy, is_s, num_lane, mingreen, is_now_phase, is_next_phase]
        """
        phase_num = len(state['tls'][self.tls_id]['phase2movements']) # phase 的个数
        delta_time = state['tls'][self.tls_id]['delta_time']
        phase_movements = state['tls'][self.tls_id]['phase2movements'] # 得到一个 phase 有哪些 movement 组成的
        
        movement_directions = state['tls'][self.tls_id]['movement_directions']
        movement_ids=state['tls'][self.tls_id]['movement_ids']
        # 1. 获取每个lane的数据
        # 2. 获取每个pahse对应的lane的标号
        # 3. 获取每个phase的数据
        
        _observation_net_info = list() # 路网的信息
        '''
        for _movement_id, _movement in enumerate(phase_movements): # 按照 movment_id 提取
            for i in range(len(phase_movements[_movement_id])):
                phase_movements[_movement_id][i]=phase_movements[_movement_id][i].replace('--','_')
        '''
        # phase_movements {0: ['E0--s', '-E1--s'], 1: ['E0--l', '-E1--l'], 2: ['-E3--s', '-E2--s'], 3: ['-E3--l', '-E2--l']}
        for _movement_id, _movement in enumerate(phase_movements): # 按照 movment_id 提取
            for i in range(len(phase_movements[_movement_id])):
                movement_id=movement_ids.index(phase_movements[_movement_id][i])
                flow_mean_speed= state['tls'][self.tls_id]['last_step_mean_speed'][movement_id] # 上一次
                mean_occupancy = state['tls'][self.tls_id]['last_step_occupancy'][movement_id]# 占有率
                jam_length_meters = state['tls'][self.tls_id]['jam_length_meters'][movement_id]# 排队车数量
                jam_length_vehicle= state['tls'][self.tls_id]['jam_length_vehicle'][movement_id]
                is_now_phase =  state['tls'][self.tls_id]['this_phase'][movement_id] # now phase id
                if is_now_phase==True:
                    is_now_phase=1
                else:
                    is_now_phase=0
                #print("obs:",[flow, mean_occupancy, max_occupancy, is_s, num_lane, min_green, is_now_phase, is_next_phase])
                _observation_net_info.append([flow_mean_speed, mean_occupancy, jam_length_meters, jam_length_vehicle, is_now_phase])
        # 不是四岔路, 进行不全
        for _ in range(8 - len(_observation_net_info)):
            _observation_net_info.append([0]*5)
        '''
        for _movement_id, _movement in enumerate(phase_movements): # 按照 movment_id 提取
            for i in range(len(phase_movements[_movement_id])):
                phase_movements[_movement_id][i]=phase_movements[_movement_id][i].replace('_','--')
        '''
        obs = np.array(_observation_net_info, dtype=np.float32) # 每个 movement 的信息
        return obs
    
    def close(self) -> None:
        return super().close()