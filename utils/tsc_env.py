'''
@Author: WANG Maonan
@Date: 2023-09-04 20:43:53
@Description: 信号灯控制环境
LastEditTime: 2024-07-11 13:35:34
'''
import gymnasium as gym

from typing import List, Dict
from tshub.tshub_env.tshub_env import TshubEnvironment

class TSCEnvironment(gym.Env):
    def __init__(self, sumo_cfg:str, num_seconds:int, tls_ids:List[str], 
                 tls_action_type:str, trip_info:str, use_gui:bool=False, ) -> None:
        super().__init__()

        self.tsc_env = TshubEnvironment(
            sumo_cfg=sumo_cfg,
            is_aircraft_builder_initialized=False, 
            is_vehicle_builder_initialized=True, # 用于获得 vehicle 的 waiting time 来计算 reward
            is_traffic_light_builder_initialized=True,
            tls_ids=tls_ids, 
            num_seconds=num_seconds,
            tls_action_type=tls_action_type,
            use_gui=use_gui,
            is_libsumo=(not use_gui), # 如果不开界面, 就是用 libsumo
            trip_info = trip_info # 记录仿真中所有 object 的指标
        )

    def reset(self):
        state_infos = self.tsc_env.reset()
        return state_infos
        
    def step(self, action:Dict[str, Dict[str, int]]):
        action = {'tls': action} # 这里只控制 tls 即可
 
        states, rewards, infos, dones = self.tsc_env.step(action)
        truncated = dones
        
        return states, rewards, truncated, dones, infos
    
    def close(self) -> None:
        self.tsc_env._close_simulation()