'''
@Author: WANG Maonan
@Date: 2023-09-08 17:45:54
@Description: 创建 TSC Env + Wrapper
@LastEditTime: 2023-09-08 18:25:42
'''
import gymnasium as gym
from utils.tsc_env import TSCEnvironment
from utils.tsc_wrapper import TSCEnvWrapper
from stable_baselines3.common.monitor import Monitor

def make_env(
        tls_id:str,num_seconds:int,sumo_cfg:str,use_gui:bool,
        log_file:str, env_index:int,
        ):
    def _init() -> gym.Env: 
        tsc_scenario = TSCEnvironment(
            sumo_cfg=sumo_cfg, 
            num_seconds=num_seconds,
            tls_ids=[tls_id], 
            tls_action_type='choose_next_phase',
            use_gui=use_gui,
        )
        tsc_wrapper = TSCEnvWrapper(tsc_scenario, tls_id=tls_id)
        return Monitor(tsc_wrapper, filename=f'{log_file}/{env_index}')
    
    return _init