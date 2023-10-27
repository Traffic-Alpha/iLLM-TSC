'''
@Author: WANG Maonan
@Author: PangAoyu
@Description: 基于规则空的TSC模型, 选择等待车辆最多的路口通行
'''
import numpy as np
from loguru import logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger
from stable_baselines3.common.env_checker import check_env
from utils.make_tsc_env import make_env

path_convert = get_abs_path(__file__)
set_logger(path_convert('./'))


if __name__ == '__main__':
    log_path=log_path = path_convert('./log/')
    sumo_cfg = path_convert("./TSCScenario/SumoNets/train_four_345/env/train_four_345.sumocfg")
    tsc_env_generate = make_env(
        tls_id='J1',
        num_seconds=3600,
        sumo_cfg=sumo_cfg, 
        use_gui=True,
        log_file=log_path,
        env_index=0,
    )
    tsc_env = tsc_env_generate()

    # Check Env
    #print(tsc_env.observation_space.sample())
    #print(tsc_env.action_space.n)
    print('-------------------------------------')
    check_env(tsc_env)
    
    def get_action(states):
        phase_list=get_phase()
        occupancy=states[:,1]
        occupancy_list=np.zeros(phase_list.shape[0])
        for i in range(0,phase_list.shape[0]):
            occupancy_list[i]=(occupancy*phase_list[i]).sum()

        max_index=occupancy_list.argmax()
        
        return max_index
    
    def get_phase():
        phases_4=[[1, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 1]]
        
        return np.array(phases_4)
    # Simulation with environment
    dones = False
    tsc_env.reset()
    total_reward=0
    action=0
    while not dones:
        
        states, rewards, truncated, dones, infos = tsc_env.step(action=action)
        print('states',states.shape)
        total_reward+=rewards
        action = get_action(states)
        logger.info(f"SIM: {infos['step_time']} \n+State:{states}; \n+Reward:{rewards}.")
    
    print('total reward',total_reward)
    tsc_env.close()

