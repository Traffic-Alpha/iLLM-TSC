'''
@Author: WANG Maonan
@Date: 2023-09-08 15:57:34
@Description: 测试 TSC Env 环境
@LastEditTime: 2023-09-13 16:33:41
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
        use_gui=False,
        log_file=log_path,
        env_index=0,
    )
    tsc_env = tsc_env_generate()

    # Check Env
    #print(tsc_env.observation_space.sample())
    #print(tsc_env.action_space.n)
    print('-------------------------------------')
    check_env(tsc_env)

    # Simulation with environment
    dones = False
    tsc_env.reset()
    while not dones:
        action = np.random.randint(4)
        print('-------------------------------------*****************************')
        states, rewards, truncated, dones, infos = tsc_env.step(action=action)
        logger.info(f"SIM: {infos['step_time']} \n+State:{states}; \n+Reward:{rewards}.")
    tsc_env.close()