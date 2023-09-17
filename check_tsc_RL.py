'''
@Author: WANG Maonan
@Date: 2023-09-05 11:38:20
@Description: Traffic Signal Control Scenario
@LastEditTime: 2023-09-15 21:36:54
'''
import numpy as np
from loguru import logger
from tshub.utils.format_dict import dict_to_str
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger
from tshub.tshub_env.tshub_env import TshubEnvironment

from TSCEnvironment.tsc_env import TSCEnvironment
from TSCEnvironment.tsc_env_wrapper import TSCEnvWrapper

path_convert = get_abs_path(__file__)
set_logger(path_convert('./'))


if __name__ == '__main__':
    sumo_cfg = path_convert("./TSCScenario/J1/env/J1.sumocfg")
    tsc_scenario = TSCEnvironment(
        sumo_cfg=sumo_cfg, 
        num_seconds=1200,
        tls_id='J4', 
        tls_action_type='choose_next_phase',
        use_gui=True
    )
    tshub_env = TshubEnvironment(
    sumo_cfg=sumo_cfg,
    is_aircraft_builder_initialized=True, 
    is_vehicle_builder_initialized=True, 
    is_traffic_light_builder_initialized=True,
    tls_ids=['htddj_gsndj'],
    vehicle_action_type='lane', tls_action_type='next_or_not',
    use_gui=True)

    obs = tshub_env.reset()
    done = False
   # tsc_wrapper = TSCEnvWrapper(tsc_scenario)

    # Simulation with environment
    dones = False
    #tsc_wrapper.reset()
    while not dones:
        actions = {
        'vehicle': dict(),
        'tls': {'htddj_gsndj':0},
        }
        obs, reward, info, done = tshub_env.step(actions=actions)
        action = np.random.randint(4)
        #states, dones, infos = tsc_wrapper.step(action=action)
        #tls_available_actions = tsc_wrapper.get_available_actions() # 获得当前的动作
        # predict_state = tsc_wrapper.predict_future_scene(phase_index=1)
        logger.info(f"SIM: {infos['step_time']} \n{dict_to_str(states)}")
    tshub_env._close_simulation()
    #tsc_wrapper.close()