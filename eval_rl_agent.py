'''
@Author: WANG Maonan
@Author: PangAoyu
@Description: 使用训练好的 RL Agent 进行测试
'''
import torch
from loguru import logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv

from utils.make_tsc_env import make_env

path_convert = get_abs_path(__file__)
logger.remove()

if __name__ == '__main__':
    # #########
    # Init Env
    # #########
    sumo_cfg = path_convert("./TSCScenario/SumoNets/train_four_345/env/train_four_345.sumocfg")
    params = {
        'tls_id':'J1',
        'num_seconds': 1600,
        'sumo_cfg':sumo_cfg,
        'use_gui':True,
        'log_file':path_convert('./log/'),
    }
    env = SubprocVecEnv([make_env(env_index=f'{i}', **params) for i in range(1)])
    env = VecNormalize.load(load_path=path_convert('./models/last_vec_normalize.pkl'), venv=env)
    env.training = False # 测试的时候不要更新
    env.norm_reward = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = path_convert('./models/last_rl_model.zip')
    model = PPO.load(model_path, env=env, device=device)

    # 使用模型进行测试
    obs = env.reset()
    dones = False # 默认是 False
    total_reward = 0

    while not dones:
        action, _state = model.predict(obs, deterministic=True)
        print('action',action)
        print('obs',obs)
        obs, rewards, dones, infos = env.step(action)
        total_reward += rewards
        
    env.close()
    print(f'Reward, {total_reward}.')