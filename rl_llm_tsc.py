'''
@Author: WANG Maonan
@Date: 2023-09-08 18:57:35
@Description: 使用训练好的 RL Agent 进行测试
@LastEditTime: 2023-09-14 14:08:03
'''
import torch
import langchain
import numpy as np
from langchain.chat_models import ChatOpenAI

from loguru import logger
from tshub.utils.format_dict import dict_to_str
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

from TSCEnvironment.tsc_env import TSCEnvironment
from TSCEnvironment.tsc_env_wrapper import TSCEnvWrapper
from TSCAgent.tsc_agent import TSCAgent
from TSCAgent.custom_tools import GetAvailableActions, GetCurrentOccupancy

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv

from utils.readConfig import read_config
from utils.make_tsc_env import make_env

path_convert = get_abs_path(__file__)
logger.remove()

if __name__ == '__main__':
    sumo_cfg = path_convert("./TSCScenario/SumoNets/train_four_345/env/train_four_345.sumocfg")
    # Init Chat
    config = read_config()
    openai_proxy = config['OPENAI_PROXY']
    openai_api_key = config['OPENAI_API_KEY']
    chat = ChatOpenAI(
        model='gpt-3.5-turbo-16k', temperature=0.0,
        openai_api_key=openai_api_key, 
        openai_proxy=openai_proxy
    )
    '''
    tsc_scenario = TSCEnvironment(
        sumo_cfg=sumo_cfg, 
        num_seconds=1200,
        tls_id='J4', 
        tls_action_type='choose_next_phase',
        use_gui=True
    )
    '''
    #tsc_wrapper = TSCEnvWrapper(tsc_scenario)

    # #########
    # Init Env
    # #########
    params = {
        'tls_id':'J1',
        'num_seconds':2600,
        'sumo_cfg':sumo_cfg,
        'use_gui':True,
        'log_file':'./log_test/',
    }
    env = SubprocVecEnv([make_env(env_index=f'{i}', **params) for i in range(1)])
    env = VecNormalize.load(load_path=path_convert('./models/last_vec_normalize.pkl'), venv=env)
    env.training = False # 测试的时候不要更新
    env.norm_reward = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = path_convert('./models/last_rl_model.zip')
    model = PPO.load(model_path, env=env, device=device)


    
    # 使用模型进行测试
    dones = False # 默认是 False
    sim_step = 0
    obs = env.reset()
    tools = [
        GetAvailableActions(state=obs, env=env),
        GetCurrentOccupancy(state=obs, env=env), # 查看当前时刻的拥堵情况
    ]
    tsc_agent = TSCAgent(llm=chat, verbose=True)
    while not dones:

        action, _state = model.predict(obs, deterministic=True)
        #print('action',action)
        tsc_agent.agent_run(sim_step=sim_step, action=action, obs=obs)
        obs, rewards, dones, infos = env.step(action)
        #print('obs',obs.shape, obs)
        sim_step+=1

    env.close()
