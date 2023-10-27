'''
@Author: WANG Maonan
@Author: PangAoyu
@Date: 2023-09-08 18:57:35
@Description: 使用训练好的 RL Agent 进行测试
@LastEditTime: 2023-09-14 14:08:03
'''
import torch
from langchain.chat_models import ChatOpenAI

from loguru import logger
from tshub.utils.format_dict import dict_to_str
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

from TSCEnvironment.tsc_env import TSCEnvironment
from TSCEnvironment.tsc_env_wrapper import TSCEnvWrapper
from TSCAssistant.tsc_assistant import TSCAgent


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
        model=config['OPENAI_API_MODEL'], 
        temperature=0.0,
        openai_api_key=openai_api_key, 
        openai_proxy=openai_proxy
    )

    # #########
    # Init Env
    # #########
    params = {
        'tls_id':'J1',
        'num_seconds':300,
        'sumo_cfg':sumo_cfg,
        'use_gui':True,
        'log_file':'./log_test/',
    }
    env = SubprocVecEnv([make_env(env_index=f'{i}', **params) for i in range(1)])#获取env信息
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

    tsc_agent = TSCAgent(llm=chat, verbose=True)
    while not dones:

        action, _state = model.predict(obs, deterministic=True)
        if sim_step>4:
            action=tsc_agent.agent_run(sim_step=sim_step, action=action, obs=obs, infos=infos), 
        obs, rewards, dones, infos = env.step(action)
        sim_step+=1

    env.close()
