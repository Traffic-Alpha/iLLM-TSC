'''
@Author: PangAoyu
@Date: 2023-02-15 14:33:49
@Description: 训练 RL 模型，训练的时候使用多个环境
@LastEditTime: 2023-02-24 23:20:10
'''
import os
import argparse
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from aiolos.utils.get_abs_path import getAbsPath
from aiolos.trafficLog.initLog import init_logging
pathConvert = getAbsPath(__file__)

from env import makeENV
from models import scnn, ernn, eattention, ecnn, inference, inference_scnn, inference_eattention, inference_ecnn
from create_params import create_params
from utils.lr_schedule import linear_schedule
from utils.env_normalize import VecNormalizeCallback, VecBestNormalizeCallback

def experiment(
        net_name,net_env,n_stack, n_delay, model_name, num_cpus
    ):
    assert model_name in ['scnn', 'ernn','eattention','ecnn','inference','inference_scnn','inference_eattention','inference_ecnn'], f'Model name error, {model_name}'   #增加模型
    # args
    N_STACK = n_stack # 堆叠
    N_DELAY = n_delay # 时延
    NUM_CPUS = num_cpus
    EVAL_FREQ = 2000 # 一把交互 700 次
    SAVE_FREQ = EVAL_FREQ*2 # 保存的频率
    MODEL_PATH = pathConvert(f'./results/models_test/{model_name}/{net_env}_{net_name}_{N_STACK}_{N_DELAY}/')
    LOG_PATH = pathConvert(f'./results/log/{model_name}/{net_env}_{net_name}_{N_STACK}_{N_DELAY}/') # 存放仿真过程的数据
    TENSORBOARD_LOG_DIR = pathConvert(f'./results/tensorboard_temp_logs/{model_name}_{net_env}_{net_name}/')
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    if not os.path.exists(TENSORBOARD_LOG_DIR):
        os.makedirs(TENSORBOARD_LOG_DIR)
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)

    train_params = create_params(
        is_eval=False, 
        N_DELAY=N_DELAY, N_STACK=N_STACK, LOG_PATH=LOG_PATH, net_env=net_env,net_name=net_name,
    )
    eval_params = create_params(
        is_eval=True, 
        N_DELAY=N_DELAY, N_STACK=N_STACK, LOG_PATH=LOG_PATH, net_env=net_env,net_name=net_name,
    )
    # The environment for training
    env = SubprocVecEnv([makeENV.make_env(env_index=f'{N_STACK}_{N_DELAY}_{i}', **train_params) for i in range(NUM_CPUS)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True) # 进行标准化 #先不标准版测试一下
    # The environment for evaluating
    eval_env = SubprocVecEnv([makeENV.make_env(env_index=f'evaluate_{N_STACK}_{N_DELAY}', **eval_params) for i in range(1)])
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True) # 进行标准化 #先不做标准化测试一下
    eval_env.training = False # 测试的时候不要更新
    eval_env.norm_reward = False
    action_space=eval_env.action_space.n 

    # ########
    # callback
    # ########
    save_vec_normalize = VecBestNormalizeCallback(save_freq=1, save_path=MODEL_PATH)
    eval_callback = EvalCallback(
        eval_env, # 这里换成 eval env 会更加稳定
        eval_freq=EVAL_FREQ,
        best_model_save_path=MODEL_PATH,
        callback_on_new_best=save_vec_normalize,
        verbose=1
    ) # 保存最优模型
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=MODEL_PATH,
    ) # 定时保存模型
    vec_normalize_callback = VecNormalizeCallback(
        save_freq=SAVE_FREQ,
        save_path=MODEL_PATH,
    ) # 保存环境参数
    callback_list = CallbackList([eval_callback, checkpoint_callback, vec_normalize_callback])


    # ###########
    # start train
    # ###########
    feature_extract = {
        'scnn': scnn.SCNN,
        'ernn': ernn.ERNN,
        'eattention': eattention.EAttention,
        'ecnn':ecnn.ECNN,
        'inference':inference.Inference,
        #'ernn_P':ernn_P.ERNN_P,
        #'ernn_C':ernn_C.ERNN_C,
        'inference_scnn':inference_scnn.Inference_SCNN,
        'inference_eattention':inference_eattention.Infer_EAttention,
        'inference_ecnn':inference_ecnn.Infer_ECNN,
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device",device)
    policy_kwargs = dict(
        features_extractor_class=feature_extract[model_name],
        features_extractor_kwargs=dict(features_dim=32,action_space=action_space), # features_dim 提取的特征维数
    )
    model = PPO(
                "MlpPolicy", env, verbose=True, 
                policy_kwargs=policy_kwargs, learning_rate=linear_schedule(3e-4), 
                tensorboard_log=TENSORBOARD_LOG_DIR, device=device
            )
    model.learn(total_timesteps=1e5, tb_log_name=f'{N_STACK}_{N_DELAY}', callback=callback_list) # log 的名称

    # #########
    # save env
    # #########
    env.save(f'{MODEL_PATH}/vec_normalize.pkl')


if __name__ == '__main__':
    init_logging(log_path=pathConvert('./log'), log_level=0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--stack', type=int, default=1)
    parser.add_argument('--delay', type=int, default=0)
    parser.add_argument('--cpus', type=int, default=3) # 同时开启的仿真数量
    parser.add_argument('--net_env', type=str, default='train_four_345')
    parser.add_argument('--net_name', type=str, default='4phases.net.xml')
    parser.add_argument('--model_name', type=str, default='scnn')
    args = parser.parse_args()

    experiment(
        net_env=args.net_env, net_name=args.net_name , n_stack=args.stack, n_delay=args.delay,
        model_name=args.model_name, num_cpus=args.cpus
    )