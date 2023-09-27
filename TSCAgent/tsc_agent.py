'''
@Author: WANG Maonan
@Date: 2023-09-04 20:51:49
@Description: traffic light control LLM Agent
@LastEditTime: 2023-09-15 20:53:32
'''
import numpy as np
from typing import List
from loguru import logger

from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.agents.tools import Tool
from langchain.memory import ConversationTokenBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate   #导入聊天提示模板
from langchain.chains import SimpleSequentialChain
from langchain.chains import LLMChain    #导入LLM链。

from TSCAgent.tsc_agent_prompt import SYSTEM_MESSAGE_SUFFIX

class TSCAgent:
    def __init__(self, 
                 llm:ChatOpenAI, 
                 verbose:bool=True,state:float=[] ) -> None:
        self.llm = llm # ChatGPT Model
        self.tools = [] # agent 可以使用的 tools
        self.state= state
        self.first_prompt=ChatPromptTemplate.from_template(   
                    "What is the best name to describe \
                     a company that makes {product}?"
                    )
        self.memory = ConversationTokenBufferMemory(llm=self.llm, max_token_limit=2048)
        self.chain_one = LLMChain(llm=llm, prompt=self.first_prompt)
        self.chain_two = LLMChain(llm=llm, prompt=self.first_prompt)
        self.agent =  SimpleSequentialChain(chains=[self.chain_one, self.chain_two],
                                             verbose=True) #构建路由链 还是构建顺序链， 需要构建提示模板
    def get_occupay(states):
        phase_list=get_phase()
        occupancy=states[:,1]
        occupancy_list=np.zeros(phase_list.shape[0])
        for i in range(0,phase_list.shape[0]):
            occupancy_list[i]=(occupancy*phase_list[i]).sum()

        occupancy_list=occupancy_list/2
        
        return occupancy_list
    
    def get_phase():
        phases_4=[[1, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 1]]
        
        return np.array(phases_4)

    
    def agent_run(self, sim_step:float, action:int=0, obs:float=[]):
        """_summary_

        Args:
            tls_id (str): _description_
            sim_step (float): _description_

        1. 现在的每个action的 movement 
        2. 这个phase 所包含的movement 的平均占有率
        3. 判断动作是否可行
        """
        logger.info(f"SIM: Decision at step {sim_step} is running:")
        # r = self.agent.run(
        #     f'Now you are a traffic signal light with id {tls_id}. Please analysis the efficient of the available actions you can make one by one.'
        # )
        # 需要返回上一步每个 movement 的排队长度
        r = self.agent.run(action,obs)
        
        print(r)
        print('-'*10)