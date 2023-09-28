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
from langchain.chains import SequentialChain
from langchain.chains import LLMChain    #导入LLM链。

from TSCAgent.tsc_agent_prompt import SYSTEM_MESSAGE_SUFFIX
from TSCAgent.tsc_agent_prompt import (
    SYSTEM_MESSAGE_SUFFIX,
    SYSTEM_MESSAGE_PREFIX,
    HUMAN_MESSAGE,
    FORMAT_INSTRUCTIONS,
    TRAFFIC_RULES,
    DECISION_CAUTIONS,
    HANDLE_PARSING_ERROR
)

class TSCAgent:
    def __init__(self, 
                 llm:ChatOpenAI, 
                 verbose:bool=True,state:float=[] ) -> None:
        self.tls_id='J1'
        self.llm = llm # ChatGPT Model
        self.tools = [] # agent 可以使用的 tools
        self.state= state
        #self.file_callback = create_file_callback(path_convert('../agent.log'))
        self.first_prompt=ChatPromptTemplate.from_template(   
                    'You can ONLY use one of the following actions: \n action:0 action:1 action:2 action:3'
    
                    )
        self.first_prompt=ChatPromptTemplate.from_template(   
                    " The action is {Action}, the ocuupance is [1. 0. 0. 0.] \n To check decision safety: "
                    )
        self.memory = ConversationTokenBufferMemory(llm=self.llm, max_token_limit=2048)
        self.chain_one = LLMChain(llm=llm, prompt=self.first_prompt)
        self.chain_two = LLMChain(llm=llm, prompt=self.first_prompt,output_key="sfaty")
        self.agent =  SequentialChain(chains=[self.chain_one, self.chain_two],
                                      input_variables=["Action"],
                                      #output_variables=["sfaty"],
                                      verbose=True) #构建路由链 还是构建顺序链， 需要构建提示模板
        
    def get_phase(self):
        phases_4=[[1, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 1]]
        
        return np.array(phases_4)
   
    def get_occupancy(self, states):
        phase_list=self.get_phase()
        occupancy=states[:,:,1]
        print('states',states)
        occupancy_list=np.zeros(phase_list.shape[0])
        for i in range(0,phase_list.shape[0]):
            occupancy_list[i]=(occupancy*phase_list[i]).sum()

        occupancy_list=occupancy_list/2
        
        return occupancy_list


    
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
        occupancy=self.get_occupancy(obs)
        print('occupancy',occupancy)
        Action=action
        Occupancy=occupancy 
        r = self.agent.run(
            f"""
            You, the 'traffic signal light', are now controlling the traffic signal in the junction with ID `{self.tls_id}`. You have already control for {sim_step*5} seconds.
            The decision you made LAST time step was `{Action}`. Your explanation was `{Occupancy}`. 
            Please make decision for the traffic signal light.You have to work with the **Static State** and **Action** of the 'traffic light'. Then you need to analyze whether the current 'Action' is reasonable based on the intersection occupancy rate, and finally output your decision.
            There are the actions that will occur and their corresponding phases：
        
                - Action 0： Phase 0
                - Action 1： Phase 1
                - Action 2： Phase 2
                - Action 3： Phase 3

            Here are your attentions points:
            {DECISION_CAUTIONS}
            
            Let's take a deep breath and think step by step. Once you made a final decision, output it in the following format: \n
            ```
            Final Answer: 
                "decision":{{"Traffic light decision-making judgment, whether the Action is reasonable in the current state"}},
                "expalanations":{{"your explaination about your decision, described your suggestions to the Crossing Guard"}}
            ``` \n
            """,
        )
        
        print(r)
        print('-'*10)