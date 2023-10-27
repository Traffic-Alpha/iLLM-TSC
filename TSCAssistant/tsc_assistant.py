'''
@Author: Pang Aoyu
@Date: 2023-09-04 20:51:49
@Description: traffic light control LLM Agent
@LastEditTime: 2023-09-15 20:53:32
'''
import numpy as np
from typing import List
from loguru import logger

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationTokenBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate   #导入聊天提示模板
from langchain.chains import LLMChain    #导入LLM链。
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

from TSCAssistant.tsc_agent_prompt import SYSTEM_MESSAGE_SUFFIX
from TSCAssistant.tsc_agent_prompt import (
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
        self.second_prompt=ChatPromptTemplate.from_template(   
                    " The action is {Action}, Your explanation was `{Occupancy}` \n To check decision safety: "
                    )
        self.memory = ConversationTokenBufferMemory(llm=self.llm, max_token_limit=2048)
        
        self.chain_one = LLMChain(llm=llm, prompt=self.first_prompt)
        
        self.chain_two = LLMChain(llm=llm, prompt=self.second_prompt,output_key="safety")
        '''
        self.assessment =  SequentialChain(chains=[self.chain_one, self.chain_two],
                                      input_variables=["Action"],
                                      #output_variables=["sfaty"],
                                      verbose=True) #构建路由链 还是构建顺序链， 需要构建提示模板
        '''
        memory = ConversationBufferMemory()
        self.assessment = ConversationChain( llm=llm, memory = memory, verbose=True )
        self.phase2movements={
                        "Phase 0": ["E0_s","-E1_s"],
                        "Phase 1": ["E0_l","-E1_l"],
                        "Phase 2": ["-E3_s","-E2_s"],
                        "Phase 3": ["-E3_l","-E2_l"],
                    } 
        self.movement_ids=["E0_s","-E1_s","-E1_l","E0_l","-E3_s","-E2_s","-E3_l","-E2_l"]                   
    def get_phase(self):
        phases_4=[[1, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 1]]
        
        return np.array(phases_4)
   
    def get_occupancy(self, states):
        phase_list=self.get_phase()
        occupancy=states[:,:,1]
        occupancy_list=np.zeros(phase_list.shape[0])
        for i in range(0,phase_list.shape[0]):
            occupancy_list[i]=(occupancy*phase_list[i]).sum()

        occupancy_list=occupancy_list/2
        
        return occupancy_list

    def get_rescue_movement_ids(self, last_step_vehicle_id_list, movement_ids):
        """获得当前 Emergency Vehicle 在什么车道上
        """
        rescue_movement_ids = []
        for vehicle_ids, movement_id in zip(last_step_vehicle_id_list, movement_ids):
            for vehicle_id in vehicle_ids:
                if 'rescue' in vehicle_id:
                    rescue_movement_ids.append(movement_id)
        return rescue_movement_ids
    
    def agent_run(self, sim_step:float, action:int=0, obs:float=[], infos: list={}):
        """_summary_

        Args:
            tls_id (str): _description_
            sim_step (float): _description_

        1. 现在的每个action的 movement 
        2. 这个phase 所包含的movement 的平均占有率
        3. 判断动作是否可行
        """
        logger.info(f"SIM: Decision at step {sim_step} is running:")
        occupancy=self.get_occupancy(obs)
        Action=action[0]
        step_time=infos[0]['step_time']
        step_time=int(step_time)
        Occupancy=infos[0]['movement_occ']
        jam_length_meters=infos[0]['jam_length_meters']
        movement_ids=infos[0]['movement_ids']
        last_step_vehicle_id_list=infos[0]['last_step_vehicle_id_list']
        rescue_movement_ids=self.get_rescue_movement_ids(last_step_vehicle_id_list,movement_ids)
        print('rescue_movement_ids',rescue_movement_ids)
        review_template="""
        decision:  Traffic light decision-making judgment  whether the Action is reasonable in the current state.
        explanations: Your explanation about your decision, described your suggestions to the Crossing Guard. The analysis should be as detailed as possible, including the possible benefits of each action.
        final_action: ONLY the number of Action you suggestion, 0, 1, 2 or 3
        
        Format the output as JSON with the following keys:  
        decision
        expalanations
        final_action


        observation: {observation}
        {format_instructions}
        """
        prompt = ChatPromptTemplate.from_template(template=review_template)
        decision = ResponseSchema(name="decision",
                             description="Judgment whether the RL Agent's Action is reasonable in the current state, you can only choose reasonable or unreasonable. ")
        expalanations = ResponseSchema(name="expalanations",
                             description="Your explaination about your decision, described your suggestions to the Crossing Guard. The analysis should be as detailed as possible, including the possible benefits of your action.")        
        final_action = ResponseSchema(name="final_action",
                             description="ONLY the number of Action you final get, 0, 1, 2 or 3")
        response_schemas = [decision, 
                    expalanations,
                    final_action]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        #prompt_template = ChatPromptTemplate.from_template(ans_template)
        #messages = prompt_template.format_messages(text=observation)
        observation=(f"""
            You, the 'traffic signal light', are now controlling the traffic signal in the junction with ID `{self.tls_id}`.
            The step time is:"{step_time}"
            The decision RL Agent make thie step is `Action：{Action}`. 
            The vehicles mean occupancy of each movement is:`{Occupancy}`. 
            The number of cars waiting in each movement is：`{jam_length_meters}`. 
            Now these movements exist emergency vehicles： `{rescue_movement_ids}`. 
            Phase to Movement: '{self.phase2movements}'
            Please make decision for the traffic signal light.You have to work with the **Static State** and **Action** of the 'traffic light'. Then you need to analyze whether the current 'Action' is reasonable based on the intersection occupancy rate, and finally output your decision.
            There are the actions that will occur and their corresponding phases：
        
                - Action 0： Phase 0
                - Action 1： Phase 1
                - Action 2： Phase 2
                - Action 3： Phase 3

            Here are your attentions points:
            {DECISION_CAUTIONS}
            
            Let's take a deep breath and think step by step. Once you made a final decision, output it in the following format: \n 
             
            """)
        messages = prompt.format_messages(observation=observation, format_instructions=format_instructions)
        print(messages[0].content)
        r = self.llm(messages)
        output_dict = output_parser.parse(r.content)
        print(r.content)
        final_action=output_dict.get('final_action')
        print('-'*10)
        final_action=int(final_action)
        return final_action