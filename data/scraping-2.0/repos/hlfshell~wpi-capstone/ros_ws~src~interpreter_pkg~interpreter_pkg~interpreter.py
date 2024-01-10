import os

import rclpy
from ament_index_python.packages import get_package_share_directory
from example_interfaces.msg import String
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               SystemMessagePromptTemplate)
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from rclpy.node import Node

data_dir = os.path.join(get_package_share_directory("interpreter_pkg"))
OpenAI.api_key = os.environ["OPENAI_API_KEY"]
MODEL_TO_USE = "gpt-4"


class LLM_Object:
    '''This is a class for an LLM object.  It is used 3 times
    by the interpreter.  The primary is the refiner
    This checks the initial request for a direct object
    request and if not, asks clarifying questions in a chat
    format to obtain a response from which an object can be 
    ascertained.  The secondisntance is a judge that takes
    the user input/response and asks an LLM with different 
    context whether it is a searchable object.  Respionds with yes or no.
    The final is an extractor which extracts and returns the object and 
    its modifiers.  Each instance uses its own context.'''
    def __init__(self, context_file: str, model: str, temperature: int):
        with open(os.path.join(data_dir, context_file), 'r') as file:
            context = file.read().replace('\n', '')
        self.context = context
        self.temp = temperature
        self.chat = ChatOpenAI(model_name=model)
        self.system_message = SystemMessage(content=self.context)
        self.rolling_context = [self.system_message]

    def reset_context(self):
        self.rolling_context = [self.system_message]


class InterpreterNode(Node):

    '''This is the iterpreter node designed to interpret
    via chat if necessary the target object that the user
    wants.  It publishes the target_message.  As part of user 
    inpout, it subscribes to user_request and in response
    publishes clarifying_question if necessary'''

    def __init__(self):
        super().__init__('interpreter')
        # self.get_logger().info("Terp Started")
        # define elements
        self.refiner = LLM_Object("refiner_context.prompt", MODEL_TO_USE, 0)
        self.judger = LLM_Object("judge_context.prompt", MODEL_TO_USE, 0)
        self.extractor = LLM_Object("extractor_context.prompt",
                                    MODEL_TO_USE, 0)
        self.new_request = True
        self.attempts_to_understand = 0
        self.target = ""
        self.last_suggestion = ""

        # define publishing
        self.target_publisher_ = self.create_publisher(String, 'target', 10)
        self.target_id_loop()

    def target_id_loop(self):
        while True:
            request = self.initiate_request()
            while self.target == "":
                request = self.get_request(request)
            self.broadcast_target(self.target)

    def initiate_request(self):
        request = input('Robot: What can I get for you?')
        return request

    def get_request(self, request: str):
        '''
        This gets the user response to a chat string to
        check if the response is yes-that means user is confirming
        your last item, if so, the last ai message is the target.
        '''
        response = ""

        if request.strip().lower() == "yes" and self.new_request is False:
            # user is responding to a direct suggestion not making a first request
            self.target = extract_request(self.extractor, self.last_suggestion)
        else:
            self.new_request = False
            # So if not responding yes, use judge to see if the item is
            # actionable in request
            judgement = judge(self.judger, request)
            if judgement == "yes":
                # identify simple target
                self.target = extract_request(self.extractor, request)
                
            # if we've been going round, restart conversation
            elif self.attempts_to_understand == 5:
                # reset rolling context
                self.refiner.reset_context()
                self.attempts_to_understand = 0
                # Chat the request to start over
                print("Robot: I'm sorry i dont inderstand. Could we start over?")
            else:
                # we dont know what the target is yet so ask another question
                response = self.ask_clarifying_question(request)
            return response

    def ask_clarifying_question(self, user_msg: str):
        self.refiner.rolling_context.append(HumanMessage(content=user_msg))
        # LLM call to generate clarifying qustion
        ai_question_to_user = self.refiner.chat(self.refiner.rolling_context)
        self.last_suggestion = ai_question_to_user.content
        # self.get_logger().info("The response from the LLM is "+ai_question_to_user.content)
        self.refiner.rolling_context.append(AIMessage(content=ai_question_to_user.content))
        self.attempts_to_understand += 1
        # clarifying question for user
        response = input("Robot: "+ai_question_to_user.content)
        self.last_suggestion = ai_question_to_user.content
        return response

    def broadcast_target(self, target: str):
        '''
        callback for publishing target name
        '''
        self.get_logger().info("Broadcasting target: " + self.target)
        target_msg = String()
        target_msg.data = target
        self.target_publisher_.publish(target_msg)
        self.reset()

    def reset(self):
        self.new_request = True
        self.attempts_to_understand = 0
        self.target = ""
        self.refiner.reset_context()
        self.last_suggestion = ""


def judge(judger: LLM_Object, request: String) -> str:
    '''
    this funtion takes a request and determines if a searchable object is in it
    returns yes or no
    '''
    system_message_prompt = SystemMessagePromptTemplate.from_template(judger.context)
    human_template = f"Am I specifically requesting an object, if so what? : {request}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chain = LLMChain(llm=judger.chat, prompt=chat_prompt)
    response = chain.run(question=request)
    response = cleanup(response)
    return response


def extract_request(extractor: LLM_Object, verbose_request: str) -> str:
    '''
    this function takes a long request and extracts the target object
    returning the object and its descriptors
    '''
    system_message_prompt = SystemMessagePromptTemplate.from_template(extractor.context)
    human_template = f"Isolate what I am requesting: {verbose_request}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chain = LLMChain(llm=extractor.chat, prompt=chat_prompt)
    response = chain.run(question=verbose_request)
    response = cleanup(response)
    return response


def cleanup(response):
    response = response.replace("\n", "")
    response = response.replace("Answer: ", "")
    response = response.replace(".", "")
    response = response.strip()
    response = response.lower()
    response = response.replace("a ", "")
    return response


def main(args=None):
    rclpy.init(args=args)
    interpreter = InterpreterNode()
    rclpy.spin(interpreter)
    interpreter.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
