from hermetic.agents.openai_chat_agent import OpenAIChatAgent
import openai

NAME = 'multichoice'
MODEL = 'gpt-3.5-turbo'
class Multichoice(OpenAIChatAgent):
    """
    Multichoice 
    """

    def __init__(self, env):
        # Call the super class constructor
        super().__init__(environment = env, id=NAME, model_name=MODEL)
        env.add_agent(NAME, self)
        self.pm = self.env.prompt_mgr
        openai.api_key = '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'

   
    # Our superclass takes care of the details, all we need to do
    # is to update the message history
    def update_message_history(self, inp): 
        # bind to the system message
        sys_msg = self.pm.bind('multichoice_system').render()
        multichoice = self.pm.bind('multichoice_query').render(query=inp)


        # Let's add our system message to the message history
        #self.message_history = [SystemMessage(content=sys_msg.render())]
        self.message_history = [{
            'role': 'system',
            'content': sys_msg
        },
        {
            'role': 'user',
            'content': multichoice
        }]
