from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from django.conf import settings

# Global variable to store a single conversation memory
global_conversation_memory = ConversationBufferWindowMemory(k=20, memory_key="chat_history", input_key="human_input")




# Replace with your actual API key
api_key = settings.OPENAI_API_KEY



def reset_conversation_memory():
    global global_conversation_memory
    global_conversation_memory = ConversationBufferWindowMemory(k=20, memory_key="chat_history", input_key="human_input")


def history_init(first_message):
    global global_conversation_memory
    print("First message: ",first_message)
    global_conversation_memory.chat_memory.add_ai_message(first_message)
    print(global_conversation_memory)


def chatbot_response(system_message, temperature, message, file=None):
    global global_conversation_memory

    

    # Determine the appropriate prompt template
    prompt = system_message + """
        Current conversation:
        {chat_history}

        Human: {human_input}
        File: ```{file}```
        AI Assistant:
        """

    # Initialize the GPT-4 model
    gpt_4 = ChatOpenAI(model_name="gpt-4", openai_api_key=api_key, temperature=temperature, max_tokens=500)

    # Create a prompt template with the necessary input variables
    prompt_template = PromptTemplate(input_variables=["chat_history", "human_input", "file"], template=prompt)

    memory = global_conversation_memory

    # Create a conversation chain with the Langchain components using the shared global conversation memory
    llm_chain = LLMChain(llm=gpt_4, memory=memory, prompt=prompt_template)

    # Generate the reply using the chatbot reply function
    reply = llm_chain.predict(human_input=message, file=file)

    print(memory)

    return reply
