from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from modules.templates import ChatPromptTemplate
from modules.llm_library import LLMLibrary
from langchain.memory import ConversationBufferMemory



class ChatHandler():

    def generic_user_message(message,conversation_contexts):

        channel_id = message['channel']
        user_id = message['user']
        context_key = f"{channel_id}-{user_id}"  # Unique key for each user-channel combination
            # Retrieve or initialize the conversation memory and history
        if context_key not in conversation_contexts:
            conversation_contexts[context_key] = {
                "memory": ConversationBufferMemory(memory_key="chat_history", output_key="answer", return_messages=True, max_token_limit=1024),
                "history": "",
            }
        user_memory = conversation_contexts[context_key]["memory"]
        history = conversation_contexts[context_key]["history"]
        user_input = message['text']
        user_chain= LLMChain(
                llm=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.0),
                prompt=ChatPromptTemplate.generic_prompt_template(),
                verbose=True,
            )          
        return user_chain, user_input, history, user_memory, conversation_contexts, context_key
    
    def doc_question_command(body, conversation_contexts):
        llmlibrary = LLMLibrary()        
        channel_id = body['channel_id']
        user_id = body['user_id']
        context_key = f"{channel_id}-{user_id}"
        prompt = ChatPromptTemplate.doc_question_prompt_template()
        if context_key not in conversation_contexts:
            conversation_contexts[context_key] = {
                "memory": ConversationBufferMemory(memory_key="chat_history", output_key="answer", return_messages=True, max_token_limit=1024),
                "history": "",
            }
        user_memory = conversation_contexts[context_key]["memory"]
        question = body['text']
        conversation = llmlibrary.doc_question(user_memory, prompt, question)
        #print(f"Conversation: {conversation}")
        return question, conversation
    
    def terraform_question_command(body, question, conversation_contexts):
        llmlibrary = LLMLibrary()
        channel_id = body['channel_id']
        user_id = body['user_id']
        context_key = f"{channel_id}-{user_id}"
        prompt = ChatPromptTemplate.terraform_prompt_template()
        if context_key not in conversation_contexts:
            conversation_contexts[context_key] = {
                "memory": ConversationBufferMemory(memory_key="chat_history", output_key="answer", return_messages=True, max_token_limit=1024),
                "history": "",
            }
        user_memory = conversation_contexts[context_key]["memory"]
        history = conversation_contexts[context_key]["history"]
        user_chain = LLMLibrary.terraform_question(user_memory, prompt)
        return user_chain, history, user_memory, conversation_contexts, context_key