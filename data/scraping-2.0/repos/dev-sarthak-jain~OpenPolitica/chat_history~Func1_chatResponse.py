from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
import os
from dotenv import load_dotenv
load_dotenv()




def initialize_openai_model():
    # Initialize the OpenAI model with your API key
    # Here, we use the `OpenAI` class from Langchain
    return OpenAI(openai_api_key=os.getenv("apikey"))

def generate_response(user_input, past_convo):
    # Construct a conversation chain with the model, prompt, and memory
    model = initialize_openai_model()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "You are a helpful and inquisitive government assistant who seeks to understand and empathize to discover the root of people's problems."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )

    # Create an LLM chain with the model, prompt, and memory
    conversation = LLMChain(llm=model, prompt=prompt, memory=memory)

    # Update memory with past conversation
    for exchange in past_convo:
        memory.save_context({"input": exchange["user"]}, {"output": exchange["AI"]})
        
    # Run the conversation chain
    response = conversation({"question": user_input})

    # Extract only the AI's response from the 'text' field
    ai_response = response['text'].split('\n')[-1]

    # If the response contains the "AI:" prefix, remove it
    if ai_response.startswith("AI: "):
        ai_response = ai_response[4:]

    return ai_response

'''`
def main():
     # Replace with your actual API key
    model = initialize_openai_model()

    past_convo = [
        {"user": "Query 1", "AI": "Answer 1"},
        {"user": "Query 2", "AI": "Answer 2"},
        {"user": "Query 3", "AI": "Answer 3"}
    ]

    while True:
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            break

        response = generate_response(user_input, past_convo, model)
        print("AI:", response)

        # Update the conversation history
        past_convo.append({"user": user_input, "AI": response})

if __name__ == "__main__":
    main()
    
    
'''