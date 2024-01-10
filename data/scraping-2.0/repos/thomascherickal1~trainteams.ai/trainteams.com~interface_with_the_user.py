import gradio as gr
from langchain.vectorstores import VectorStore
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import DeepLake
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

embedding_function = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')

# Initialize vector store  
vectorstore = DeepLake("./deeplake_db",embedding_function)
 

# Define the LLM
llm = OpenAI(temperature=0)  

# Create the conversation chain
chain = ConversationChain(llm=llm, vectorstore=vectorstore)



#Prompt formate to the LLM
prompt = PromptTemplate(
    input="Human: {human_input}\nAssistant: ",
    output="Human: "
)


def chat(input):
    # Chatbot logic
    response = chain.predict(prompt, vectorstore=vectorstore, input=input)["output"] 
    return response

iface = gr.Interface(
    fn=chat,
    inputs=gr.inputs.Textbox(lines=2, placeholder="Ask a question..."), 
    outputs=gr.outputs.Textbox(),
    title="Rust Programming Chatbot",
    description="I'll teach you the Rust Programming Language!" 
)

if __name__ == "__main__":
    iface.launch()