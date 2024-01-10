import streamlit as st
import os
import openai
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from langchain.text_splitter import CharacterTextSplitter


index_name = "chatbot"
PINECONE_KEY = st.secrets["PINECONE_KEY"]
if PINECONE_KEY is None:
    raise ValueError("Pinecone key not found. Please set the PINECONE_KEY environment variable.")
  
pinecone.init(api_key=PINECONE_KEY, environment="gcp-starter")

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
if OPENAI_API_KEY is None:
    raise ValueError("OpenAI key not found. Please set the OPENAI_API_KEY environment variable.")
  
os.environ["TOKENIZERS_PARALLELISM"] = "false"
TOKENIZERS_PARALLELISM = "false"
model_name = 'text-embedding-ada-002'

system_prompt = (
"This is a knowledgeable Tourism Assistant designed to provide visitors with "
"information, recommendations, and tips for exploring and enjoying their destination. "
"The assistant is familiar with a wide range of topics including historical sites, "
"cultural events, local cuisine, accommodations, transportation options, and hidden gems. "
"It offers up-to-date and personalized information to help tourists make the most of their trip."
)
# Check if the index exists
if index_name not in pinecone.list_indexes():
    # If the index doesn't exist, create it
    pinecone.create_index(index_name=index_name, metric='cosine', shards=1)
index = pinecone.GRPCIndex(index_name)

# Streamlit app
def main():
    st.title("Question Answering with Pinecone and GPT-3")

    with st.spinner("Loading dataset..."):
        # Load dataset
        dataset = load_dataset("jayantdocplix/falcon-small-test-dataset")
        df = dataset['train'].to_pandas().head(50)

    # Initialize text splitter and tokenizer
    text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(GPT2TokenizerFast.from_pretrained("gpt2-xl"), chunk_overlap=50)

    # Initialize the session state for history if it doesn't exist
    if 'history' not in st.session_state:
        st.session_state.history = []
        
    # Process dataset and upsert data into Pinecone (only if not already done)
    if 'data_upserted' not in st.session_state:
        with st.spinner("Processing data and upserting to Pinecone..."):
            df['text'] = df['text'].apply(text_splitter.split_text)
            flattened_texts = [text for sublist in df['text'].tolist() for text in sublist]
            
            embed = OpenAIEmbeddings(api_key=OPENAI_API_KEY).embed_documents(flattened_texts)
            # Upsert data into Pinecone
            items_to_upsert = [{"id": str(index), "values": embedding, "metadata": {"text": text}} for index, (embedding, text) in enumerate(zip(embed, flattened_texts))]
            index.upsert(items_to_upsert)
            st.session_state.data_upserted = True

    # Streamlit input for user question
    question = st.text_input("Please enter your question:")
    response = None  # Initialize response

    if st.button("Submit") and question:
        try:
            # Query Pinecone and generate response
            context = query_pinecone(question)
            response = generate_response(context, question)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            response = "I'm sorry, there was an error generating a response."
            # Output the exception for debugging purposes
            st.exception(e)  # Or use `st.exception(e)` to display the full traceback

    # Output the response and append it to the history
    if response:
        st.session_state.history.append((question, response))
        # Rerun the app to update the display
        st.rerun()
    
    # Display the history in a scrollable container
    with st.container():
        for i, (q, r) in enumerate(st.session_state.history):
            st.write(f"Q{i+1}: {q}")
            st.write(f"R{i+1}: {r}")
            st.write("---")  # Separator line

    # Output the response and append it to the history
    #st.session_state.history.append((question, response))

def query_pinecone(question, top_k=5):
       # Generate the embedding for the question
       question_embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY).embed_query(question)

       # Query the Pinecone index with the question embedding to retrieve the top_k closest matches
       query_results = index.query(
           vector=question_embedding, 
           top_k=top_k, 
           include_metadata=True
       )

       # Extract the text metadata from the query results
       return [item['metadata']['text'] for item in query_results['matches']]

def generate_response(context, question):
    # Format the context and question into a prompt for the language model
    prompt = "\n".join([" ".join(sub_list) for sub_list in context])

    # Limit the length of the prompt
    prompt = prompt[:3096]  # or any other number less than 4097

    prompt = prompt + "\n\n" + question
    # Generate the response using OpenAI GPT-3.5 Turbo
    chat_completion = openai.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt,}
        ],
        model="gpt-3.5-turbo",
    )
    # Return the content of the chat completion, which is the model's response, stripped of leading/trailing whitespace
    if chat_completion.choices[0].message.content:
        return chat_completion.choices[0].message.content.strip()
    else:
         return "I'm sorry, I couldn't generate a response."

# Run the Streamlit app
if __name__ == "__main__":
    main()
