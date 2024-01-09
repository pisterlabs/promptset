import streamlit as st

# Import the LangChain library
import langchain

# Load the AI model
model = langchain.load_model("model.pkl")

# Create a function to get the feedback from the AI model
def get_feedback(statement):
    # Get the predictions from the AI model
    predictions = model.predict(statement)

    # Create a list of feedback
    feedback = []
    for prediction in predictions:
        feedback.append(prediction["feedback"])

    return feedback

# Create a function to display the feedback
def display_feedback(statement):
    # Get the feedback from the AI model
    feedback = get_feedback(statement)

    # Display the feedback to the user
    st.write("Here is the feedback from the AI model:")
    st.write(feedback)

# Create a main function
def main():
    # Get the personal statement from the user
    statement = st.text_input("Enter your personal statement:")

    # Display the feedback to the user
    display_feedback(statement)

# Run the main function
if __name__ == "__main__":
    main()

# print("Start!")
# load_dotenv(find_dotenv())

# # pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))

# dataset_path = "./dataset.txt"
# loader = TextLoader(dataset_path)
# comments = loader.load_and_split()

# embeddings = OpenAIEmbeddings(model_name="ada")
# vectordb = Chroma.from_documents(comments, embedding=embeddings, persist_directory=".")
# vectordb.persist()
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# # Assuming that GPT-4 is used for grammar, structure, and fact-checking
# # and Claude is used for providing tips and encouraging students to do their own research
# grammar_llm = OpenAI(temperature=0.8)
# tips_llm = Claude(temperature=0.8)

# grammar_qa = ConversationalRetrievalChain.from_llm(grammar_llm, vectordb.as_retriever(), memory=memory)
# tips_qa = ConversationalRetrievalChain.from_llm(tips_llm, vectordb.as_retriever(), memory=memory)



# st.title('AI Statement Reviewer')

# user_input = st.text_area("Enter your personal statement here:")

# if st.button('Get feedback'):
#     grammar_result = grammar_qa({"question": user_input})
#     tips_result = tips_qa({"question": user_input})
#     st.write("Grammar and Structure Feedback:")
#     st.write(grammar_result["answer"])
#     st.write("Tips and Recommendations:")
#     st.write(tips_result["answer"])

