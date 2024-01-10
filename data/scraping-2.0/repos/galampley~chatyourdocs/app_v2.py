import os
import openai
import streamlit as st
from functions_v2 import save_uploaded_file, learn_document, Answer_from_documents, create_connection
from database_v2 import initialize_database, save_conversation_history, clear_conversation_history  # Importing the database initialization function
from dotenv import load_dotenv
from custom_exceptions import LearningError, DatabaseError, verify_file_read, log_error

# Load the environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

initialize_database()

def clear_database():
    initialize_database()
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM documents")
    cursor.execute("DELETE FROM embeddings")
    cursor.execute("DELETE FROM conversation")
    conn.commit()
    conn.close()

# Introducing a new session state variable 'new_session' to track the start of a new session
if 'new_session' not in st.session_state:
    st.session_state.new_session = True
else:
    st.session_state.new_session = False

if st.session_state.new_session:
    clear_database()
    st.session_state.db_initialized = True  # Mark that the database has been initialized for this session
    st.session_state.new_session = False  # Set 'new_session' to False for subsequent reruns in the same session

def main():
    # Initialize or retrieve the conversation history from the session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    if 'show_read_button' not in st.session_state:
        st.session_state.show_read_button = True  # Initialize to True initially

    st.title("ChatYourDocs")
    st.write("Using OpenAI models for embedding and inference. Using SQLite db with cosine similarity, instead of vector db, for easy setup. SQLite is single file based and has less CPU requirement.")
    st.write("Vector db is better for similarity search as unstructured dataset gets larger and larger.")

    uploaded_files = st.file_uploader("Choose files to upload (PDF or TXT)", type=["pdf", "txt"], accept_multiple_files=True)
    read_files = []  # List to keep track of the names of the read files
    #conversation_history = get_conversation_history()
    
    for uploaded_file in uploaded_files:
        st.session_state.show_read_button = True  # Reset the show_read_button state variable to True for each new file
        
        # Determine the file type using the file extension (Suggested Approach 1)
        file_type = os.path.splitext(uploaded_file.name)[1][1:]
    
        # file_type = uploaded_file.type.split('/')[-1]

        read_button = st.empty() # Create an empty space for the button

        if st.session_state.show_read_button and read_button.button(f"Read {uploaded_file.name}"):
            save_uploaded_file(uploaded_file)
            file_path = uploaded_file.name

            try:
                # Check if the file is read successfully
                if not verify_file_read(file_path):
                    st.write(f"Failed to read {uploaded_file.name}. Please check the file format.")
                    continue
                
                # Learn from the uploaded document and store in the database
                learn_document(file_path, uploaded_file.name, file_type)
                read_files.append(uploaded_file.name)
                st.write(f"{uploaded_file.name} reading completed! Now you may ask a question")

                st.session_state.show_read_button = False
                
            except FileNotFoundError as e:
                st.write(f"{uploaded_file.name} not found: {str(e)}")
            except LearningError as e:  # Custom exception for learning errors
                st.write(f"An error occurred while learning from {uploaded_file.name}: {str(e)}")
            except DatabaseError as e:  # Custom exception for database errors
                st.write(f"An error occurred while storing {uploaded_file.name} in the database: {str(e)}")
            except Exception as e:
                st.write(f"An unexpected error occurred while processing {uploaded_file.name}: {str(e)}")
                # Optionally log the error
                log_error(e)
                
            finally:
                os.remove(uploaded_file.name)

    st.write("Documents read: ", ", ".join(read_files))  # Display the names of the read files

    user_input = st.text_input("Enter your question:")
    
    if st.button("Ask"):
        if user_input:
            response = Answer_from_documents(user_input)
            save_conversation_history(user_input, response)
            st.write(response)
            # Update conversation history in the session state
            st.session_state.conversation_history.append({"user": user_input, "bot": response})

    st.subheader("Conversation History")
    for conversation in reversed(st.session_state.conversation_history):
        st.markdown(f"**You:**\n{conversation['user']}\n\n**Bot:**\n{conversation['bot']}")

    if st.button("Clear Conversation History"):
        clear_conversation_history()  # This line clears the conversation history in the database
        st.session_state.conversation_history = []  # This line clears the conversation history in the session state
        st.write("Conversation history has been cleared.")
        st.experimental_rerun()  # This line forces Streamlit to rerun the script immediately


if __name__ == "__main__":
    main()
