# Import required libraries  
import streamlit as st  
import openai  
import langchain  
import boto3  
  
# Set up OpenAI API key  
openai.api_key = "YOUR_OPENAI_API_KEY"  
  
# Set up LangChain API key  
langchain_api_key = "YOUR_LANGCHAIN_API_KEY"  
langchain_client = langchain.Client(langchain_api_key)  
  
# Set up AWS S3 credentials  
aws_access_key_id = "YOUR_AWS_ACCESS_KEY_ID"  
aws_secret_access_key = "YOUR_AWS_SECRET_ACCESS_KEY"  
s3_bucket = "YOUR_S3_BUCKET_NAME"  
s3_client = boto3.client(  
    "s3",  
    aws_access_key_id=aws_access_key_id,  
    aws_secret_access_key=aws_secret_access_key,  
)  
  
# Define chatbot function  
def chatbot(prompt, model, lang="en"):  
    # Generate response using OpenAI  
    response = openai.Completion.create(  
        engine=model,  
        prompt=prompt,  
        max_tokens=1024,  
        n=1,  
        stop=None,  
        temperature=0.7,  
    )  
    # Get text from OpenAI response  
    text = response.choices[0].text.strip()  
    # Translate text to specified language using LangChain  
    translated_text = langchain_client.translate(text, source="en", target=lang)  
    # Return translated text  
    return translated_text  
  
# Define save message to S3 function  
def save_message_to_s3(message):  
    # Write message to S3 bucket  
    s3_client.put_object(Bucket=s3_bucket, Key="chatbot_messages.txt", Body=message)  
  
# Define load messages from S3 function  
def load_messages_from_s3():  
    # Read messages from S3 bucket  
    response = s3_client.get_object(Bucket=s3_bucket, Key="chatbot_messages.txt")  
    messages = response["Body"].read().decode("utf-8")  
    return messages  
  
# Create Streamlit app  
def app():  
    # Set app title and description  
    st.set_page_config(page_title="Chatbot Demo", page_icon=":robot_face:", layout="wide")  
    st.title("Chatbot Demo")  
    st.write("This is a simple demo of a chatbot that integrates with LangChain and AWS S3.")  
      
    # Load chat history from S3  
    chat_history = load_messages_from_s3() if s3_client else ""  
      
    # Get user input  
    prompt = st.text_input("User: ", "")  
      
    # Generate chatbot response  
    if prompt:  
        # Add user input to chat history  
        chat_history += f"\nUser: {prompt}"  
        # Save updated chat history to S3  
        save_message_to_s3(chat_history) if s3_client else None  
        # Generate chatbot response  
        response = chatbot(chat_history, "davinci")  
        # Add chatbot response to chat history  
        chat_history += f"\nChatbot: {response}"  
        # Save updated chat history to S3  
        save_message_to_s3(chat_history) if s3_client else None  
        # Display chatbot response  
        st.text_area("Chatbot: ", value=response, height=200)  
          
# Run Streamlit app  
if __name__ == "__main__":  
    app()  
