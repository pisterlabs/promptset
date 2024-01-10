import streamlit as st
import openai

# Set your OpenAI API key
openai.api_key =  st.secrets["OPENAI_API_KEY"] 

def ask_openai_about_image(image, question):

    # Encode the image as a base64 string
    image_bytes = open(image, "rb").read()
    image_base64 = image_bytes.encode("base64")

    # Create the OpenAI request
    request = {
        "file": image_base64,
        "question": question,
    }

    # Make the OpenAI request
    response = openai.Answer.create(**request)

    # Get the OpenAI response
    answer = response["answers"][0]

    # Return the OpenAI response
    return answer

def ask_openai_question(pdf_file, question):
  """Asks OpenAI a question about a PDF file.

  Args:
    pdf_file: A PDF file object.
    question: A string containing the question.

  Returns:
    A string containing the answer to the question.
  """

  # Extract the text from the PDF file.
  pdf_text = pdf_file.read()

  # Create an OpenAI completion request.
  completion_request = openai.CompletionRequest(
      prompt=question,
      context=pdf_text,
      engine="text-davinci"
  )

  # Get the completion response from OpenAI.
  completion_response = openai.Completion.create(completion_request)

  # Return the completion text.
  return completion_response.choices[0].text

def main():
  """The main function of the Streamlit app."""

  # Create a sidebar for user input.
  st.sidebar.header("Ask a question about the uploaded PDF:")

  # Allow the user to upload a PDF file.
  pdf_file = st.sidebar.file_uploader("Upload PDF file:")

  # Allow the user to enter a question.
  question = st.sidebar.text_input("Question:")

  # Ask OpenAI the question about the PDF file.
  answer = ask_openai_question(pdf_file, question)

  # Display the answer to the user.
  st.header("Answer:")
  st.write(answer)

if __name__ == "__main__":
  main()