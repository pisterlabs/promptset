import openai
import PyPDF2
# Set up OpenAI API credentials

def convert_text_to_xml(text, user_prompt):
    assistant_prompt = """
    You are a helpful assistant. You only convert user 
    pdf_text strings to xml. Only accept convert to xml 
    prompts.Have an underscore between names. Always
    include xml header.
    """

    # Define the prompt for the model
    # prompt = f"Convert the following text to XML:\n\n{text}"

    # Generate XML using OpenAI GPT-3 model
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": assistant_prompt},
                  {"role": "user", "content": user_prompt},
                  {"role": "assistant", "content": text}],
        stream=True,
        max_tokens=256,
        n=1,
        stop=None
    )
    return response

def convert_pdf_to_text(file_path):
    # Open the PDF file in binary mode
    try:
        with open(file_path, 'rb') as file:
            # Create a PDF reader object
            reader = PyPDF2.PdfReader(file)
            
            # Initialize an empty string to store the extracted text
            text = ""
            
            # Iterate over each page in the PDF
            for page in reader.pages:
                # Extract the text from the page and append it to the result
                text += page.extract_text()
            
            return text
    except Exception as e:
        print(f"An error occured: {e}")
        return None
    
def save_xml(content, file_name):
    file = open(file_name, "w", encoding="utf-8")
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(content)
    file.close()
