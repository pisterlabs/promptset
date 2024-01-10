import spacy
import openai
import gradio as gr


# Loading the pre-trained English language model in spaCy
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Function to obtain word embeddings
def get_word_embeddings(sentence):
    tokens = nlp(sentence)
    embeddings = [token.vector for token in tokens]
    return embeddings

# Function to get the chatbot response
def get_response(message):
    openai.api_key = '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'
    
    # Obtain word embeddings for the user message
    embeddings = get_word_embeddings(message)
    
    # Convert the embeddings to a string representation
    embeddings_str = [str(embedding) for embedding in embeddings]
    
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo-16k',
        messages=[
            {"role":"system", "content":"You are a polite, helpful postpartum care assistant who answers anything related to POSTPARTUM CARE, for other questions politely say you cannot answer, if you have any concern related then I can help."},
            {"role":"user","content": message},
            {"role":"assistant","content":""},

            {"role": "user", "content": message + " "+ " ".join(embeddings_str) }
        ],
        max_tokens=200,
        temperature=0.9,
    )
    return response["choices"][0]["message"]["content"]

def chatbot_interface(input_text):
    response = get_response(input_text)
    return response

# code to create the Gradio app interface
iface = gr.Interface(
    fn=chatbot_interface,
    inputs="text",
    outputs="text",
    layout="vertical",
    title="Postpartum Care Chatbot",
    description="Ask any postpartum care-related questions!",
    theme='HaleyCH/HaleyCH_Theme',
    # inputs_layout="textarea",
    # outputs_layout="textarea",
    examples=[
        ["Hi, how can I take care of my newborn's skin?"],
        ["What is postpartum depression? What are the signs and symptoms?"],
        ["What activities are safe to do in the first few days? Which activities should I avoid?"],
        ['Are there certain foods or beverages I should avoid when breastfeeding?'],
        ["What should I do to prevent deep vein thrombosis?"],
        ['What should I do to help prevent post-delivery infections?']
    ],
)

# Start the Gradio app
if __name__ == "__main__":
    print("Welcome to the Postpartum Care Chatbot!")
    print("How can I assist you today?")
    iface.launch(share=True)
