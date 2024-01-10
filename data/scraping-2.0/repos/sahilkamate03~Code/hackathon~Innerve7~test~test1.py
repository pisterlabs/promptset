import openai
import pandas as pd

# Authenticate with your OpenAI API key
openai.api_key = "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"

# Read in the data from a CSV file
df = pd.read_csv('test/test.csv')

# Define the fine-tuning parameters
model = "text-davinci-002"
temperature = 0.5
max_tokens = 1024
epochs = 3
batch_size = 4

# Define the prompt to use for fine-tuning
prompt = (
    "The following is a conversation between a customer and a customer service representative:\n\n"
)

# Loop over each row in the data and fine-tune the model
for i, row in df.iterrows():
    
    # Define the input text as the prompt followed by the customer's message
    input_text = prompt + row['customer_message']
    
    # Define the target text as the customer service representative's response
    target_text = row['cs_rep_response']
    
    # Fine-tune the GPT model using the input text and target text
    for epoch in range(epochs):
        response = openai.Completion.create(
            engine=model,
            prompt=input_text,
            temperature=temperature,
            max_tokens=max_tokens,
            n = 1,
            stop=None,
            frequency_penalty=0,
            presence_penalty=0,
            batch_size=batch_size,
            response_format="json"
        )
        
        # Extract the generated text from the API response
        output_text = response.choices[0].text
        
        # Calculate the loss between the generated text and the target text
        loss = openai.Completion.create(
            engine=model,
            prompt=input_text + "\nTarget: " + target_text + "\nOutput: " + output_text,
            temperature=0,
            max_tokens=0,
            n = 1,
            stop=None,
            frequency_penalty=0,
            presence_penalty=0,
            response_format="json"
        ).choices[0].text.split(":")[-1].strip()
        
        # Print out the loss and generated text
        print(f"Epoch {epoch+1} - Loss: {loss}, Output: {output_text.strip()}")
        
        # Concatenate the input text and generated text to use as the input for the next epoch
        input_text += output_text
