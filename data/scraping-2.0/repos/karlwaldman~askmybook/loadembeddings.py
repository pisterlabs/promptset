import openai
import pandas as pd

# Set the OpenAI API key
openai.api_key = "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"

# Load the pre-trained model and its embeddings
#model = openai.Model.from_pretrained('openai/text-davinci-002')
model = openai.GPT(engine="davinci", temperature=0.5, max_tokens=150)
# Read the CSV file into a DataFrame
df = pd.read_csv("embeddings.csv")

# Extract the content of page column from the DataFrame
text = df["content of page"].tolist()

# Use the transform method to get the embeddings for the text
embeddings = model.transform(text)

print(embeddings)
