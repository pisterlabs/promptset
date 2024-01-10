import openai
import os


llm = ChatOpenAI()
llm.predict("Hello, world!")

# Set the environment variables
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'

# Access the environment variables
api_key = os.environ.get('LANGCHAIN_API_KEY')
tracing_v2 = os.environ.get('LANGCHAIN_TRACING_V2')
endpoint = os.environ.get('LANGCHAIN_ENDPOINT')

# Use the API key in your code
print(f"API Key: {api_key}")
print(f"t_v2:{tracing_v2}")
print(f"endpoint: {endpoint}")

url = "http://localhost:8080/completions"
headers = {
  "Authorization": "Bearer sk-XXXXXXXXXXXXXXXXXXXX"
}
data = {
  "prompt": query,
  "temperature": 0.8
}
query = input("what is your query?")

chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello world"}])

def main():
    query = input("What do you want to ask? ")
    response = get_response(query)
    print(response)


if __name__ == "__main__":
    main()


