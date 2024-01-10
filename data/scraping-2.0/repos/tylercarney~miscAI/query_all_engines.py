import openai
import keyring

# Retrieve API key from the system keyring
openai.api_key = keyring.get_password("system", "openai_api_key")

# Function to query a specific engine
def query_engine(engine_id, content):
    try:
        # Try older completions API
        response = openai.Completion.create(
            model=engine_id,
            prompt=content,
            max_tokens=50
        )
        reply = response.choices[0].text.strip()
        print(f"Response from {engine_id}: {reply}")
    except openai.error.OpenAIError as e:
        try:
            # If older API fails, try modern chat-based API
            response = openai.ChatCompletion.create(
                model=engine_id,
                messages=[{"role": "user", "content": content}]
            )
            reply = response.choices[0].message.content.strip()
            print(f"Response from {engine_id}: {reply}")
        except openai.error.OpenAIError as e:
            print(f"Failed to query {engine_id}: {e}")

# Retrieve all available engines
engine_list_response = openai.Engine.list()
engines = [engine['id'] for engine in engine_list_response['data']]

content = "What model are you?"

# Loop over all engines and query each
for engine in engines:
    query_engine(engine, content)
