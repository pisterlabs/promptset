```python
# Code_Samples/API_Monitoring_Tools.py

import openai
import time

# Set your OpenAI API key
openai.api_key = 'your-api-key'

# Define a function to monitor the usage of the API
def monitor_api_usage():
    # Get the usage details
    usage = openai.API.list_usage()

    # Print the usage details
    print("API Usage Details:")
    print(f"Total tokens: {usage['total_tokens']}")
    print(f"Total requests: {usage['total_requests']}")
    print(f"Total compute: {usage['total_compute']}")

    # Check if the usage exceeds a certain limit
    if usage['total_tokens'] > 1000000:
        print("Warning: You have exceeded 1,000,000 tokens this month.")

# Define a function to monitor the health of the API
def monitor_api_health():
    try:
        # Make a test API call
        response = openai.Completion.create(engine="text-davinci-002", prompt="Hello, world!", max_tokens=5)

        # If the API call is successful, the API is healthy
        print("API is healthy.")
    except Exception as e:
        # If the API call fails, the API is not healthy
        print(f"API is not healthy. Error: {str(e)}")

# Monitor the API usage and health every 5 minutes
while True:
    monitor_api_usage()
    monitor_api_health()

    # Wait for 5 minutes
    time.sleep(300)
```
