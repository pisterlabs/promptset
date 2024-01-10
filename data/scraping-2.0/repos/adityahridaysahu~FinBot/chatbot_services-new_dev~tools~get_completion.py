import openai,os
from dotenv import load_dotenv

load_dotenv()

def get_completion(prompt, task="classification"): 
    
    if task == "classification":
        openai.api_key = os.environ["OPENAI_API_KEY_1"]
    else :
        openai.api_key = os.environ["OPENAI_API_KEY_2"]

    messages = [{"role": "user", "content": prompt}]
    status = "processing"
    content = ""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0, 
        )

        status = "working"
        content = response.choices[0].message["content"]
    except (openai.error.RateLimitError):
        status = "Rate Limit Error"
        content = "Rate Limited. Please try again in a while."
    except:
        status = "Unknown Error"
        content = "System is not working, we are working on fixing the issue."

    return {
        "status" : status,
        "content" : content
    }