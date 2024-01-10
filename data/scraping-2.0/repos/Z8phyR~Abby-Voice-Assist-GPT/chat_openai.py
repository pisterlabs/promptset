import openai
import time

model = "gpt-3.5-turbo"
system = {
    "helper": "I'm Abby, virtual assistant for my owner. I give short, but complete responses",
    "summarize": "I summarize in one to two sentences.",
}

def chat(assistant, user, chat_history=[]):
    try:
        messages = []
        messages.append({"role": "system", "content": system["helper"]})
        messages.append({"role": "system", "content": assistant})
        for message in chat_history[-8:]:
            messages.append({"role": "user", "content": message["input"]})
            messages.append({"role": "assistant", "content": message["response"]})
        messages.append({"role": "user", "content": user})
        # Uncomment for debug purposes
        # print(messages) 
        
        retry_count = 0
        while retry_count < 3:
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=0.6
                )
                status_code = response["choices"][0]["finish_reason"]
                assert status_code == "stop", f"The status code was {status_code}."
                return response["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"An error occurred while processing the chat request: {e}")
                print("Retrying...")
                time.sleep(1)  # Wait for 1 second before retrying
                retry_count += 1
        
        return "Oops, something went wrong. Please try again later."
    
    except Exception as e:
        print(f"An error occurred while processing the chat request: {e}")
        return "Oops, something went wrong. Please try again later."

def summarize(assistant_response):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": f"{system['summarize']}"},
                {"role": "assistant", "content": f"{assistant_response}"}
            ],
            max_tokens=250,
            temperature=0
        )
        
        status_code = response["choices"][0]["finish_reason"]
        assert status_code == "stop", f"The status code was {status_code}."
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"An error occurred while processing the summarize request: {e}")
        return "Oops, something went wrong. Please try again later."
