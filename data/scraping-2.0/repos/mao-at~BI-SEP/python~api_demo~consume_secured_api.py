import openai
import os
import time


SYSTEM_PROMPT= "you are a food classifier bot. You will respond 'junk' when the user prompt mentions any kind of junk food, and respond 'not junk' if all food mentioned are not junk food."
USER_PROMPT = "apple,pear,orange,banana"

def junk_or_not(system_prompt=SYSTEM_PROMPT, user_prompt=USER_PROMPT, retries=3):
    for i in range(retries):
        try:
            # openai.api_key= os.getenv("OPENAI_API_KEY")
            openai.api_key="abcdefg1234567890"
            completion = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": f"{system_prompt}"},
                    {"role": "user", "content": f"{user_prompt}"},
                ]
            )
            result = completion.choices[0].message.get("content")
            break  # If it gets to this line, no exception was raised, so break the loop
            
        except Exception as e:
            print(f"Attempt {i+1} failed with error: {str(e)}")
            print(f"retrying in {(i+1)**2} seconds...")
            time.sleep((i+1)**2)  # Pause before the next attempt
            
        if i == retries - 1:  # If this was the last attempt
            msg = f"ALL RETRIES FAILED"
            print(msg)
            return None
    print(f"result: {result}")
    return result

if __name__ == "__main__":
    junk_or_not()
