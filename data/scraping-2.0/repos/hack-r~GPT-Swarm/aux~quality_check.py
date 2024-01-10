from prompts.prompts import quality_prompt

import openai
import os

def check_task_output(task_prompt, task_response):
    # Assuming you have set your OPENAI_API_KEY as an environment variable
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # You can add additional logic here to check the quality
    # For now, let's just pass it to GPT-3 for a simple check
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or another model if preferred
            messages=[
                {"role": "system", "content": quality_prompt},
                {"role": "user", "content": task_prompt},
                {"role": "assistant", "content": task_response},
                {"role": "user", "content": "Check the quality of the above output."},
            ]
        )
        quality_check_passed = "quality check passed" in response.choices[0].message['content']
        if quality_check_passed:
            return task_response  # If passed, return original response
        else:
            return response.choices[0].message['content']  # If failed, return corrected response
    except Exception as e:
        print(f"Error occurred during quality check: {e}")
        raise
