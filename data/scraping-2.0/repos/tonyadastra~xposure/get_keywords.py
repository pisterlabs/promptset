import openai
import json


def extract_keywords(prompt, max_tokens=100):
    openai.api_key = "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"
    messages = [{"role": "user", "content": prompt}]
    functions = [
        {
            "name": "extract_highlight_keywords",
            "description": "Define the domain of this topic, key highlights of the domain, and do not account for background information. Stay focused on the primary actions of the domain. Identify the key protagonist of the domain. If it is not a human, identify the secondary protagonist but the human specifically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "the domain of the video, such as 'basketball' or 'soccer'"
                    },
                    "key_highlights": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The key moments to look for in this type of video, such as 'slam dunk' or '3 pointer' or 'goal'",
                    },
                },
                "required": ["key_highlights", "domain"],
            },
        }
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        temperature=0.0,
        function_call={
            "name": "extract_highlight_keywords",
        }
    )
    response_message = response["choices"][0]["message"]["function_call"]["arguments"]
    
    print(response_message)
    
    # This function is called from main.py
    # It returns a list of keywords
    # Example: ['successful slam dunk', 'successful 3 pointer']
    return json.loads(response_message)



if __name__ == "__main__":
    # print(extract_keywords("I want to see the top momemnts of a world cup game"))
    
    print(extract_keywords("top moments of a dance performance"))
    
    
    # input sports name
    # key highlights they're interested in
    
    
