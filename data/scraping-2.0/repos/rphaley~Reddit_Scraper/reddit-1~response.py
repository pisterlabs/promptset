import openai, os

def generateResponse(config, author, post_body, debug):
    # Get prompt from config file
    prompt = config.get('AISettings', 'UserPrompt')
    prompt = prompt.format(author=author, post_body=post_body)
    system_prompt = config.get('AISettings', 'SystemPrompt')

    # Set OpenAI API key
    openai.organization = config.get('AISettings', 'OpenAIOrgazination')
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Make GPT API request
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    )

    # Check if the request was successful
    if response['object'] == 'error':
         if debug == 1: print(f"[-] Failed to make API request. Error message: {response['error']['message']}")
    else:
        if debug == 1: print("[+] API request successful!")
        if debug == 1: print(response)
        if debug == 1: print(response['usage'])
        if debug == 1: print("API response:")
        if debug == 1: print(response['choices'][0]['message']['content'])
        return response['choices'][0]['message']['content']


