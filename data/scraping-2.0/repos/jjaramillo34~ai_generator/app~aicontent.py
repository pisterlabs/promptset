import openai
import app.configuration as configuration
openai.api_key = configuration.OPENAI_API_KEY

    """_summary_
    
    """
def open_ai_query(query):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=query,
        temperature=0.9,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
    
    if 'choices' not in response:
        return 'Error: API request failed. You may have exceeded your request quota.'
    else:
        if len(response.choices) == 0:
            return 'Error: API response had no choices. You may have exceeded your request quota.'
        else:
            print(response)
            return response.choices[0].text
        