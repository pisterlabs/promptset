import openai

# Set your OpenAI API key here
openai.api_key = 'API KEY'

def generate_budget_recommendation(account_balance, spending_timeframe):
    prompt = f"You are a budget advising assistant. Given an account balance of ${account_balance:.2f} and a spending timeframe of {spending_timeframe} days, provide a budget recommendation for a college student."
    
    response = openai.Completion.create(
        engine = "text-davinci-002",
        prompt = prompt,
            temperature = 0.7,
            max_tokens = 2048
    )
    return response.choices[0].text.strip()
