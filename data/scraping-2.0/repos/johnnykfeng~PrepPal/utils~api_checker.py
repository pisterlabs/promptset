import openai

def api_test(api_key, model="gpt-3.5-turbo"): 
    try:
        output = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            api_key=api_key,
            temperature=0,
            messages=[{"role": "user", "content": "Hello"}]
        )
    except Exception as e:
        print(f"{str(e)}")
        return False
    
    return True

