import openai
while True:
    prompt_value=input("\nPrompt :>> ")
    if(prompt_value=="exit"):            #if you want to close Press "exit"
        break
    else:
        # enter api key here for authientication
        openai.api_key="key_vaalue"  
        output=openai.Completion.create(prompt=prompt_value,model="text-davinci-003",max_tokens=2000)
        output_text=output["choices"][0]["text"]
        print(output_text[1:)
