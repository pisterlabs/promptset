import openai
openai.api_key = "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"

def getresp(text):
    completion =  openai.Completion.create(
    engine="text-davinci-002",
    max_tokens = 1024,
    n=2,
    prompt="Give me keywords related to"+text
    )
    textsum=''
    for choice in completion.choices:
        textsum+=choice.text,"\n\n"
        
    return textsum

