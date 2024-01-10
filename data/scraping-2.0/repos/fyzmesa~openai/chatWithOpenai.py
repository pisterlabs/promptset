import openai

openai.api_key = ""

message = {"role":"user", "content": input("\n*************************************************\nWelcome to this chat fyzmesa, to exit, send \"exit\".\n*************************************************\n\n[fyzmesa ~]# ")};

conversation = [{"role": "system", "content": "DIRECTIVE_FOR_gpt-3.5-turbo"}]

while(message["content"]!="Goodbye"):
    conversation.append(message)
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=conversation) 
    message["content"] = input(f"\n[ChatGPT ~]# : {completion.choices[0].message.content} \n\n[fyzmesa ~]# : ")
    print()
    conversation.append(completion.choices[0].message)
