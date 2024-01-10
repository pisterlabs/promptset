from langchain import OpenAI, ConversationChain

#Calling the LLM
llm = OpenAI(temperature=0)
#initializing the conversation 
conversation = ConversationChain(llm=llm, verbose=True)
# We will build the sentence in a way that is person don't put stop then it will generate the flow
converstaion=""
print("write stop to end the converation")
while True:
    convertaion=input("Me: \t")
    if convertaion.lower() == "stop":
        print("happy to help you")
        break
    else:
     print(conversation.predict(input=convertaion))