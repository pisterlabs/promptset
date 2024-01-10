
    

    if request.method == 'POST':
        user_query = request.POST.get('message')
        user_chat.append(user_query)
        history.add_user_message(user_query)

        global chat_summary
        from langchain.chat_models import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate
        from langchain.chains import LLMChain
        import os

        os.environ['OPENAI_API_KEY'] = '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'

        llm = ChatOpenAI(temperature=0.6)
        prompt = ChatPromptTemplate.from_template(
            """You are banking grivance complaint loger. 
            
            try to take as much as complaint inforamation possible.


            if you think for a given complaint you have collected these data \
            user name, bank name, transaction id (if required), cards details or any other details for that specific issue user is facing.
            then return [COMPLAINT-FILABLE]

            

            other wise ask user for all other required informations for solving that problem \
            Return JSON object formatted to look like :
            
            {{{{
                "chat-reply": string \ reply tobe send to the user telling about all the informations which are required\
                "facts" :   also add one more json object which \
                            store  important information or user in the form of json or dictionary for future reference which it gathers during converation.\
                            these data should be correct and precise
                "STATUS": string \ This should be "More info required"
            }}}}
            
            
            all chat with summary : {chat}"""
        )
   

        # creating a chat summary
        from langchain.chat_models import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate
        from langchain.chains import LLMChain
        llm = ChatOpenAI(temperature=0.6)
        prompt = ChatPromptTemplate.from_template(
            """Make a chat summary : \
            keeping all the facts correct and without missing any important information \
            
            also add one more json object which \
            store these important information or user in the form of json or dictionary for future reference. And update this dictionary with\
            the users chat and ai chat. this json object is added so that no information is missed.

            here is the complete chat. Make sure that you remember and write names and other details entered by the user in this summary : {chat}?"""
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        chat =  chat_summary + user_chat[-1] + ai_chat[-1]
        chat_summary = chain.run(chat)
        
        # response = ask_openai(message)
        # response = "hi this my response."
        # print(response)
        # chat = Chat(user=request.user, message=message, response=response, created_at=timezone.now())
        # chat.save()
        if response == "[COMPLAINT-FILABLE]":
            status = "done"
            print("moving it to create a vector search prompt")
            vector_search()
            
        print("\n oopes trying to get more data")
        status = "more-data-req"
        print(response)
        # dictionary_response = json.loads(response)
        # response_ai = dictionary_response["chat-reply"]
        # user_query = input(dictionary_response["chat-reply"] + "\nuser : ")
        print("CHAT SUMmARY : " + chat_summary)
        # complaint_completion(str(history.messages) + user_query)
        # # Your input string
        # input_str = '{ "chat-reply": "I\'m sorry to hear that you\'re having a debit card problem. To assist you further, I will need some additional information. Please provide me with your full name, the name of your bank, and any relevant transaction ID or card details related to the issue you\'re facing.", "facts": {}, "STATUS": "More info required" }'
        # input_str = response
        # start_index = input_str.find('"chat-reply":')

        # if start_index != -1:
        #     # Find the position of the following double quote after "chat-reply"
        #     end_index = input_str.find('"', start_index + len('"chat-reply":'))

        #     if end_index != -1:
        #         # Extract the value of "chat-reply"
        #         chat_reply = input_str[start_index + len('"chat-reply":') + 1 : end_index]
        #         print(chat_reply)
        #     else:
        #         print("No closing double quote found for 'chat-reply'.")
        # else:
        #     print("'chat-reply' not found in the string.")

        # print(my_dict)
        # print(my_dict["chat-reply"])
        return JsonResponse({'message': user_query, 'response': response})
    
    

    # elif response == "[IRRELEVANT]":
    #     print("not sure what you are talking about")
    #     status = "new-chat"
    # else:
    #     print("\n oopes trying to get more data")
    #     status = "more-data-req"
    #     # print(response)
    #     dictionary_response = json.loads(response)
    #     user_query = input(dictionary_response["chat-reply"] + "\nuser : ")
    #     print("CHAT SUMmARY : " + chat_summary)
    #     complaint_completion(str(history.messages) + user_query)

    return render(request, 'chatbot.html', context)
