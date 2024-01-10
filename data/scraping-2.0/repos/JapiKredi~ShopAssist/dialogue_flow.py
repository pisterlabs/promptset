from functions import initialize_conversation, initialize_conv_reco, get_chat_model_completions, moderation_check,intent_confirmation_layer,dictionary_present,compare_laptops_with_user,recommendation_validation

import openai
import ast
import re
import pandas as pd
import json

openai.api_key = open("api_key.txt", "r").read().strip()

def dialogue_mgmt_system():
    conversation = initialize_conversation()
    introduction = get_chat_model_completions(conversation)
    print(introduction + '\n')
    top_3_laptops = None
    user_input = ''

    while(user_input != "exit"):
        user_input = input("")

        moderation = moderation_check(user_input)
        if moderation == 'Flagged':
            print("Sorry, this message has been flagged. Please restart your conversation.")
            break

        if top_3_laptops is None:
            conversation.append({"role": "user", "content": user_input})

            response_assistant = get_chat_model_completions(conversation)

            moderation = moderation_check(response_assistant)
            if moderation == 'Flagged':
                print("Sorry, this message has been flagged. Please restart your conversation.")
                break

            confirmation = intent_confirmation_layer(response_assistant)

            moderation = moderation_check(confirmation)
            if moderation == 'Flagged':
                print("Sorry, this message has been flagged. Please restart your conversation.")
                break

            if "No" in confirmation:
                conversation.append({"role": "assistant", "content": response_assistant})
                print("\n" + response_assistant + "\n")
                print('\n' + confirmation + '\n')
            else:
                print("\n" + response_assistant + "\n")
                print('\n' + confirmation + '\n')
                response = dictionary_present(response_assistant)

                moderation = moderation_check(response)
                if moderation == 'Flagged':
                    print("Sorry, this message has been flagged. Please restart your conversation.")
                    break


                print('\n' + response + '\n')
                print("Thank you for providing all the information. Kindly wait, while I fetch the products: \n")
                top_3_laptops = compare_laptops_with_user(response)

                validated_reco = recommendation_validation(top_3_laptops)

                if len(validated_reco) == 0:
                    print("Sorry, we do not have laptops that match your requirements. Connecting you to a human expert.")
                    break

                conversation_reco = initialize_conv_reco(validated_reco)
                recommendation = get_chat_model_completions(conversation_reco)

                moderation = moderation_check(recommendation)
                if moderation == 'Flagged':
                    print("Sorry, this message has been flagged. Please restart your conversation.")
                    break

                conversation_reco.append({"role": "user", "content": "This is my user profile" + response})

                conversation_reco.append({"role": "assistant", "content": recommendation})

                print(recommendation + '\n')

        else:
            conversation_reco.append({"role": "user", "content": user_input})

            response_asst_reco = get_chat_model_completions(conversation_reco)

            moderation = moderation_check(response_asst_reco)
            if moderation == 'Flagged':
                print("Sorry, this message has been flagged. Please restart your conversation.")
                break

            print('\n' + response_asst_reco + '\n')
            conversation.append({"role": "assistant", "content": response_asst_reco})




dialogue_mgmt_system()