# https://medium.com/@neonforge/chatgpt-api-how-to-build-ai-applications-with-python-step-by-step-guide-with-examples-b7a2cd425651


import openai

openai.api_key = "YOUR_OPENAI_API_KEY"

topics = ['Frozen movie', 'Avatar movie']

impersonated_role = "You are acting as Joe Rogan the uncensored comedian."

previous_output = ''
for i in range(len(topics)):

    if i == 0:
        output = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": impersonated_role},
                {"role": "user", "content": f"Give a long ranting stand up about {topics[i]} "
                                            f"don't hold back your uncensored opinions. "
                                            f"Delve deeply in each key point and build a cohesive response. "
                                            f"Feel free to swear and curse at any time."},
            ]
        )
    else:
        output = openai.ChatCompletion.create(
                          model="gpt-3.5-turbo",
                          messages=[
                                {"role": "system", "content": impersonated_role},
                                {"role": "user", "content": f"Give a long ranting stand up about {topics[i]} "
                                                            f"don't hold back your uncensored opinions. "
                                                            f"Delve deeply in each key point and build a cohesive response. "
                                                            f"Feel free to swear and curse at any time. "
                                                            f"Build your response as continuation of this text: {previous_output}"},
                            ]
                        )

    for item in output['choices']:
        print(f"Let's talk about {topics[i]}!" )
        print(item['message']['content'])
        previous_output = item['message']['content']
        print('\n\n')