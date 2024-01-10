import pandas as pd
import openai
import time

def gptclassifier(df,system_message, template, examples, completions, timer_frequency):

    i=0    
    for txt in df.loc[:,["caption","username"]].iterrows():
        
        # timer
        i+=1
        if i%timer_frequency==5:
            print(f"Counter at {i}")

        messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": template},
                examples[0], examples[1], examples[2], examples[3],
                {"role": "user", "content": f"Post: '{txt[1]['caption']}'. User: @{txt[1]['username']}"}]
        
        # try except to prevent openAIs limits
        try:
            response = openai.ChatCompletion.create(model="gpt-4",
                                                messages=messages)
            completions.append(response["choices"][0]["message"]["content"])
        except Exception as err:
            print("Waiting for 65s", err.__class__.__name__)
            time.sleep(65)
            try:
                response = openai.ChatCompletion.create(model="gpt-4",
                                                    messages=messages)
                completions.append(response["choices"][0]["message"]["content"])
            except Exception as err:
                print("Waiting for 65s again", err.__class__.__name__)
                time.sleep(65)
                response = openai.ChatCompletion.create(model="gpt-4",
                                                    messages=messages)
                completions.append(response["choices"][0]["message"]["content"])
    completions_as_boolean = [True if ((response.endswith("rue"))
                                   or (response.endswith("rue."))
                                   or (response.endswith("True/Uncertain.")))
                          else False if ((response.endswith("alse"))
                                         or (response.endswith("alse."))
                                         or (response.endswith("False/Uncertain.")))
                          else response for response in completions]
    four_labels = ["True." if ((response.endswith("rue"))
                                   or (response.endswith("rue.")))
                          else "True/Uncertain." if ((response.endswith("True/Uncertain.")) or (response.endswith("True/Uncertain.")))
                          else "False." if ((response.endswith("alse"))
                                         or (response.endswith("alse.")))
                          else "False/Uncertain." if ((response.endswith("False/Uncertain.")) or (response.endswith("False/Uncertain.")))
                          else response for response in completions]
    return completions, completions_as_boolean, four_labels