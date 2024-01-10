import pandas as pd
import openai
import time

def gptclassifier(df, messages, completions, timer_frequency, model="gpt-3.5-turbo"):

    i=0    
    for txt in df.loc[:,["caption","username"]].iterrows():
        
        # timer
        i+=1
        if i%timer_frequency==1:
            print(f"Counter at {i}")

        messages.append(
                {"role": "user", "content": f"Post: '{txt[1]['caption']}'. User: @{txt[1]['username']} \nkeep the reasoning very concise:"})
        
        # try except to prevent openAIs limits
        try:
            response = openai.ChatCompletion.create(model=model,
                                                messages=messages)
            completions.append(response["choices"][0]["message"]["content"])
        except Exception as err:
            print("Waiting for 65s", err.__class__.__name__)
            time.sleep(65)
            try:
                response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                    messages=messages)
                completions.append(response["choices"][0]["message"]["content"])
            except Exception as err:
                print("Waiting for 65s again", err.__class__.__name__)
                time.sleep(65)
                response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                    messages=messages)
                completions.append(response["choices"][0]["message"]["content"])
        
        messages.pop()
                

    four_labels = ["Likely sponsored." if ((response.endswith("ly sponsored."))
                                   or (response.endswith("ly sponsored")))
                          else "Self advertisement." if ((response.endswith("lf advertisement.")) or (response.endswith("lf advertisement")))
                          else "Ambiguous." if ((response.endswith("Ambiguous."))
                                         or (response.endswith("Ambiguous")))
                          else "Likely not sponsored." if ((response.endswith("not sponsored.")) or (response.endswith("not sponsored")))
                          else response for response in completions]
    return four_labels, completions


# Note: gpt-3.5 by default tends to avoid the label 'sponsored'. That's why the prompt emphasises a strong emphasis in favor of it. I suspect this is due to this truthfulness fine-tuning, making it afraid to unjustly judge someone's post

standard_message = [{"role": "system", "content": "You are an assistant helping an academic to reason about whether a post contains (potentially non-commerical) promotional activity or even is potentially sponsored. I will provide you with the caption of an instagram post. You give me a short and concise reasoning why or why not the post might be an ad, i.e. the result of a financial contract. For later classification there are four labels available, 'Potentially sponsored', 'Self advertisement', 'Ambiguous' and 'Likely not sponsored'. Be concise in your reasoning and always strictly adhere to the pattern from the examples, i.e. always decide for one and only one label and finish your response with it. Err strongly towards 'Potentially sponsored', the slightest indication of potential sponsoring is sufficient to return 'Potentially sponsored'. Also strongly prefer 'Self advertisement' over 'Ambigous'. Always keep responses short and concise."},
{"role": "user", "content": "Post: ''I DO NOT OWN THE RIGHTS TO THIS SONG. Upload attempt #2.... I COULD NOT STOP playing this song over Christmas break for some odd reason. It’s my favorite joint off of @badgalriri ‘s #anti album. Listening repeatedly made wonder what it would sound like with drums... 🤔😏 #thepocketqueen 👸🏾♥️🤦🏾\u200d♀️ #practicemakespocket #jamesjoint #groovewithme #drummerbae\n\nHair: @hairbylucylomuro_ \nThreads: @truequeenbrand'. Author: @thepocketqueen"},
{"role": "assistant", "content": "Key indicators: 'of @badgalriri ‘s #anti album', 'Threads: @truequeenbrand', 'Hair: @hairbylucylomuro'.\nReasoning: The post clearly promotes a song, another artist @badgalriri. Additionally there are several businesses featured in the. Each of those four aspects by itself is some indication of sponsoring, so all together clearly potentially sponsored. Label: Potentially sponsored."},
{"role": "user", "content": "Post: 'I love cheeseburgers so much!😱 @barneysburgershop'. Author: @stevietheking"},
{"role": "assistant", "content": "Key indicators: '@barneysburgershop'.\nReasoning: The post clearly promotes a restaurant called barneysburgershop. However it is also extremely common that people feature restaurants because they genuinely enjoy their food or want to show off with it. Lacking further evidence, it rather Ambigous than a paid partnership. Label: Ambiguous."},
{"role": "user", "content": "Post: 'She drives me INSANE every other hour, but i don’t know what i would do without her crazy ass! #sisters'. User: @thestilettomeup"},
{"role": "assistant", "content": "Key indicators: '#sisters'.\nReasoning: Clearly a personal post about the author's sister. Label: Likely not sponsored."},
{"role": "user", "content": "Post: 'weworewhat swim spring collection home designs and prints inspired by elements from my home and favorite interior design what you can expect this silhouette print inspired by the many female body sculptures that can be found in my apartment marbles cowhide florals and more @shopweworewhat'. Author: weworewhat"},
{"role": "assistant", "content": "Key indicators: 'weworewhat', '@shopweworewhat'.\nReasoning: This post clearly advertises swim suits. However the shop @shopweworewhat is clearly a shop of the author weworewhat herself, so its very unlikely a paid partnership but clearly self advertisement. Label: Self advertisement."},
{"role": "user", "content": "Post: 'A night in San Francisco 💋 I am so excited to meet all of the @createcultivate attendees and share more of my story... #moreofwhatmatters \nTop: @storets \nPants : @jacquemus \nShoes: @gianvitorossi \nStyled By: @monicarosestyle'. Author: iamcattsadler"},
{"role": "assistant", "content": "Key indicators: 'Top: @storets', 'Pants : @jacquemus', 'Shoes: @gianvitorossi', 'Styled By: @monicarosestyle'.\nReasoning: This post promotes various fashion brands and stylers. There is no evidence suggesting it is not paid. Label: Potentially sponsored."}]