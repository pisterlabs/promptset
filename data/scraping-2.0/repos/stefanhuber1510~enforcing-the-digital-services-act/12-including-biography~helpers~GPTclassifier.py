import pandas as pd
import openai
import time
def find_closest_posts(df, shortcode):
    # Get the original post's timestamp and author
    original_post = df[df['shortcode'] == shortcode]
    original_timestamp = original_post['dt_year_mon'].values[0]
    author = original_post['username'].values[0]

    # Filter out posts made by the same author
    same_author_posts = df[df['username'] == author].copy()

    # Calculate the absolute time difference
    same_author_posts['time_difference'] = (same_author_posts['dt_year_mon'] - original_timestamp).abs()

    # Exclude the original post and find the two posts with smallest time difference
    closest_posts = same_author_posts[same_author_posts['shortcode'] != shortcode].nsmallest(2, 'time_difference')

    # Return the shortcodes of the closest posts
    return closest_posts['caption'].values

def gptclassifier(df,base_messages,completions,model="gpt-3.5-turbo",temperature=1,CoT=False,biography=False,context_posts=False,timer_frequency=5):

    i=0    
    for txt in df.loc[:,["caption","username",'biography',"shortcode"]].iterrows():
        
        # timer
        i+=1
        if i%timer_frequency==2:
            print(f"Counter at {i}")

        messages = base_messages.copy()
        if context_posts:
            next_posts = find_closest_posts(df, txt[1]['shortcode'])
            if CoT:
                messages.append({"role": "user", "content": f"Post: '{txt[1]['caption']}'. \nAuthor: @{txt[1]['username']}. \nAuthor Bio: '{txt[1]['biography']}' \nTemporally closest other posts from same user (this is potentially helpful context for you, only classify the main post!): \nPost 1:'{next_posts[0]}' \nPost 2:'{next_posts[1]}'"})
            else: # not(CoT)
                messages.append({"role": "user", "content": f"Post: '{txt[1]['caption']}'.\nAuthor: @{txt[1]['username']}. \nAuthor Bio: '{txt[1]['biography']}'. \nTemporally closest other posts from same user (this is potentially helpful context for you, only classify the main post!): \nPost 1:'{next_posts[0]}' \nPost 2:'{next_posts[1]}'. Again, return only the label and a dot after. No reasoning:"})
        else: # not(context_posts)
            if CoT and biography:
                messages.append({"role": "user", "content": f"Post: '{txt[1]['caption']}'. \nAuthor: @{txt[1]['username']}. \nAuthor Bio: '{txt[1]['biography']}'"})
            elif not(CoT) and biography:
                messages.append({"role": "user", "content": f"Post: '{txt[1]['caption']}'.\nAuthor: @{txt[1]['username']}. \nAuthor Bio: '{txt[1]['biography']}'. Again, return only the label and a dot after. No reasoning:"})
            elif (not(CoT)) and (not(biography)):
                messages.append({"role": "user", "content": f"Post: '{txt[1]['caption']}'. User: @{txt[1]['username']}. Again, return only the label and a dot after. No reasoning:"})
            else: # CoT and not(biography)
                messages.append({"role": "user", "content": f"Post: '{txt[1]['caption']}'. User: @{txt[1]['username']}."})

        # try except to prevent openAIs limits
        try:
            response = openai.ChatCompletion.create(model=model,
                                                messages=messages,
                                                temperature=temperature)
            completions.append(response["choices"][0]["message"]["content"])
        except Exception as err:
            print("Waiting for 65s", err.__class__.__name__)
            print("-----------------")
            print(err)
            time.sleep(65)
            try:
                response = openai.ChatCompletion.create(model=model,
                                                    messages=messages,
                                                temperature=temperature)
                completions.append(response["choices"][0]["message"]["content"])
            except Exception as err:
                print("Waiting for 65s again", err.__class__.__name__)
                time.sleep(65)
                response = openai.ChatCompletion.create(model=model,
                                                    messages=messages,
                                                    temperature=temperature)
                completions.append(response["choices"][0]["message"]["content"])

    four_labels = generate_labels(completions)
    
    return completions, four_labels

def generate_labels(completions):
    # might make potential endings as dictionary variable, later
    return [True if ((response.endswith("ly sponsored"))
                                   or (response.endswith("ly sponsored."))
                                   or (response.endswith("ly Sponsored."))
                                   or (response.endswith("ly Sponsored")))
                          else False if ((response.endswith("ly not sponsored"))
                                         or (response.endswith("ly not sponsored."))
                                         or (response.endswith("ly not Sponsored"))
                                         or (response.endswith("ly not Sponsored.")))
                          else response for response in completions]