#%%
import os
import numpy as np
import openai
import pandas as pd
import re
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential
)
openai.api_key_path=".openai-key"

@retry(wait=wait_random_exponential(min = 1, max = 60), stop = stop_after_attempt(5))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def classify_reason_and_feedback_with_gpt(reason, write_csv = False, example_df = None):
    try :
        completion = completion_with_backoff(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user", 
                    "content": """
                    You will be passed a string explaining why someone may or may not know I was dewormed.
                    Classify the reason into the following categories, examples are provided after each category. Only 
                    return the category name, not the example or any explanation. If you're unsure which category to use, reply 'unsure: $category' to 
                    request additional examples for a specific category, or 'all categories' to request examples for all categories. 

                    'campaign' - There was a large deworming campaign in the area, which meant many people went at the same time
                        e.g.: 'didnt see me there', 'it was announced', 'found me at the treatment point', 'because treatment was for free', 'almost everyone come'.
                    'communication' - I informed the other person. 
                        e.g.: 'I told them I was dewormed', 'he informed him'.
                    'relationship' - I do/don't have a relationship with the person
                        e.g.: 'because of ignorance', 'we are not so close', 'family member', 'always interact', 'are relatives'.
                    'signal' - Observing (or not) a bracelet, ink on my thumb, or a calendar
                        e.g.: 'he saw my ink', 'he didn't see my bracelet'.
                    'type' - They know I'm a good person that cares about my health and community (or a bad person that doesn't)
                        e.g.: 'I always attend such activities', 'knows i cannot miss', 'because she knows i love such things', 'according to how she understands me when it comes to such things'.
                    'circumstances' - There were circumstances that prevented me from getting dewormed 
                        e.g.: 'he knows i didnt get information', 'they know am blind', 'am very old', 'am too old to walk', 'because he knows am a student'.
                    """ 
                    }, 
                {"role": "user", "content": "The string to classify is: " + reason}, 
            ]
        )
        i = 1
        category_examples = ""
        # if detect "unsure: $category" in response, then ask for more examples
        while "unsure" in completion.choices[0].message.content:
            i += 1
            category = completion.choices[0].message.content.split(":")[1].strip()
            category = re.search(r":\s*(\w+)", completion.choices[0].message.content).group(1).lower()

            print("unsure about category: " + category + " for reason " + reason + ", example: ", i) 
            print(category)
            # if all returned, then sample one from each category
            np.random.seed(i)
            if "all" in completion.choices[0].message.content:
                subset_df = example_df.groupby("category_sob_reason_short").sample(n = 1, random_state = i)[['category_sob_reason_short', 'second.order.reason']]
                new_category_examples = str(dict(zip(subset_df["category_sob_reason_short"], subset_df['second.order.reason'])))
            else:
                # sample 5 random examples from the category 
                subset_df = example_df[example_df["category_sob_reason_short"] == category].sample(5, random_state = i)
                new_category_examples = str(dict(zip(subset_df["category_sob_reason_short"], subset_df['second.order.reason'])))


            category_examples = new_category_examples + category_examples
            print("Current examples: " + category_examples)

            completion = completion_with_backoff(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user", 
                        "content": """
                        You will be passed a string explaining why someone may or may not know I was dewormed.
                        Classify the reason into the following categories, examples are provided after each category. Only 
                        return the category name, not the example or any explanation. If you're unsure, reply 'unsure: $category' to 
                        request additional examples for a specific category, or 'all categories' to request examples for all categories. 

                        'campaign' - There was a large deworming campaign in the area, which meant many people went at the same time
                            e.g.: 'didnt see me there', 'it was announced', 'found me at the treatment point', 'because treatment was for free'.
                        'communication' - I informed the other person. 
                            e.g.: 'I told them I was dewormed', 'he informed him'.
                        'relationship' - I do/don't have a relationship with the person
                            e.g.: 'because of ignorance', 'we are not so close', 'family member', 'always interact', 'are relatives'.
                        'signal' - Observing (or not) a bracelet, ink on my thumb, or a calendar
                            e.g.: 'he saw my ink', 'he didn't see my bracelet'.
                        'type' - They know I'm a good person that cares about my health and community (or a bad person that doesn't)
                            e.g.: 'I always attend such activities', 'knows i cannot miss', 'because she knows i love such things', 'according to how she understands me when it comes to such things'.
                        'circumstances' - There were circumstances that prevented me from getting dewormed 
                            e.g.: 'he knows i didnt get information', 'they know am blind', 'am very old', 'am too old to walk', 'because he knows am a student'.

                        Additional examples for category: """ + category_examples
                        }, 
                    {"role": "user", "content": "The string to classify is: " + reason}, 
                ]
        )
            # break if asked for more than 3 examples
            if i > 5:
                completion = completion_with_backoff(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "user", 
                            "content": """
                            You will be passed a string explaining why someone may or may not know I was dewormed.
                            Classify the reason into the following categories, examples are provided after each category. Only 
                            return the category name, not the example or any explanation. You must make a final decision now or choose "other". 

                            'campaign' - There was a large deworming campaign in the area, which meant many people went at the same time
                                e.g.: 'didnt see me there', 'it was announced', 'found me at the treatment point', 'because treatment was for free'.
                            'communication' - I informed the other person. 
                                e.g.: 'I told them I was dewormed', 'he informed him'.
                            'relationship' - I do/don't have a relationship with the person
                                e.g.: 'because of ignorance', 'we are not so close', 'family member', 'always interact', 'are relatives'.
                            'signal' - Observing (or not) a bracelet, ink on my thumb, or a calendar
                                e.g.: 'he saw my ink', 'he didn't see my bracelet'.
                            'type' - They know I'm a good person that cares about my health and community (or a bad person that doesn't)
                                e.g.: 'I always attend such activities', 'knows i cannot miss', 'because she knows i love such things', 'according to how she understands me when it comes to such things'.
                            'circumstances' - There were circumstances that prevented me from getting dewormed 
                                e.g.: 'he knows i didnt get information', 'they know am blind', 'am very old', 'am too old to walk', 'because he knows am a student'.
                            'other' - The reason is not in the above categories

                            Additional examples for category: """ + category_examples
                            }, 
                        {"role": "user", "content": "The string to classify is: " + reason}, 
                    ]
                )
                break
        gpt_output = completion.choices[0].message.content
    except:
        gpt_output = "NA"
        
    print([reason, gpt_output])

    # write to csv and append:
    if write_csv:
        pd.DataFrame([[reason, gpt_output]]).to_csv("temp-data/sob-other-reasons-feedback-gpt-3.5-turbo.csv", index=False, header=False, mode="a")
    return gpt_output 

#%%
def classify_reason_with_gpt(reason, write_csv = False):
    completion = completion_with_backoff(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user", 
                "content": """
                You will be passed a string explaining why someone may or may not know I was dewormed.
                Classify the reason into the following categories, examples are provided after each category. Only 
                return the category name, not the example. If none of the categories fit well, return 'other'.:

                'campaign' - There was a large deworming campaign in the area, which meant many people went at the same time
                    e.g.: 'didnt see me there', 'it was announced', 'found me at the treatment point', 'because treatment was for free'.
                'communication' - I informed the other person. 
                    e.g.: 'I told them I was dewormed', 'he informed him'.
                'relationship' - I do/don't have a relationship with the person
                    e.g.: 'because of ignorance', 'we are not so close', 'family member', 'always interact', 'are relatives'.
                'signal' - Observing (or not) a bracelet, ink on my thumb, or a calendar
                    e.g.: 'he saw my ink', 'he didn't see my bracelet'.
                'type' - They know I'm a good person that cares about my health and community (or a bad person that doesn't)
                     e.g.: 'I always attend such activities', 'knows i cannot miss', 'because she knows i love such things', 'according to how she understands me when it comes to such things'.
                'circumstances' - There were circumstances that prevented me from getting dewormed 
                    e.g.: 'he knows i didnt get information', 'they know am blind', 'am very old', 'am too old to walk', 'because he knows am a student'.
                'other' - None of the above fit well.
                """ 
                }, 
            {"role": "user", "content": "The string to classify is: " + reason}, 

        ]
    )
    print([reason, completion.choices[0].message.content])

    # write to csv and append:
    if write_csv:
        pd.DataFrame([[reason, completion.choices[0].message.content]]).to_csv("temp-data/sob-other-reasons-gpt-3.5-turbo.csv", index=False, header=False, mode="a")
    return completion.choices[0].message.content

# classify_reason_with_gpt("I told them I was dewormed")
# classify_reason_with_gpt("a family member")
# classify_reason_with_gpt("according to how she knows me")
# classify_reason_with_gpt("all went")
# classify_reason_with_gpt("almost eveyone comes")

#%%
sob_reason_csv = pd.read_csv("temp-data/sob-other-reasons.csv")
sob_example_df = pd.read_csv(
        "temp-data/second-order-reason-raw-data.csv"
)


# filter out any other categories
sob_example_df = sob_example_df[sob_example_df["category_sob_reason_short"].isin(["campaign", "communication", "relationship", "signal", "type", "circumstances"])]
#%%


classify_reason_and_feedback_with_gpt(
    "almost everyone come", 
    example_df = sob_example_df
)

#%%

classify_reason_and_feedback_with_gpt(
    "I told them I was dewormed", 
    example_df = sob_example_df
)

classify_reason_and_feedback_with_gpt(
    "because everyone was to take deworming medicine",
    example_df = sob_example_df
)

#%%

sob_reason_csv["gpt-3.5-turbo"] = sob_reason_csv["second.order.reason"].apply(classify_reason_and_feedback_with_gpt, write_csv = True, example_df = sob_example_df)

sob_reason_csv.to_csv("temp-data/sob-other-reasons-feedback-gpt-3.5-turbo-single-pass.csv", index=False)


