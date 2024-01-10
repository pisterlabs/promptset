import os
import openai
import config
import textwrap as tw
import re
from pprint import pprint
import key

openai.api_key = key.secret

def product_observation(prompt_product_desc):
    print("Running product observation")
    response = openai.Completion.create(
        model="text-davinci-002",
        # trained responses
        prompt="The following is a conversation with an AI Customer Segment Recommender. \
      The AI is insightful, verbose, and wise, and cares a lot about finding the product market fit.  \
      AI, please state a insightful observation about " + prompt_product_desc + ".",
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6
        #stop=[" Human:", " AI:"]
    )

    pprint(re.split('\n', response.choices[0].text.strip()))

    return response['choices'][0]['text']

def segment_generator(prompt_product_desc, prompt_seller_persona):
    print("Running Segment Generator")
    response = openai.Completion.create(
        model="text-davinci-002",
        # trained responses
        prompt="The following is a conversation with an AI Customer Segment Recommender. \
        The AI is insightful, verbose, and wise, and cares a lot about finding the product market fit.  \
        What are the top 5 types of customer should a seller who is " + prompt_seller_persona + \
        " sell " + prompt_product_desc + " to?",
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6
    )

    pprint(re.split('\n', response.choices[0].text.strip()))

    return response['choices'][0]['text']

def segment_selector(prompt_product_desc, prompt_seller_persona, prompt_focus_segment):
    print("Running Segment Selector")
    response = openai.Completion.create(
        model="text-davinci-002",
        # trained responses
        prompt="The following is a conversation with an AI Customer Segment Recommender. \
        The AI is insightful, verbose, and wise, and cares a lot about finding the product market fit.  \
        What do " + prompt_focus_segment + \
        " look for when buying " + prompt_product_desc + " from "+ prompt_seller_persona + "and where do I typically \
        find " + prompt_focus_segment + "?",
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6
    )

    pprint(re.split('\n', response.choices[0].text.strip()))

    return response['choices'][0]['text']

def differentiator(prompt_product_desc, prompt_seller_persona, prompt_focus_segment, prompt_differentiator):
    print("Running Segment Selector")
    response = openai.Completion.create(
        model="text-davinci-002",
        # trained responses
        prompt="The following is a conversation with an AI Customer Segment Recommender. \
        The AI is insightful, verbose, and wise, and cares a lot about finding the product market fit.  \
        Suggest a minimum and maximum range of annual spend for a " + prompt_product_desc + "that offers " + prompt_differentiator + \
        "targeting " + prompt_focus_segment +".",
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6
    )

    pprint(re.split('\n', response.choices[0].text.strip()))

    return response['choices'][0]['text']

def ads1(prompt_product_desc, prompt_seller_persona, prompt_focus_segment, prompt_differentiator, prompt_sale_price):
    print("Running Segment Selector")
    response = openai.Completion.create(
        model="text-davinci-002",
        # trained responses
        prompt="The following is a conversation with an AI Customer Segment Recommender. \
        The AI is playful with words, insightful, witty, clever, has great emphathy, and believes that " + prompt_focus_segment +\
          "would be highly satisfied when they buy" + prompt_product_desc + "from " + prompt_seller_persona + \
          "which is known for " \
          + prompt_differentiator + \
          "at the price of " \
          + prompt_sale_price + ". "\
          + "Write a compelling advertisement for this product. ",
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6
    )

    pprint(re.split('\n', response.choices[0].text.strip()))

    global ads1_copy

    ads1_copy = response['choices'][0]['text']
    return ads1_copy

def ads2(prompt_product_desc, prompt_seller_persona, prompt_focus_segment, prompt_differentiator, prompt_sale_price, prompt_second_ad):
    print("Running Second Recommendation")
    response = openai.Completion.create(
        model="text-davinci-002",
        # trained responses
        prompt="The following is a conversation with an AI Customer Segment Recommender. \
        The AI is playful with words, insightful, witty, clever, has great emphathy, and believes that " + prompt_focus_segment +\
          "would be highly satisfied when they buy" + prompt_product_desc + "from " + prompt_seller_persona + \
          "which is known for " \
          + prompt_differentiator + \
          "at the price of " \
          + prompt_sale_price + ". " \
            "It wrote this copy. : " + ads1_copy + ", which is good, but it could be improved by " + prompt_second_ad + \
            "\n. Write an improved compelling advertisement for this product. ",
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6
    )

    pprint(re.split('\n', response.choices[0].text.strip()))

    return response['choices'][0]['text']