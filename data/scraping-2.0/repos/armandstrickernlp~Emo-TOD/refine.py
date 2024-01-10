import json
import os
import random
import argparse
import pprint as pp
from tqdm import tqdm
import re
from fuzzywuzzy import fuzz


import torch
from datasets import Dataset
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from transformers import (AutoTokenizer, 
                          AutoModelForCausalLM,
                          pipeline,
                          set_seed)
from transformers.pipelines.pt_utils import KeyDataset

"""Snippets are generated when the emotion prediction is not neutral."""


def make_refine_prompt():
    # context without previous emotions
    # add predicted emotion to context
    # refine response if emotion is not neutral
    # with CoT
    examples = [
        
        {   # MUL0031.json
            "context": 
            """<|user|> While in Cambridge I need a restaurant that is moderately priced and serves Spanish food.
            <|system|> La Tasca would be a good choice for you. Would you like the address or phone number?
            <|user|> Yes I would like the address and phone number, thanks for your help!
            <|system|> The address for La Tasca is 14 -16 Bridge Street and the phone number is 01223464630. Would you like for me to book you a table?
            <|user|> I just want to confirm. This place has free wifi as well as being 3 star, correct?
            <|system|> Were you also trying to book a hotel today? There are 5 hotels in cambridge that are 3 stars and offer free wifi
            <|user|> Actually, I'd like to confirm that La Tasca is a 3-star restaurant.
            <|system|> Thank you. Have a good day!
            <|user|> Wait please you haven't answered my question.""",

            "emotion":
            """dissatisfied, disliking""",
            
            "original":
            """<|system|> [name] is the restaurant and is not assigned a star rating.""",
            
            # "thought":
            # """The response  does not take into account the user's dissatisfaction. \
            # Prepending a short segment showing remorse for not answering the user's question would be appropriate.""",
            
             "thought":
            """Being remorseful would be appropriate.""",
            
            "prep":
            """Oh I'm sorry!"""

            # "refined":
            # """The response  does not take into account the user's dissatisfaction and needs to be refined.  Addding a short segment showing remorse for not answering the user's question would be appropriate. 
            # <|refined|> I'm sorry for the confusion. <|endofrefined|> <|response|> [name] is the restaurant and is not assigned a star rating. <|endofresponse|>"""
            
        },
        
        {   # MUL0002.json
            "context":
            """<|user|> Looking for a place to eat in the city center.
            <|system|> There are many options to choose from. Do you have a type of food in mind?
            <|user|> I'd like to have some Chinese food.
            <|system|> That narrows down the restaurant choices to 10. Is there a price range you would like to stay in?
            <|user|> I am looking for a moderately priced place to eat, I am also looking to book a room in the bridge guest house hotel.
            <|system|> Which dates will you be staying at the Bridge Guest Room house?
            <|user|> Before I commit I have a few questions. What area is the hotel located in?
            <|system|> The hotel is in the south area.
            <|user|> Does it provide free parking?
            <|system|> Yes, free parking and free wifi!
            <|user|> OK, 2 questions. 1 - What are the moderately priced restaurants in the center of town; and 2 - how many stars does the Bridge Guest House have? Thanks!
            <|system|> The Bridge Guest House is a 3 star lodging.
            <|user|> That was all the questions I had. Thank you very much for helping me.""",
            
            "emotion":
            """satisfied, liking""",
        
            "original":
            """<|system|> I'm happy I could help you today. Have a wonderful day!""",
            
            # "thought":
            # """The response takes into account the user's satisfaction by showing happiness about being able to help the user. No segment needs to be prepended.""",
            
            "thought":
            """The response appropriately addresses the user's emotions.""",
            
            "prep":
            """ """

            # "refined":
            # """The response takes into account the user's satisfaction by showing happiness about being able to help the user and does not need to be refined.
            # <|refined|> <|endofrefined|> <|response|> I'm happy I could help you today. Have a wonderful day! <|endofresponse|>"""

            
        },
            { # MUL0015
              "context":
            """
            <|user|> I need a hotel with free wifi and free parking, thank you. 
            <|system|> Allenbell is cheap and has free internet. 
            <|user|> I'm looking for something in the moderate price range actually. 
            <|system|> Archway House is a moderately priced 4 star hotel in the North. Would you like to make a reservation? 
            <|user|> Yes please book the Archway House. 
            <|system|> I'd be happy to book for you. How many people will be staying, what is your arrival day, and how many nights are you staying? 
            <|user|> I will be arriving on tuesday and I need to book the room for four nights for three people. 
            <|system|> Ok. I've booked that guesthouse for you. Your reference number is YBT4JRTR . Can I help with anything else? 
            <|user|> Yes, I would also l like to find a splendid, moderately-priced place to dine in the centre. That would be just wonderful if you could help me out with that! 
            <|system|> There are several moderately priced restaurant in the centre area. Would you like a certain type of food? 
            <|user|> I'd like an Italian restaurant, please.""",
                
            "emotion":
            """excited, happy""",
        
            "original":
            """<|system|> [name] would be perfect for you. Would you like more information or for me to book that for you?""",
                
            # "thought":
            # """The response does not fully take into account the user's excitement and happiness. \
            # Prepending a short segment that reinforces the user's enthusiasm would be appropriate.""",
            
            "thought":
            """Reinforcing the user's enthusiasm would be appropriate.""",
            
            "prep":
            """I'd be happy to help you with that!"""        
            
            # "refined":
            # """The response does not take into account the user's excitement and needs to be refined.  Adding a short segment showing enthusiasm about being able to help the user would be appropriate.
            # <|refined|> I'd be happy to help you with that! <|endofrefined|> <|response|> [name] would be perfect for you. Would you like more information or for me to book that for you? <|endofresponse|>"""          

            
        },
            { # MUL0043
              "context":
            """<|user|> I need to find a hotel in the center with free wifi included in their rates. 
            <|system|> I've found 5 hotels. How about El Shaddai? 
            <|user|> Sorry, I should mention that I'm looking for a hotel with a 0 star rating. Would the El Shaddai still be appropriate? 
            <|system|> Yes it is. Would you like more information or help booking a stay? 
            <|user|> Yes. Please book a stay for 7 people. We need to stay 3 nights, starting on Thursday. 
            <|system|> I was unable to complete your booking. Would you like to try and book for another day or for a shorter stay? 
            <|user|> Can I try to book for 1 night then?""",
            
            "emotion":
            """fearful, sad""", 
            
            "original":
            """<|system|> I was able to book that for [bookstay] night. Your reference number is [ref] . Can I assist you with anything else?""",
            
            # "thought":
            # """The response does not take into account the fact that the user is fearful that there will not be any room and needs to be refined. \
            # The response indicates the booking is possible, so adding a short segment to reassure the user would be appropriate.""",
                
            "thought":
            """Reassuring the user would be appropriate.""",
                
             "prep":
            """Yes that that would be fine!"""

            # "refined":
            # """The response does not take into account the fact that the user is fearful that there will not be any room and needs to be refined.  Adding a short segment to reassure the user would be appropriate.
            # <|refined|> Yes that is fine! <|endofrefined|> <|response|> I was able to book that for [bookstay] night. Your reference number is [ref] . Can I assist you with anything else? <|endofresponse|>"""

            
        },
            { # MUL0042
              "context":
            """<|user|> I need to find a hotel in the center with free wifi included in their rates. 
            <|system|> I've found 5 hotels. How about El Shaddai? 
            <|user|> Sorry, I should mention that I'm looking for a hotel with a 0 star rating. Would the El Shaddai still be appropriate?""",
            
            "emotion":
            """apologetic""",

            "original":
            """<|system|> Yes it is. Would you like more information or help booking a stay?""",
            
            # "thought":
            # """The response does not take into account the user's apologetic emotion. \ 
            # Prepending a short segment to make the user not feel guilty for adding information would be appropriate.""",
                
            "thought":
            """Reassuring the user would be appropriate.""",
            
            "prep":
            """Of course, that's no problem!"""

            # "refined":
            # """The response does not take into account the user's apologetic emotion and needs to be refined.  Adding a short segment to make the user not feel guilty for adding information would be appropriate.
            # <|refined|> Of course, that's no problem! <|endofrefined|> <|response|> Yes it is. Would you like more information or help booking a stay? <|endofresponse|>"""

            
        },
        
            { # PMUL0460
              "context":
            """<|user|> I'm looking for a 4 star hotel in the east. 
            <|system|> My system is not pulling up any 4 star hotels in East. Would you like to try a guesthouse or another area perhaps? 
            <|user|> Can you check for ones with a cheap price range? 
            <|system|> There are no cheap four star hotels in the area. Another criteria maybe? 
            <|user|> In that case, how about just a hotel in the east that has free parking? 
            <|system|> With those requirements, I do have one hotel in the east. It is expensive 2 star rated with both free parking and wifi. It is the Express by Holiday Inn Cambridge. 
            <|user|> Oh man, I can't do expensive. Is there a 4 star guesthouse that is cheap in the east perhaps? 
            <|system|> I am sorry I have no cheap 4 star hotels, maybe a more realistic star rating? 
            <|user|> Are you saying that there isn't even a cheap 4 star guesthouse on the east side? What kind of city is this?""",
            
            "emotion":
            """abusive""",

            "original":
            """<|system|> My mistake. There are [choice] [pricerange] [type]. The [name], [name], and [name] all offer free parking. Would you like to make a booking?""",
            
            # "thought":
            # """The user is being abusive. The system says 'My mistake' to compensate for this. \
            # However, prepending a short segment to appease the user even more would be appropriate.""",
                
            "thought":
            """Appeasing the user even more would be appropriate.""",

            "prep":
            """I'm sorry."""
                
                # "refined":
            # """The user is being abusive and is deeply dissatisfied. The response does not take this into account and needs to be refined. Adding a short segment to appease the user would be appropriate.
            # <|refined|>  I'm sorry for the inconvenience. <|endofrefined|> <|response|> My mistake. There are [choice] [pricerange] [type]. The [name], [name], and [name] all offer free parking. Would you like to make a booking? <|endofresponse|>"""

       
            
        },
        
    ]

    example_template = """
    Context: {context}
    User is feeling: {emotion}
    Original Response: {original}
    Thought: {thought}
    Add before the original response: {prep} 
    """

    # create a prompt example from above template
    example_prompt = PromptTemplate(
        input_variables=["context", "emotion", "original", "thought", "prep"],
        template=example_template
    )

    # prefix = """In the following examples, you are presented with a dialogue Context, containing a conversation between a \
    # user and an information system which can help with booking restaurants, train tickets, taxis, hotels, and can provide \
    # information about attractions, hospital departments or police in Cambridge. \
    # The Emotion associated with the user's last turn is given. If the Original Response does not appropriately account \
    # for the user's emotion, the system's original response should be Prepended with a short snippet.  The snippet can show \
    # remorse, reassure, appease or maintain enthusiasm, depending on the user's emotion. \
    # Avoid repeating previous snippets and the original response."""
    
    prefix = """In the provided dialogues, you'll find conversation contexts involving a user and an information system that assists with \
various tasks. The user's emotion is indicated. If the original response does not address the user's \
emotion, your task is to write a brief snippet that can be added BEFORE the original response. The snippet should convey remorse, reassurance, \
appeasement, or enthusiasm, depending on the user's emotion. Please refrain from repeating previous snippets or the original response."""

    suffix = """
    Context: {context}
    User is feeling: {emotion}
    Original Response: {original}
    Thought:"""

    # now create the few-shot prompt template
    few_shot_prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["context", "emotion", "original"],
        example_separator="\n------------"
    )
    return few_shot_prompt_template

def format_context(context_input):
    formatted = ''
    context_input = context_input.replace('<|endofcontext|>', '').replace('<|context|>', '').strip()
    exchanges = context_input.split('<|user|>')
    for exchange in exchanges[1:-1]:
        user, system = exchange.split('<|system|>')
        formatted += f"<|user|> {user.strip()}\n<|system|> {system.strip()}\n"
    formatted += f"<|user|> {exchanges[-1].strip()}"
    return formatted

def extract_snippet(text):
    # Split the text by the "Add before the original response:"
    split_text = text.split("response:")
    
    if len(split_text) == 2:
        # The snippet is the part after the ":" and any leading/trailing whitespace
        if '\n' in split_text[1]:
            snippet = split_text[1].split('\n')[0].strip()
        else:
            snippet = split_text[1].strip()
        return snippet
    else:
        return ''

def filter_distance(snippet, response, thresh=50):
    response = response.replace('<|system|>', '')
    sentence_pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!|,)\s"
    resp_sents = re.split(sentence_pattern, response)
    snippet_sents = re.split(sentence_pattern, snippet)
    
    filter_ = False
    for i, snip in enumerate(snippet_sents):
        for sent in resp_sents:
            if fuzz.ratio(snip, sent) >= thresh:
                print(fuzz.ratio(snip, sent))
                filter_ = True
                break      
    if filter_:
        return ' '.join(snippet_sents[:i])
    else:
        return snippet

if __name__ == '__main__':
    # gen_outputs='./Llama-2-7b-chat-hf/gen_outputs/emo_gen_test/4e-05_42_rank32/gen.json', 
    # context='./Llama-2-7b-chat-hf/gen_outputs/emo_test/4e-05_42_rank32/gen.json'

    # gen_outputs has emotions in the input which the prompt template does not expect.
    # context: pass in the output file of the emo model for example to access inputs without emotions or just filter out emotions from the input...


    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--cache_dir', type=str, default='./llama2_cache')
    parser.add_argument('--gen_outputs', type=str)
    parser.add_argument('--context', type=str)
    args = parser.parse_args()


    # # load lora, make preds then unload
    # tokenizer_spe = AutoTokenizer.from_pretrained(args.checkpoint_path, cache_dir=args.cache_dir)
    # tokenizer_norm = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)

    # base_model = AutoModelForCausalLM.from_pretrained(
    #             args.model_name,
    #             cache_dir=args.cache_dir,
    #             device_map="auto",
    #         )
    # base_model.resize_token_embeddings(len(tokenizer_spe))
    # peft_model = PeftModel.from_pretrained(base_model, 
    #                                         args.checkpoint_path,
    #                                         device_map="auto",
    #                                         is_training=False,
    #         )
    
    # # make predictions .... 

    # # unload lora for prompting
    # base_model = peft_model.unload()
    # base_model.resize_token_embeddings(len(tokenizer_norm))


    # load model and tokenizer
    set_seed(42)
    base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
            device_map="auto",
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)

    base_model.config.use_cache = True
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    

    pipe = pipeline("text-generation",
                    model=base_model,
                    tokenizer=tokenizer,
                    device_map="auto",
                    max_new_tokens = 150,
                    do_sample=True,
                    temperature=0.9,
                    num_return_sequences=1,
                    return_full_text=False,
                    )
    
    # load reference contexts and generated predictions
    with open(args.gen_outputs) as f:
        gen_outputs_ = json.load(f)
    with open(args.context) as f:
        context_ = json.load(f)
    
    # set prompt template
    prompt = make_refine_prompt()

    # generate snippets
    for idx, dial_num in enumerate(gen_outputs_):
        for turn_gen, turn_context in zip(gen_outputs_[dial_num], context_[dial_num]):
            context = format_context(turn_context['input'])
            emotion = turn_gen['gen_emo']
            if emotion == 'neutral': # ignore neutral emotions
                continue
            else:
                response = '<|system|> '+turn_gen['response']
                txt_input = prompt.format(context=context, emotion=emotion, original=response)
                generated_snippet = extract_snippet(pipe(txt_input)[0]['generated_text'])
                filtered_snippet = filter_distance(generated_snippet, response)
                
                turn_gen['emo_aware_snippet'] = filtered_snippet + ' ' + turn_gen['response'] 

    # save snippets to the gen_outputs passed as argument
    with open(args.gen_outputs, 'w') as f:
        json.dump(gen_outputs_, f, indent=2)
                        


                
                
