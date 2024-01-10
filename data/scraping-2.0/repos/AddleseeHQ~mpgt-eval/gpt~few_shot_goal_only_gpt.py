import openai
import pandas as pd
from time import sleep

openai.organization = "org-blah"
openai.api_key = "sk-blah"

def generate_basic_prompt(src, few):
    prompt = "Each conversation has a window between [start] and [end]. Return this window with the [MASK] tags replaced with the goal annotations. Here are some examples. " + src
    return prompt

def generate_specific_prompt(src, few):
    prompt = "Each of these conversations is between two people and a robot called ARI. There is a section of each conversation between the [start] and [end] tags. I want you to return this section of the conversation, but I want you to replace the [MASK] tags with the user goals. Do not change any of the other words in the section, only replace [MASK]. Every [MASK] should be replaced. Here are some examples. " + src
    return prompt

def generate_annotation_instructions_prompt(src, few):
    prompt = "Each of these conversations is between two people and a robot called ARI. I want you to first extract the text between [start] and [end]. There are [MASK] tags in the extracted text. I want you to replace the [MASK] tags with goal annotations. Do not change any of the other taxt. If the person's goal can be determined by that turn, add an '@' symbol followed by 'G' (G for goal), and then brackets with the speaker ID and what their goal is. If it is a shared goal, you can annotate both speakers with a '+' sign between them. For example, if you think LC and RP share the goal, you can write LC+RP. If you think the goal is being answered, you can do the same but with 'AG' (AG for Answer Goal) instead of 'G'. Finally, if you think the person is closing the goal, you can do the same annotation using 'CG' (CG for Close Goal) instead of 'G' or AG'. Here are some examples. " + src
    return prompt

def generate_story_prompt(src, few):
    prompt = "There once was a conversation between a patient, a companion, and a robot called ARI. One bit of the conversation was confusing. A helpful researcher noted the start with [start], and the end with [end]. The confusing bits are marked with [MASK]. Can you help us figure out the goals that should replace the [MASK] tags? Here are some examples. " + src
    return prompt

def generate_roleplay_prompt(src, few):
    prompt = "You are listening to a conversation between two people and a robot called ARI. You are a helpful assistant that needs to figure out what goals the people have. You need to pay attention to the [MASK] tags between the [start] and [end] tags in the given conversation. Your job is to replace these [MASK] tags with the correct goal annotations. Here are some examples. " + src
    return prompt

def generate_reasoning_prompt(src, few):
    prompt = "I will give you a conversation between two people and a robot called ARI. You need to return the text between [start] and [end] with the [MASK] tags replaced by user goals. Let's step through how to figure out the correct annotation. If the conversation included 'LC: I really need the toilet [MASK]', then we would first know that the speaker is called LC. The turn also ends with [MASK], so we know that we need to replace it with a goal. We know that LC needs the toilet, so their goal is to go to the nearest toilet. Goals always begin with the '@' symbol, and then a 'G' if we have found a person's goal. We would therefore replace [MASK] with @ G(LC, go-to(toilet)). If someone tells LC where the toilets are, they have answered their goal. We would therefore annotate that turn with @ AG(LC, go-to(toilet)). We use AG here to indicate Answer Goal. Finally, if LC then said thank you, we know their goal has been met. We would annotate the thank you with @ CG(LC, go-to(toilet)) because LC's goal is finished. CG stands for Close Goal. Do this goal tracking for each [MASK] in this conversation. Here are some examples. " + src
    return prompt

def create_examples(tdf):
    final_text = ""
    for x in range(0, 2): #len(tdf)-1):
        final_text = final_text + "input: " + tdf["src"][x] + " output: " + tdf["tgt"][x] + " "
    final_text = final_text + "input: "
    return final_text

def predict(src, few, style):
    if style == "basic":
        msgs = [{"role": "user", "content": generate_basic_prompt(src, few)}]
    elif style == "specific":
        msgs = [{"role": "user", "content": generate_specific_prompt(src, few)}]
    elif style == "annotation":
        msgs = [{"role": "user", "content": generate_annotation_instructions_prompt(src, few)}]
    elif style == "story":
        msgs = [{"role": "user", "content": generate_story_prompt(src, few)}]
    elif style == "roleplay":
        msgs = [{"role": "user", "content": generate_roleplay_prompt(src, few)}]
    elif style == "reasoning":
        msgs = [{"role": "user", "content": generate_reasoning_prompt(src, few)}]
    print(msgs)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = msgs,
        temperature=0.4
        )
    return response.choices[0].message.content

def process(style, few, run):
    test_data_path = "../Preprocessing/data/split-diogs/pcent-20/goal-only/run" + run + "-test.jsonl"
    test_df = pd.read_json(test_data_path, lines=True)

    if few:
        train_data_path = "../Preprocessing/data/split-diogs/pcent-20/goal-only/run" + run + "-train.jsonl"
        train_df = pd.read_json(train_data_path, lines=True)

    predictions = []

    for row in test_df['src']:
        if few:
            shots = create_examples(train_df)
            row = shots + row + " output: "
        try:
            prediction = predict(row, few, style)
        except openai.error.RateLimitError:
            print("Hit Rate Limit")
            sleep(120)
            try:
                prediction = predict(row, few, style)
            except openai.error.RateLimitError:
                print("Hit Second Rate Limit")
                sleep(500)
                prediction = predict(row, few, style)
        # print("#########################")
        # print(row)
        print("-------------------------")
        print(prediction)
        print("#########################")
        predictions.append(prediction)
        sleep(45)

    test_df['preds'] = predictions
    test_df.to_csv('few-shot-goal-only-' + style + '-run' + run + '.csv') #TODO, change each experiment

styles = ["basic", "specific", "story", "roleplay", "annotation", "reasoning"]
for style in styles:
    process(style, True, str(1))
    process(style, True, str(2))
    process(style, True, str(3))