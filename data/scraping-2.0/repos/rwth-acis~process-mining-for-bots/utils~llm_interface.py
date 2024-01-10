from openai import OpenAI
import pm4py
import os


def recommendations_from_event_log(log):
    prompt = "Here is a DFG of a chatbot conversation. Performance is how long a request took on average in seconds. Frequency is how often this edge has been traversed.\n"
    prompt += pm4py.llm.abstract_dfg(log)
    prompt += "\nList five improvements that can be made to the chatbot? Split the recommendations into ones for bot developers and backend specialists. Format your response as html\n\n"
    return prompt


def recommendations_for_intents(intents_df):
    prompt = "Knowing that a confidence score bigger than 0.9 is considered good. nlu_fallback describes cases where intents were not recognized, which is also problematic \n\n"
    prompt += "Here is a list of intents and the average bot confidence score:\n\n"
    prompt += get_intent_list(intents_df)
    prompt += "\nWhat improvements can be made to the chatbot? Note that the training data is passed to Rasa to train. Also include the list along with the confidence scores as a table \n\n"
    prompt += "Format the response as html. \n\n"
    return prompt


def custom_prompt(inputPrompt, intents_df, log, net, initial_marking, final_marking):
    if ("`botModel`" in inputPrompt):
        replacement = pm4py.llm.abstract_petri_net(
            net, initial_marking, final_marking)
        prompt = replacePlaceholder(inputPrompt, "`botModel`", replacement)

    if ("`botIntents`" in inputPrompt):
        prompt = replacePlaceholder(
            inputPrompt, "`botIntents`", get_intent_list(intents_df))

    if ("`botLog`" in inputPrompt):
        replacement = pm4py.llm.abstract_dfg(log)
        prompt = replacePlaceholder(inputPrompt, "`botLog`", replacement)

    return prompt


def find_subprocesses(net, initial_marking, final_marking):
    prompt = "Here is a chatbot conversation model:\n\n"
    prompt += pm4py.llm.abstract_petri_net(net, initial_marking, final_marking)
    prompt += "\n\What subprocesses can you identify for this chatbot?\n\n"
    return prompt


def describe_bot(net, initial_marking, final_marking):
    prompt = "Here is a chatbot conversation model:\n\n"
    prompt = pm4py.llm.abstract_petri_net(net, initial_marking, final_marking)
    prompt += "\n\Describe all the things that this bot can do\n\n"
    return prompt


def send_prompt(prompt, api_key, openai_model="gpt-3.5-turbo-1106"):
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(model=openai_model,
                                              messages=[
                                                  {"role": "system", "content": "You are a helpful Process Mining Expert. You are helping users improve their chatbot. Petri nets refer to the chatbot conversation model. DFG refers to the chatbot conversation model. Furthermore, you should only return html. dont include ```html"},
                                                  {"role": "user",
                                                   "content": prompt}
                                              ])
    content = response.choices[0].message.content
    return content


def replacePlaceholder(prompt, placeholder, replacement):
    return prompt.replace(placeholder, replacement)


def get_intent_list(intents_df):
    res = ""
    for index, row in intents_df.iterrows():
        if row['intentKeyword'] is not None:
            res += f"{row['intentKeyword']}: {row['averageConfidence']}\n"
    return res
