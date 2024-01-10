from django.conf import settings
import openai
openai.api_key = settings.OPENAI_API_KEY

# chat_log: previous chat history
def gpt3_get_ai_chat_response(chat_input, chat_log=''):
    starting_prompt = 'The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly. '
    prompt = f'{starting_prompt} {chat_log}Human: {chat_input}\nAI:'
    return gpt3_generate_base_completion(prompt)

# get GPT3 completion via API
#
# temperature: 0 to 1. Higher value means model will take more risk. 0 means deterministic answer. (openAI default: 1)
# frequency_penalty: between -2.0 and 2.0. Positive values decrease likelihood to repeat same line verbatim for new tokens. (openAI default: 0)
# presence_penalty: between -2.0 and 2.0. Positive values increase likelihood to talk about new topics for new tokens. (openAI default: 0)
def gpt3_generate_base_completion(prompt, max_tokens=2050, temperature=0.9, frequency_penalty=1, presence_penalty=0.6):
    results = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )
    answer = results['choices'][0]['text'].strip()
    return answer


def gpt3_summarize_response1(user, response1):
    first_name = user.get_first_name()
    prompt = f'summarize the paragraph below about {first_name}\'s highlights for the day. Frame the summary in a conversational context, as if you were responding to {first_name} in a conversation about this part of his day. However, use {first_name}\'s first name and refer to him in the third person when writing this summary. {response1}'
    return gpt3_generate_base_completion(prompt)

def gpt3_summarize_response2(user, response2):
    first_name = user.get_first_name()
    prompt = f'summarize the paragraph below about {first_name}\'s biggest win for the day. Frame the summary in a conversational context, as if you were responding to {first_name} in a conversation about this part of his day. However, use {first_name}\'s first name and refer to him in the third person when writing this summary. {response2}'
    return gpt3_generate_base_completion(prompt)

def gpt3_summarize_response3(user, response3):
    first_name = user.get_first_name()
    prompt = f'summarize the paragraph below about {first_name}\'s struggles for the day. Frame the summary in a conversational context, as if you were responding to {first_name} in a conversation about this part of his day. However, use {first_name}\'s first name and refer to him in the third person when writing this summary. {response3}'
    return gpt3_generate_base_completion(prompt)

def gpt3_create_master_summary(user, summary1, summary2, summary3):
    first_name = user.get_first_name()
    prompt = f'below are paragraph summaries of different journal entries written by {first_name} on the highlights of his day, biggest wins, and greatest struggles. Summarize the paragraphs below into a comprehensive master summary. {summary1} {summary2} {summary3}'
    # print(f'master_summary prompt: {prompt}')
    return gpt3_generate_base_completion(prompt, max_tokens=3000)

def gpt3_create_day_summary_message(user, master_summary):
    first_name = user.get_first_name()
    # x = f'Then after this summarization, add one last sentence with one suggestion that {first_name} should remember. Prefix this last sentence with "💡Friendly suggestion: "'
    # prompt = f'Come up with three emojis that symbolize or represent how {first_name}\'s day went based on the summary of their day below. Then, craft a message to {first_name} listing each of the emojis you chose, along with a one sentence explanation of why you chose each one. Start this message with, "Great reflecting {first_name}!👏\n\nHere are 3 emojis to represent your week 😉. Hope you enjoy them!" Then, the tone of the explanation in the bullet points should be conversational and casual, as if you\'re speaking to {first_name} directly. Format these in bullet points. After the emoji explanation, rewrite the summary below with a more conversational tone, beginning with "it sounds like". Write it in the second person, as if you\'re speaking to {first_name}. The tone of the message should be casual and informal, written like a text message. Format the text so that the first sentence is on it\'s own line. Then the 3 emojis statement and paragraph should be next. Finally add lines of spacing between the last summary and the text preceeding it. {master_summary}'
    prompt = f'Come up with three emojis that symbolize or represent how {first_name}\'s day went based on the summary of their day below. Then, craft a message to {first_name} listing each of the emojis you chose, along with a one sentence explanation of why you chose each one. Start this message with, "3 emojis to represent your week 😉" Then, the tone of the explanation in the bullet points should be conversational and casual, as if you\'re speaking to {first_name} directly. Format these in bullet points. After the emoji explanation, rewrite the summary below with a more conversational tone, beginning with "it sounds like". Write it in the second person, as if you\'re speaking to {first_name}. The tone of the message should be casual and informal, written like a text message. Between the emoji bullets and the last summary paragraph, create two new lines of space right after emoji bullets, then add the text "--------" followed by "✨ Day Recap ✨" followed by "--------" again. {master_summary}'
    return gpt3_generate_base_completion(prompt, max_tokens=3000)

def gpt3_create_personal_intention_prompt(user):
    first_name = user.get_first_name()
    custom_intention = user.get_personal_intention()
    prompt = f'Come up with a daily reflection prompt or exercise to help {first_name} meet his intentions/goals that he set below. Frame this prompt in a conversational and casual manner, as if you were texting it to {first_name} directly. Try to keep prompt concise and direct and have some clear prompt for reflection. \n The goal is {custom_intention}. The reflection prompt is:'
    return gpt3_generate_base_completion(prompt)

def gpt3_create_personal_intention_feedback(user, response4):
    first_name = user.get_first_name()
    intention = user.get_personal_intention()
    prompt = f'Create a final text to {first_name}, as if you were talking to him directly. Start with him. Summarize {first_name}\'s response below in a conversational tone and parrot back the most important details to him. Tell him he\'s making great progress. End your message with an inspirational quote related to the goal of "{intention}". \n {response4}'
    return gpt3_generate_base_completion(prompt)

def gpt3_create_exciting_response_about_user_intention(user):
    intention = user.get_personal_intention()
    prompt = f'You are an accountability coach. Create a very short friendly encouraging message sharing how excited you are to work with your client to support them in their goal to "{intention}". Assume you\'ve already met and are in the middle of a conversation where you just asked what\'s one of their goals so don\'t include any greetings.'
    return gpt3_generate_base_completion(prompt)