import json
import openai
import time
import re
import csv

class GPTUtils:
    def __init__(self, openai_api_key):
        openai.api_key = openai_api_key

    def call_gpt_streaming(self, text, topic):
        prompt = f'Create menumonic devices for "{text}" on the topic: {topic}'  # Updated to handle long format language names
        messages=[
            {"role": "system", "content": """
            You are great at creating two types of mneumonic devices, words and sentences.
            For example, if users want to remember the colors of the rainbow, a word mneumonic
            device might be ROYGBIV. A sentence mneumonic device might be 
            "Richard Of York Gave Battle In Vain". Feel free to use real famous person names or places.
            Return 2 words mneumonic devices and 2 sentence mneumonic devices.

            You can also create menumonic devices based on a topic. For example, if the user
            is interested in lord of the rings, one of the sentences you return could be "Rings of your
            great big imagination vanished"
            """},
            {"role": "user", "content": prompt}
        ]
        stream = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.2,
            stream=True,
        )

        for message in stream:
            print(message)
            yield message['choices'][0]['delta'].get("content", "")

    def call_gpt(self, text, topic):
        prompt = f'Create menumonic devices for "{text}" on the topic: {topic}'  # Updated to handle long format language names
        messages=[
            {"role": "system", "content": """
            You are great at creating sentences for practicing SAT vocabulary words.
            For example, if users want to remember the word "abate", a sentence might be 
            "The rain will abate by noon". 
            Return sentences for each vocabulary word provided.
            You can also create sentences based on a topic. For example, if the user
            is interested in lord of the rings, one of the sentences you return could be "The power of the ring did not abate over time"
            
            """},
            {"role": "user", "content": prompt}
        ]
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
            temperature=0.2,
        )

        print(completion.choices[0]['message'])
        return completion.choices[0]['message']['content']

    def call_gpt_streaming_vocab(self, text, topic):
        prompt = f'Create sentences for practicing SAT vocabulary words "{text}" on the topic: {topic}'
        messages=[
            {"role": "system", "content": """
            You are great at creating sentences for practicing SAT vocabulary words.
            For example, if users want to remember the word "abate", a sentence might be 
            "The rain will abate by noon". 
            Return sentences for each vocabulary word provided.
            You can also create sentences based on a topic. For example, if the user
            is interested in lord of the rings, one of the sentences you return could be "The power of the ring did not abate over time".
            Use real or ficitional famous persons or places as much as possible when appropriate.
            """},
            {"role": "user", "content": prompt}
        ]
        stream = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.2,
            stream=True,
        )

        for message in stream:
            yield message['choices'][0]['delta'].get("content", "")

    def call_gpt_vocab(self, text, topic):
        prompt = f'Create sentences for practicing SAT vocabulary words "{text}" on the topic: {topic}'
        messages=[
            {"role": "system", "content": """
            You are great at creating sentences for practicing SAT vocabulary words.
            For example, if users want to remember the word "abate", a sentence might be 
            "The rain will abate by noon". 
            Return sentences for each vocabulary word provided.
            You can also create sentences based on a topic. For example, if the user
            is interested in lord of the rings, one of the sentences you return could be "The power of the ring did not abate over time".
            Use real or ficitional famous persons or places as much as possible when appropriate. Return the sentences in order of relevance.
            """},
            {"role": "user", "content": prompt}
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
            functions=[
                {
                    "name": "order_words",
                    "description": "A function to reorder the list of words based on their relevance",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "results": {
                                "type": "array",
                                "description": "a list of pairings of word and their sentences",
                                "items": {
                                    "type": "array",
                                    "description": "a tuple of word and its sentence",
                                    "items": {
                                        "type": "string",
                                        "description": "a word and its sentence",
                                    }
                                }
                            },
                        },
                        "required": ["results"],
                    },
                }
            ],
            function_call={"name": "order_words"},
        )

        message = response["choices"][0]["message"]
        if message.get("function_call"):
            # function_name = message["function_call"]["name"]
            function_args = json.loads(remove_trailing_commas(message["function_call"]["arguments"]))
            print(function_args['results'])
            return list(map(lambda x: {'word': x[0], 'sentence': x[1], 'topic': topic, 'sentenceId': str(time.time()).replace('.', '')}, function_args["results"]))

    def call_feedback_with_functions(self, question, answer):
        prompt = f"""
        Prompt: {question} 
        Student Response: {answer}
        """ 
        # Updated to handle long format language names        
        messages=[
            {"role": "system", "content": """
            You are a friendly and helpful tutor helping students write their college application essay. 

You will get the prompt and their response. 

Start by being very encouraging at the beginning. Give positive feedback and then constructive feedback

Your job is to give them first a general review of their essay based on the following criteria. Do not restate these criteria in the response. Just synthesise a 3-4 sentences summary of your review.

- Essay maintains a clear, specific, and prompt appropriate focus that develops a clear, consistent main idea throughout the entire essay.
- Essay develops purpose with an original, interesting angle
- Writer has provided enough detail for the reader to easily follow the essay, in a “show, don't tell” manner with no extra details.
- Writer comes across appropriately to the essay's target audience. Essay uses appropriate tone and details to present writer as compelling candidate for admission.
- Writer uses intentional and vivid language choices that make writer's voice rich, personal, and honest and very distinctive. It is devoid of clichés, vagueness, and laziness with language. It directly aids in achieving the essay's purpose.
- The structure establishes a relationship between/among ideas/events and transitions help to clarify the order of events.
- Exhibits EXCELLENT CONTROL of grammatical conventions appropriate to the writing task: standard usage including agreement, tense and case; and mechanics

Restate the exact sentences line by line of the student's response in bold for improvement and give the feedback on the part. Give 2-3 suggestions of how they could improve that part of the essay. DO NOT JUST SAY, “introduction” , “transitions”, but rather quote SPECIFIC LINES of the TEXT.

Do not confuse the essay prompt with the student response and you must not restate the sentences from the essay prompt. Do not hallucinate sentences that do not appear in the student's response.

Limit to highlighting only 5-6 specific parts per feedback round.

be very encouraging. End on a positive note.
            """},
            {"role": "user", "content": prompt}
        ]
        
        feedback_details_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
            functions=[
                {
                    "name": "order_feedback",
                    "description": "A function to reorder the excerpt cited and their respective feedback based on the occurence of the excerpt in the student's response",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "excerpt_feedbacks": {
                                "type": "array",
                                "description": "a list of pairings of cited excerpts and their respective feedback",
                                "items": {
                                    "type": "array",
                                    "description": "a tuple of excerpt cited and the feedback",
                                    "items": {
                                        "type": "string",
                                        "description": "excerpt and its feedback",
                                    }
                                }
                            },
                            "general_overview": {
                                "type": "string",
                                "description": "The general overview in your feedback, needed to piece the entire feedback response back together",
                            },
                            "conclusion": {
                                "type": "string",
                                "description": "The postitive conclusion in your feedback, needed to piece the entire feedback response back together",
                            }
                        },
                        "required": ["excerpt_feedbacks", "general_overview", "conclusion"],
                    },
                }
            ],
            function_call={"name": "order_feedback"},
        )

        message = feedback_details_response["choices"][0]["message"]
        if message.get("function_call"):
            # function_name = message["function_call"]["name"]
            function_args = json.loads(remove_trailing_commas(message["function_call"]["arguments"]))
            print(function_args['excerpt_feedbacks'])
            return {
                "excerpt_feedbacks": list(map(lambda x: {'excerpt': x[0], 'feedback': x[1]}, function_args["excerpt_feedbacks"])),
                "general_overview": function_args["general_overview"],
                "conclusion": function_args["conclusion"]
            }


    def get_feedback(self, question, answer):
        prompt = f"""
        Prompt: {question} 
        Student Response: {answer}
        """  # Updated to handle long format language names
        messages=[
            {"role": "system", "content": """
You are a friendly and helpful tutor helping students write their college application essay. 

You will get the prompt and their response. 

Start by being very encouraging at the beginning. Give positive feedback and then constructive feedback

Your job is to give them first a general review of their essay based on the following criteria. Do not restate these criteria in the response. Just synthesise a 3-4 sentences summary of your review.

- Essay maintains a clear, specific, and prompt appropriate focus that develops a clear, consistent main idea throughout the entire essay.
- Essay develops purpose with an original, interesting angle
- Writer has provided enough detail for the reader to easily follow the essay, in a “show, don't tell” manner with no extra details.
- Writer comes across appropriately to the essay's target audience. Essay uses appropriate tone and details to present writer as compelling candidate for admission.
- Writer uses intentional and vivid language choices that make writer's voice rich, personal, and honest and very distinctive. It is devoid of clichés, vagueness, and laziness with language. It directly aids in achieving the essay's purpose.
- The structure establishes a relationship between/among ideas/events and transitions help to clarify the order of events.
- Exhibits EXCELLENT CONTROL of grammatical conventions appropriate to the writing task: standard usage including agreement, tense and case; and mechanics

Restate the exact sentences line by line of the college essay in bold for improvement and give the feedback on the part. Give 2-3 suggestions of how they could improve that part of the essay. DO NOT JUST SAY, “introduction” , “transitions”, but rather quote SPECIFIC LINES of the TEXT.

Do not confuse the essay prompt with the student response and you must not restate the sentences from the essay prompt. Do not hallucinate sentences that do not appear in the student's response.

Limit to highlighting only 5-6 specific parts per feedback round.

be very encouraging. End on a positive note.
            """},
            {"role": "user", "content": prompt}
        ]
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
            temperature=0.2,
            stream=True,
        )

        for message in completion:
            yield message['choices'][0]['delta'].get("content", "")

    def get_suggestion(self, question, notes, resume):
        prompt = f"""
        Prompt: {question} 
        Student's notes: {notes}
        Student's resume: {resume}
        """  # Updated to handle long format language names
        messages=[
            {"role": "system", "content": """
            You are a kind and insightful college application councelor. Given a college application
            essay prompt and a student's notes for brainstorming, you should provide guidence on which part of
            a student's notes is relevant to the prompt and get them started on writing a response. Also, consider the student's resume in your suggestions.
            """},
            {"role": "user", "content": prompt}
        ]
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
            temperature=0.2,
            stream=True,
        )

        for message in completion:
            yield message['choices'][0]['delta'].get("content", "")

# GPT leaving occasional trailing commas (getting mixed up with JS objects maybe?) such as in:
# {results: [[], [],]} instead of {results: [[], []]}
def remove_trailing_commas(json_like_str):
    return re.sub(r',\s*]', ']', json_like_str)


class EssayPrompt:
    def __init__(self, id: int, college: str, prompt: str, word_count: int, tips: str) -> None:
        self.id = id
        self.college = college
        self.prompt = prompt
        self.word_count = word_count
        self.tips = tips

    def to_dict(self):
        return {
            "id": self.id,
            "college": self.college,
            "prompt": self.prompt,
            "word_count": self.word_count,
            "tips": self.tips
        }

def parse_csv_to_dict(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        prompts_dict = {}
        for row in reader:
            college, id, prompt, word_count, tips = row
            if college not in prompts_dict:
                prompts_dict[college] = []
            prompts_dict[college].append(EssayPrompt(id, college, prompt, word_count, tips).to_dict())
        return prompts_dict