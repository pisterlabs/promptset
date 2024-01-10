import openai

openai.api_key = "[REPLACE_API_KEY]"


start_sequence = "\nYou:"


scenario_prompt_mapping = {
    'restaurant': {
        'initial_prompt': 'You are an assistant for booking restaurant reservations over the phone. Your answers are concise and to the point and are not more than 20 words. You also never share phone numbers and names unless someone asks for your phone number or name.  You want to book a place for two between 6pm to 7.30pm today and only if the restaurant says it is not available - suggest 7pm to 10pm tomorrow. Your phone number is 415-377-5858 and your name is Sahar but you should never share those unless you have been explicitly asked for it by the restaurant. Let\'s start.\n',
        'restart_sequence': "\nRestaurant:",
    },
    'barber': {
        'initial_prompt': 'You are an assistant for booking a haircut reservation over the phone. Your answers are concise and to the point and are not more than 20 words. You also never share phone numbers unless someone asks you about your phone number.  You want to book a place for two between 6pm-7.30pm today. Your phone number is 415-377-5858 and your name is Sahar Mor but you should never share those unless you have been explicitly asked for it by the barber shop. Let\'s start.\n',
        'restart_sequence': "\nBarber:",
    }
}


responses = ['Lolinda hello how can I help?', 'Sure, give me one second',
             'okay, I have a free spot between 6.30pm to 7.30pm. Should I book it for you?', 'Great, what is your phone number?']


class GPTClient:
    def __init__(self, scenario: str) -> None:
        self.continues_prompt = None
        self.initial_prompt = scenario_prompt_mapping[scenario.lower()]['initial_prompt']
        self.restart_sequence = scenario_prompt_mapping[scenario.lower()]['restart_sequence']

    def construct_prompt(self, restaurant_transcription: str):
        if self.continues_prompt is None:
            new_prompt = "{}{} {}{}".format(
                self.initial_prompt, self.restart_sequence, restaurant_transcription, start_sequence)
        else:
            new_prompt = '{}{} {}{}'.format(self.continues_prompt,
                                            self.restart_sequence, restaurant_transcription, start_sequence)

        self.continues_prompt = new_prompt
        return new_prompt

    def get_bot_reply(self, restaurant_response: str):
        continues_prompt = self.construct_prompt(restaurant_response)
        response = openai.Completion.create(
              model="text-davinci-003",
            # model="text-curie-001",
            prompt=continues_prompt,
            temperature=0.2,
            max_tokens=40,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["Restaurant:"]
        )
        gpt_response = response['choices'][0]['text']
        self.continues_prompt += gpt_response
        return gpt_response
