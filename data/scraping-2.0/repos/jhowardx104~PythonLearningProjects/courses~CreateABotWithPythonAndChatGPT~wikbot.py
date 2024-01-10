from open_ai_service import OpenAiService
from secret_service import get_secrets

secrets = get_secrets()

openai_service = OpenAiService(
    secrets['openai']['key']
)

user_input = ''
user_prompt = 'What topic page would you like to summarize?\n'
gpt_starter_prompt = 'Provide a 5 bullet point summary of the following topic: \n'

while user_input != '/quit':
    user_input = input(user_prompt)
    if user_input == '/quit':
        continue

    openai_service.append_sys_message('You are assisting a student by generating 1 sentence summaries in bullet point form for given prompts.')
    openai_service.append_user_message(gpt_starter_prompt + user_input)
    print('Tokens for prompt: ', openai_service.num_tokens_from_messages())
    openai_service.get_completion()
    openai_service.clear_messages()
