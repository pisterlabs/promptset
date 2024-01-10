import openai

# set GPT-3 API key from the environment vairable
openai.api_key = OPENAI_API_KEY

# GPT-3 completion questions
prompt_questions = "Make a list of astronomical observatories:"

openai.Completion.create(
    model = "text-davinci-003",
    prompt="Make a list of astronomical observatories:"
)