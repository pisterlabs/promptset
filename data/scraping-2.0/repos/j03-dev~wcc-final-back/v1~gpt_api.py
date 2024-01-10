import openai

openai.api_key = "kBiVXOYpBlAOT94Pqs01MsYcS_Nhaz3CYQdNSvRda_Q"
openai.api_base = "https://chimeragpt.adventblocks.cc/v1"

def ask_gpt(request: str):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'user', 'content': str(request)},
        ]
    )

    return response