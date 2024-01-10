import openai
import secretKeys

_promptTokens = 0  # tokens utilizzati per il prompt 0.0015 dollari
_completionTokens = 0  # tokens utilizzati per la completion 0.002 dollari
_totalTokens = 0
_budget = 0.05  # 5 centesimi di dollaro


def remainingTokens(balance, totalTokens):
    # Costo per 1000 tokens
    cost_per_1000_tokens = 0.002
    cost_per_token = cost_per_1000_tokens / 1000

    balance = balance - (cost_per_token * totalTokens)
    remaining_tokens = float(balance / cost_per_token)

    return balance, f"Ti rimangono ${balance:.4f}. Puoi ancora utilizzare {int(remaining_tokens)} tokens."


openai.api_key = secretKeys.openAi
model = "gpt-3.5-turbo"
# Cambia questa variabile per fare una domanda diversa
prompt = "Ciao Chat! Questo è un test per vedere se le API funzionano correttamente. Come va?"
messages = [
    {"role": "user", "content": f"{prompt}"},
]

response = openai.ChatCompletion.create(model=model, messages=messages)
print(response)
answer = response['choices'][0]['message']['content']
_promptTokens += int(response['usage']['prompt_tokens'])
_completionTokens += int(response['usage']['completion_tokens'])
_totalTokens += int(response['usage']['total_tokens'])

print(answer)
budget, remaining = remainingTokens(_budget, _totalTokens)
print(remaining)
