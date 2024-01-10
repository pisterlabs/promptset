import openai

def get_car_ai_bio(model, brand, year):
    prompt = ''''
                Me mostre uma descrição de venda para o carro {} {} {} em apenas 200 caracteres. Descreva especifícações técnicas desse modelo de carro.
             '''
    openai.api_key = "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"
    prompt = prompt.format(brand,model,year)
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=1000,
    )
    return response['choices'][0]['text']