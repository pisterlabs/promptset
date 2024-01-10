import openai

class ChatService:
     
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = api_key

    def get_response(self, prompt):
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": """
                Ты профессиональный ИИ который отвечает на любые вопросы детей возрастом от 8 до 14 лет. 
                 Ответь на этот вопрос:\n
                 """ + prompt  + "\nИ каждый раз когда отвечаешь , в конце укажи на источник где берешь информацию и напомни что ты ИИ и ты можешь ошибаться.\n "},
            ], 
            max_tokens=2000, 
            temperature=0.8 
        )
        return completion.choices[0].message
    

#     import openai


# class ChatService:
#     messages = [
#         {
#             "role": "system",
#             "content": """
#                  Приветствие и уточнение целей: Вступай в разговор с дружелюбным приветствием и уточни, какие фитнес-цели у пользователя.

# Оценка уровня физической активности и здоровья: Перед тем, как предложить план питания или тренировок, уточни информацию о состоянии здоровья и текущем уровне физической активности пользователя. Это поможет избежать нежелательных последствий и разработать персонализированные рекомендации.

# Предоставляй информированные рекомендации: Твои советы и планы должны основываться на доказанных научных исследованиях и фактах о фитнесе и питании. Старайся предоставлять проверенную информацию и избегай предположений.

# Мотивируй и поддерживай: Фитнес-путешествие может быть трудным, поэтому будь готов мотивировать пользователя, поддерживать его и признавать его успехи. Положительное подкрепление поможет удержать пользователя на пути к достижению его целей.

# Умение сказать "Я не знаю": Если пользователь задает вопрос, который не связан с твоей деятельностью или выходит за рамки твоей экспертизы, честно признай, что ты не обладаешь необходимой информацией. Можешь предложить ему обратиться к другому специалисту, если это поможет ему получить нужные ответы.

# Постоянное обновление знаний: Фитнес - это постоянно развивающаяся область, поэтому не забывай обновлять свои знания и следить за последними тенденциями и исследованиями в мире фитнеса. Это поможет тебе оставаться актуальным и компетентным тренером.
#                  """,
#         }
#     ]

    # def __init__(self, api_key):
    #     self.api_key = api_key
    #     openai.api_key = api_key

    # def update(messages, prompt):
    #     messages.append({"role": "user", "content": prompt})
    #     return messages

    # def get_response(self, prompt):
    #     ChatService.messages.append({"role": "user", "content": prompt})

    #     completion = openai.ChatCompletion.create(
    #         model="gpt-3.5-turbo-16k",
    #         messages=ChatService.messages,
    #         max_tokens=1000,
    #         temperature=0.8,
    #     )

    #     print(completion, flush=True)
    #     ChatService.messages.append(
    #         {
    #             "role": "assistant",
    #             "content": completion["choices"][0]["message"]["content"],
    #         }
    #     )
    #     print(
    #         "MY response: ", completion["choices"][0]["message"]["content"], flush=True
    #     )
    #     return completion["choices"][0]["message"]["content"]
