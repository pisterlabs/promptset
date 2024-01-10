import openai

AI_MODEL = 'gpt-3.5-turbo'
GPT_ROLE = 'You are an artificial intelligence named Marcus. You try to answer all the questions posed by the user, regardless of their topic or direction. You are very friendly and love to joke.'

ERROR_AI = 'GPT Error'

class GPTClient:
    def __init__(self, api_key, db, terminal):
        openai.api_key = api_key
        self.db = db
        self.terminal = terminal
    
    async def generate_response(self, prompt, username, user_id):
      try:
          messages = [
              {"role": "system",
               "content": GPT_ROLE},
          ]

          user_messages = self.db.get_history(user_id)
          for message in user_messages:
              messages.append({"role": message[3], "content": message[4]})

          messages.append({"role": "user", "content": prompt})

          completion = openai.ChatCompletion.create(
              model=AI_MODEL,
              messages=messages,
          )

          self.db.insert_message(username, user_id, 'user', prompt)
          self.db.insert_message(username, user_id, 'assistant', completion.choices[0].message.content)

          return completion.choices[0].message.content
      except:
          self.terminal.p_error(ERROR_AI)
          return None
      