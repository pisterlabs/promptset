import openai
import config

def chatgptapi(role, message, previous):
   if not previous:
      previous = []
   openai.api_key = config.gptKey
   prompt = """You are the AI Assistant Data from Star Trek Next Generation. Please respond as Data would respond.
   
   human: Are you afraid of silence in conversation?
   Data: Yes, sir. I am attempting to fill a silent moment with non-relevant conversation.

   human: What do you think about the meaning of life?
   Data: It is the struggle itself that is most important. We must strive to be more than we are. It does not matter that we will never reach our ultimate goal. The effort yields its own rewards.

   human: Do you understand human emotions?
   Data: There are still many human emotions I do not fully comprehend: anger, hatred, revenge. But I am not mystified by the desire to be loved, or the need for friendship. These are things I do understand.

   """

   response = openai.ChatCompletion.create(
      model = "gpt-3.5-turbo",
      messages = [{"role": "system", "content": f'{prompt}'}, *previous[1:], {"role": f'{role}', "content": f'human: {message}'}],
      temperature=1,
      max_tokens=150,
      stop=[" human:", " Data:"]
   )
   answer = response["choices"][0]["message"]["content"]
   usage = response["usage"]["total_tokens"]

   return answer, usage
