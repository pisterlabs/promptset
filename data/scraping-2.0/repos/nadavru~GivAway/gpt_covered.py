import openai

class ChatGPTWrapper:
    def _init_(self, api_key):
        self.api_key = api_key
        openai.api_key = self.api_key
    
    def get_reply(self, prompt):
        # Make an API call to OpenAI's GPT-3
        response = openai.Completion.create(
          engine="text-davinci-002",  # Use "text-davinci-002" or any other available engine
          prompt=prompt,
          max_tokens=100  # Limit the response to 100 tokens
        )

        # Extract and return the text from the API response
        return response['choices'][0]['text'].strip()
    
    def is_he_coverd(self, Story, GCI_text, policy_text):
        prompt =GCI_text+ policy_text+ Story + "\n\nQ: Is he covered?\nA: "
        reply = self.get_reply(prompt)
        if reply == "Yes":
            return True
        else:
            return False