import tiktoken
import typing
import openai

# OpenAI Official Pricing Guide: https://openai.com/pricing
# Documentation: https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/chatgpt?pivots=programming-language-chat-completions#managing-conversations

# MODEL_GPT_35 = "gpt-3.5-turbo"
# GPT_35_PRICE = 0.002            # 0.002 / 1k tokens -> Input tokens + Output tokens price
# GPT_35_TOKEN_LIMIT = 4096
# GPT_35_MAX_RESPONSE_TOKENS = 250

MODEL_GPT_4 = "gpt-4-1106-preview"
GPT_4_PMPT_PRICE = 0.01         # 0.01 / 1k tokens -> Input price
GPT_4_CMPL_PRICE = 0.02         # 0.02 / 1k tokens -> Output price
GPT_4_TOKEN_LIMIT = 128000
GPT_4_MAX_RESPONSE_TOKENS = 550

SERVICE_PRICE = 0.03            # Discord server fee (Added by me)

# Given a message list, returns the total token size of the messages
def getTokenSizeFromMsgs(msgs: list, model = MODEL_GPT_4) -> int:
    # Use GPT-4 encoding
    token_size = 0
    encoding = tiktoken.encoding_for_model(MODEL_GPT_4)

    for msg in msgs:
        # Every message has a 'role' and a 'content' field, which is 4 tokens
        token_size += 4
        for key, val in msg.items():
            token_size += len(encoding.encode(val))

    # Every API response(output / sample / reply) is primed with an 'assistant' field behind the scenes
    token_size += 2
    return token_size

# Get answer from the OpenAI API and calculates the fee for messages
# This function is from another class (discord bot) and CANNOT be used directly
async def getAnswer(self, user_message: str, input_model) -> typing.Tuple[str, int, int, float]:
        # Append the current user message to the message context list
        self.msgs_gpt_4.append({"role": "user", "content": user_message})

        # Calculates the token size
        prev_token_size = price.getTokenSizeFromMsgs(self.msgs_gpt_4)

        # If the token size limit exceeded the limit (128K), remove the earliest message from the list,
        # keep doing this until the token size is within the limit
        while (prev_token_size + GPT_4_MAX_RESPONSE_TOKENS >= GPT_4_TOKEN_LIMIT):
            del self.msgs_gpt_4[1]
            prev_token_size = price.getTokenSizeFromMsgs(self.msgs_gpt_4)

        # Initialize parameters for later calculation
        res, pmpt_tokens, cmpl_tokens, prices = '', 0, 0, 0.0

        try:
            # Start request via the OpenAI API
            response = await openai.ChatCompletion.acreate(
                    model = input_model,
                    messages = self.msgs_gpt_4,
                    temperature = 0.8,
                    max_tokens = GPT_4_MAX_RESPONSE_TOKENS,
            )
            
            # Extract the response from ChatGPT
            res = response['choices'][0]['message']['content'].strip()

            # Price calculation
            self.gpt_4_pmpt_tokens = response['usage']['prompt_tokens']
            self.gpt_4_cmpl_tokens = response['usage']['completion_tokens']
            prices = \
                self.gpt_4_pmpt_tokens / 1000 * price.GPT_4_PMPT_PRICE + \
                self.gpt_4_cmpl_tokens / 1000 * price.GPT_4_CMPL_PRICE + \
                price.SERVICE_PRICE
            
            # Necessary for the integrity of the API conversation 
            self.msgs_gpt_4.append({"role": "assistant", "content" : res})
            
            # Set prompt & completion tokens
            pmpt_tokens = self.gpt_4_pmpt_tokens
            cmpl_tokens = self.gpt_4_cmpl_tokens

            print(f'ChatGPT: {res}')

        # Error handling
        except openai.error.APIError as e:
            if 'timed out' in str(e):
                print('[ERROR] Request timed out')
            else:
                print(f'[ERROR] {e}')
            
            res = str(e)
        
        return (res, pmpt_tokens, cmpl_tokens, prices)