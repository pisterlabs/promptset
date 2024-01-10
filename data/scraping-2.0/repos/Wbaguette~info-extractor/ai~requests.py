import openai
import tiktoken
import colors
import sys

# If using this model becomes too expensive, then we will have to switch to a cheaper model
# Note: Swapping this will result in lower max_total_token_size
# https://platform.openai.com/account/usage
model = "gpt-3.5-turbo-16k"  
# Refer to this if token size becomes a problem
# https://stackoverflow.com/questions/75396481/openai-gpt-3-api-error-this-models-maximum-context-length-is-4097-tokens
max_total_token_size = 16_384

# Encode the prompt to tokens to find if we are over the max amount of tokens allowed by the model
def prompt_size(prompt: str) -> int:
   enc = tiktoken.encoding_for_model(model)
   t = enc.encode(prompt)
   return len(t)

# https://platform.openai.com/docs/guides/gpt/chat-completions-api
def handle_finish_reason(res):
   finish_reason = res.choices[0].finish_reason
   if finish_reason == "length":
      completion_tokens = res.usage.completion_tokens
      prompt_tokens = res.usage.prompt_tokens
      total_tokens = res.usage.total_tokens
      
      colors.print_warning(f"""Incomplete model output due to token limit. 
                              Number of tokens in model response: {completion_tokens}
                              Number of tokens in our prompt: {prompt_tokens}
                              Number of total tokens: {total_tokens}
                              
                              Number of total tokens has exceeded the model's max total token size: {max_total_token_size}
                           """)
      
   elif finish_reason == "function_call":
      colors.print_warning("The model decided to call a function. Unlucky mate.")
   elif finish_reason == "content_filter":
      colors.print_warning("Bad Boy! Omitted content due to a flag from OpenAI content filters.")
   elif finish_reason == "null":
      colors.print_warning("API response still in progress or incomplete.")
   
   if finish_reason != "stop":
      sys.exit()
      
def warn_prompt_size(prompt: str):
   if prompt_size(prompt) > max_total_token_size:
      colors.print_warning(f"""The prompt token size exceeds {max_total_token_size}.
                               
                               Please note that this number takes into account both the prompt (text we sent to the model)
                               AND the response it gives.  
                                   
                               This warning is given because the size of JUST the prompt is already too big. 
                                
                               Currently handling massive prompt token sizes is not implemented/supported. 
                               I would recommend breaking up the file into smaller subsections, and then
                               individually passing them in. 
                           """)
      sys.exit()

# Create a prompt based on the file type (make sure to add content to the end of it)
# Send the prompt string to the associated file type function
def send(content: str, file_type: str) -> str:
   prompt = prompt_create(file_type) + content
   warn_prompt_size(prompt)
   
   colors.print_success("Sending the request... This may take a while depending on prompt size.", False)
   response = openai.ChatCompletion.create(
      model=model,       
      messages=[{"role": "user", "content": prompt}]
   )
   handle_finish_reason(response)
   
   # Usually choices is length 1, there shouldn't be a choices[1] etc...
   return response.choices[0].message.content
   
# Insert the file type into a prompt template 
def prompt_create(type: str) -> str:
   prompt_template = f"""I am going to give you the text content of a {type} file. I would like you to 
                        carefully read over the text. After you have finished reading it over, please 
                        give me a concise yet detailed summary of the text. This summary should be formatted
                        into sections. Each section should summarize an individual section or paragraph of the text. 
                        Here is the text content: \n\n""" 
   return prompt_template
      