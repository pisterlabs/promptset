from openai import OpenAI
from server.prompts import formatValidateOccupationPrompt, formatFillParagraph, getPromptBySection

def ask_ai(prompt, maxTokens=50, model="gpt-3.5-turbo"):
  client = OpenAI()
  messages = [{"role": "user", "content": prompt}]
  response = client.chat.completions.create(
  model=model,
  messages=messages,
  max_tokens=maxTokens,
  temperature=0,
  )
  return response.choices[0].message.content

def validateOccupation(occupation):
  prompt = formatValidateOccupationPrompt(occupation)
  return(ask_ai(prompt))

def fillParagraph(occupation, section):
  sectionPrompt = getPromptBySection(occupation, section)
  prompt = formatFillParagraph(sectionPrompt)
  return(ask_ai(prompt))
