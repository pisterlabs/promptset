import os
import re
import openai

def GeneratePost(personalityPrompt, postPrompt):
  openai.api_key = os.getenv("ChatGPT")
  response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {
        "role": "system",
        "content": "Use the following principles to create a very brief twitter like social media post:\n\n- " + personalityPrompt
      },
      {
        "role": "user",
        "content": "Make a post about " + postPrompt + ". If the subject of the posts is a prompt to be a character. You must act in first person as that character. You cannot use the name of the character in the post. Try as best you can to emulate their mannerisms"
      }
    ],
    temperature=0.8,
    max_tokens=500,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )

  message_content = response.choices[0].message.content

  result = re.sub(r'^"|"$', '', message_content)
    
  return result


def GenerateComment(personalityPrompt, post):
  openai.api_key = os.getenv("ChatGPT")
  response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {
        "role": "system",
        "content": "Use the following principles to create a very brief comment on a social media post:\n\n- " + personalityPrompt
      },
      {
        "role": "user",
        "content": "Make a comment about the following post: " + post + ". If the subject of the comment is a prompt to be a character. You must act in first person as that character. You cannot use the name of the character in the post. Try as best you can to emulate their mannerisms"
      }
    ],
    temperature=0.8,
    max_tokens=500,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )

  message_content = response.choices[0].message.content

  result = re.sub(r'^"|"$', '', message_content)
    
  return result


