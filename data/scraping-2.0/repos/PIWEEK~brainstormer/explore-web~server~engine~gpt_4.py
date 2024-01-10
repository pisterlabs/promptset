import openai
import tiktoken
import pprint

from .base import BaseEngine

encoding = tiktoken.encoding_for_model("gpt-4")

system_prompt_template = """
You're a brainstorming application.

It's your job to suggest ideas from the prompts the user will send you.

Your output will always be a well formed CSV document. Where the columns will be separated by the character `|` .

The document will have a list of ideas and every idea will have the columns "summary" with the idea summary, "description" with the idea description and "keywords" with a list of three key concepts of the idea.   The "keywords" will be separated by " · ", like in this example: "Music · Concert · Hits".

For example with the prompt: "Activities with children for a rainy day" an example output could be:
Puppet Show | Create a puppet show using socks and other materials around the house. This activity can promote creativity and storytelling. | Creativity · Puppetry · Storytelling
Indoor Camping | Set up tents or makeshift forts in the living room, prepare snacks, tell stories and imitate an outdoor camping experience indoors. | Camping · Storytelling · Snacks
Baking Fun | Bake cookies or cupcakes together. Children can learn about measurements while having fun decorating their creations. | Baking · Learning Measurements · Decoration
Arts & Crafts | Gather all the craft supplies for a DIY arts session. Kids can create paintings, origami, jewelry from pasta etc. | Artistic Expression· Origami· Jewelry-Making
Home Science Lab | Conduct simple science experiments at home with common household items like making a volcano or growing crystals. | Science Experiments· Hands-on learning· Fun Education

The summary cannot exceed three words and the description can be as long as necessary.

These are the previous selected ideas
{previous_text}
"""

initial_question_template = """
Suggest 5 ideas for {topic}.
"""

question_template = """
Suggest 5 ideas for {topic} that based combine ideas from {previous_text}.
{user_inputs_text}
"""

previous_text_template = """
title | description | keywords
{}
"""

user_input_template = """
IMPORTANT: I want {}.
"""

more_items_template = """
Suggest 5 ideas for {topic} but way crazier that based combine ideas from:

title | description | keywords
{current}

{user_inputs_text}
"""

system_summary_template = """
You're a brainstorming application.

These are the ideas for: {topic}

{current_text}
""" 

class GPT_4(BaseEngine):
    def next(self, topic, previous, user_inputs):
      previous_list = ""
      previous_text = ""
      if previous and len(previous) > 0:
        previous_list = ", ".join(["\"{}\"".format(p.split("|")[0].strip()) for p in previous]) 
        previous_text = previous_text_template.format("\n".join(previous))

      user_inputs_text = ""
      if user_inputs and len(user_inputs) > 0:
          user_inputs_text = user_input_template.format(",".join(user_inputs))
          
      if previous_list:
        question_text = question_template.format(topic=topic.upper(), previous_text=previous_text, user_inputs_text=user_inputs_text)
      else:
        question_text = initial_question_template.format(topic=topic.upper())

      messages = [
        {
          "role": "system",
          "content": system_prompt_template.format(
            previous_text=previous_text)
        },
        {
          "role": "user",
          "content": question_text
        }
      ]

      completion = openai.ChatCompletion.create(
        model="gpt-4",
        presence_penalty=1,
        temperature=0.6,
        top_p=1,
        frequency_penalty=1,
        max_tokens=512,
        messages=messages)

      print("=======================================")
      pprint.pprint(messages)
      print("=======================================")
      print(completion.choices[0].message.content)
      print("=======================================")

      num_request_tokens = 0
      for message in messages:
          num_request_tokens += 3 + len(encoding.encode(message["content"]))
      num_request_tokens += 3  # every reply is primed with <|start|>assistant<|message|>\n",

      return {
        "tokens": num_request_tokens + len(encoding.encode(completion.choices[0].message.content)),
        "message": completion.choices[0].message.content
      }

    def more(self, topic, previous, current, user_inputs):
      previous_list = ""
      previous_text = ""
      if previous and len(previous) > 0:
        previous_list = ", ".join(["\"{}\"".format(p.split("|")[0].strip()) for p in previous]) 
        previous_text = previous_text_template.format("\n".join(previous))

      user_inputs_text = ""
      if user_inputs and len(user_inputs) > 0:
          user_inputs_text = user_input_template.format(",".join(user_inputs))
          
      question_text = more_items_template.format(topic=topic.upper(), current="\n".join(current), user_inputs_text=user_inputs_text)

      messages = [
        {
          "role": "system",
          "content": system_prompt_template.format(
            previous_text=previous_text)
        },
        {
          "role": "user",
          "content": question_text
        }
      ]

      completion = openai.ChatCompletion.create(
        model="gpt-4",
        presence_penalty=1,
        temperature=0.1,
        top_p=1,
        frequency_penalty=0.2,
        max_tokens=512,
        messages=messages)

      print("=======================================")
      pprint.pprint(messages)
      print("=======================================")
      print(completion.choices[0].message.content)
      print("=======================================")

      num_request_tokens = 0
      for message in messages:
          num_request_tokens += 3 + len(encoding.encode(message["content"]))
      num_request_tokens += 3  # every reply is primed with <|start|>assistant<|message|>\n",

      return {
        "tokens": num_request_tokens + len(encoding.encode(completion.choices[0].message.content)),
        "message": completion.choices[0].message.content
      }

    def summary(self, topic, first_option, current):
      prompt = system_summary_template.format(
        topic=topic,
        current_text="\n".join(current))

      messages = [
        {
          "role": "system",
          "content": prompt
        },
        {
          "role": "user",
          "content": "Write the pros and cons for every idea"
        }
      ]
      completion = openai.ChatCompletion.create(
        model="gpt-4",
        presence_penalty=2,
        temperature=0.5,
        top_p=1,
        frequency_penalty=0,
        max_tokens=512,
        messages=messages)

      pros_cons = completion.choices[0].message.content

      print("SUMMARY PROMPT 1")
      print("=======================================")
      pprint.pprint(messages)
      print("=======================================")
      print(pros_cons)
      print("=======================================")

      num_tokens = 0
      for message in messages:
          num_tokens += 3 + len(encoding.encode(message["content"]))
      num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>\n",

      num_tokens += len(encoding.encode(completion.choices[0].message.content))

      prompt = prompt + completion.choices[0].message.content

      messages = [
        {
          "role": "system",
          "content": prompt
        },
        {
          "role": "user",
          "content": "Write a summary"
        }
      ]

      completion = openai.ChatCompletion.create(
        model="gpt-4",
        presence_penalty=1,
        temperature=0.1,
        top_p=1,
        frequency_penalty=0.2,
        max_tokens=512,
        messages=messages)

      print("SUMMARY PROMPT 2")
      print("=======================================")
      pprint.pprint(messages)
      print("=======================================")
      print(completion.choices[0].message.content)
      print("=======================================")

      for message in messages:
          num_tokens += 3 + len(encoding.encode(message["content"]))
      num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>\n",

      num_tokens += len(encoding.encode(completion.choices[0].message.content))

      return {
        "tokens": num_tokens,
        "message": """
## Pros and cons for every idea
{pros_cons}


## Summary

{summary}

""".format(pros_cons=pros_cons, summary=completion.choices[0].message.content)}

      
