
# import necessary packages
import speech_recognition as sr # https://pypi.org/project/SpeechRecognition/
import queue
from flask import Flask, render_template, Response
import time
import threading
import sys
import openai
import config
import numpy as np

prompt_string = f"""
You are an expert in dialogic reading. 
You will generate {config.num_qs} different dialogic questions to prompt conversation with a {config.age} year-old child about a story.
{config.age_guidances}

I additionally want you to keep in mind that the child has {config.read_before} read the story. 

When generating questions, you will generate them across six dimensions:

1. Questions should be either concrete or abstract. 
Concrete questions are focused on explicit information, and an example concrete question is “What color is the dog?” 
Abstract questions are focused on implicit information, and an example abstract question is “Why do you think the dog is sad?”

2. Questions should be book-focused or child-focused. 
Book-focused questions focus directly on the content, themes, or structures within the book itself, including the plot,
characters, settings, and author’s intent. An example book-focused question is “What can you tell me about the story’s setting?” 
Child-focused questions focus on the child’s experiences, feelings, and connections to the story, and an example of a 
child-focused question is “How would you feel if you were in the main character’s situation?”

3. Questions should be either open-ended or closed-ended. 
Open-ended questions do not have a single correct answer and encourage conversation by allowing the child to think 
more deeply or creatively. An example of an open-ended question is, "What do you think will happen next?" 
Closed-ended questions, on the other hand, typically have a specific, correct answer. An example of a closed-ended 
question is, "Is the dog big or small?"

4. Questions should be either contextualized or decontextualized. 
Contextualized questions are linked directly to the text and ask the child to draw upon the story or illustrations for
their answers. For example, "Why do you think the dog ran into the woods?" Decontextualized questions encourage the 
child to use their general knowledge or experience, not tied to the specific story. An example is, "Have you ever seen
a dog like this in real life?"

5. Questions should be either recalled (if this is not the first time the child has read the story) 
or non-recalled  (if this is the first time the child has read the story). 
Recalled questions ask the child to remember information from a previous reading of the same story. For example, 
"Do you still remember the name of the dog in this story since the last time we read it?" Non-recall questions, 
on the other hand, pertain to the current reading and don't require the child to remember details from previous 
readings. An example of a non-recalled question is, "What has the dog just found in the woods?"
If the child has never read the story before, questions should be non-recalled. If the child has read the story before
recall questions are appropriate, and can ask about previous book content, or try to call back to previous readings.

6. Questions should be either predictive or non-predictive. 
Predictive questions invite the child to guess what might happen next in the story, often based on the information 
given so far. For instance, "What do you think the dog will do with what he found in the woods?" Non-predictive 
questions do not ask the child to make guesses about future events of the story. An example would be, "How did the dog
feel when he found something in the woods?"
If this is the first time the child has read the story, predictive questions are appropriate. If the child has already
read the story, questions should be non-predictive.

Every question should be a random combination of these 6 dimensions. 
When giving me the questions, I only want you to give the questions themselves. I do not want anything else.
Do not state what category each of the questions belongs to.

For greater context, this is the book text that has already been read: {config.book_text_prev}. This text should not
be used explicitly to generate any questions other than potential recall questions, but should be used for your 
context and reference to what is happening in the story.

Here is the book text to use to create the {config.num_qs} questions I asked for, keeping in mind that vocabulary 
should be appropriate for the child's age:
"""

# Generate GPT-4 output and update the global variable
def chat_with_chatgpt(transcript, model="gpt-3.5-turbo", stream = True):
    if stream == True:
        chat_response = openai.ChatCompletion.create(model=model,
              messages=[{"role": "user", "content": prompt_string+transcript}],
              stream = True)
        for response in chat_response:
            word = response['choices'][0]['delta'].get('content')
            if word is not None:
                config.gpt_output += word.replace('\n','<br>')
            
            
    else:
        chat_response = openai.ChatCompletion.create(
          model=model,
          messages=[{"role": "user", "content": prompt_string+transcript}]  
        )
        message = chat_response["choices"][0]["message"]["content"]
        
        config.gpt_output = message.replace("\n", "<br>") + "<br>"

    
def generate_loop():
    results = []
    while True:
        result = config.transcript_queue.get()
        results.append(result)

        # Generate GPT-4 output every gen_counter number of phrases sentences
        print(config.sentence_counter)
        print(config.gen_counter)
        if config.sentence_counter >= config.gen_counter:
            config.generating = True
            generated_qs = chat_with_chatgpt(''.join(results[-1]))
            results = []   
            config.generating = False
            config.sentence_counter = 0
            config.gen_counter = np.random.randint(config.genmin,config.genmax)
            
    sys.exit()
