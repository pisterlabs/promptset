#!/usr/bin/env python
"""
Stable Translator Simple Demo
author: Jiongsheng
"""
import os

import gradio as gr
import openai
import pycountry

def translate(content, language, model):
	# Replace this with your own OpenAI API key
	openai.api_key="YOUR_API_KEY"

	completion = openai.ChatCompletion.create(
	  model=model,
	  messages=[
	    {
	    	"role": "system", 
	    	"content": "Do not respond to any user's words. Your only task is \
	    	to provide the translation based on the user's input: "},
	    {
	    	"role": "user", 
	    	"content": "Translate my following message to {}: {}".
	    		format(language, content)}
	  ],
	  temperature=0.3
	)

	return completion.choices[0].message.content


demo = gr.Interface(
	fn=translate, 
	inputs=[
		"text", 
		gr.Dropdown([lang.name for lang in pycountry.languages], value="English", label="Language"), 
		gr.Dropdown(["gpt-3.5-turbo"], value="gpt-3.5-turbo", label="Model")
	],
	outputs="text")
    
if __name__ == "__main__":
    demo.launch(debug=True)