from langchain.prompts import PromptTemplate, ChatPromptTemplate

def blog_generator(word_count:int=1000, target_audience:str='all'):

    prompt_text = f"""
Create a well-structured blog post from the provided Context. 

The blog post should be around {word_count} words long and should effectively capture the key points, insights, and information from the Context.

Focus on maintaining a coherent flow and using proper grammar and language. 

Incorporate relevant headings, subheadings, and bullet points to organize the content. 

Ensure that the tone of the blog post is engaging and informative, catering to {target_audience} audience. 

Feel free to enhance the transcript by adding additional context, examples, and explanations where necessary. 

The goal is to convert context into a polished and valuable written resource while maintaining accuracy and coherence.

CONTEXT: {{text}}

BLOG POST: """

    prompt = ChatPromptTemplate.from_template(template=prompt_text)

    return prompt

def tweet_generator(word_count:int=258, target_audience:str='all', thread:bool=False):

    prompt_text = f"""
Transform the given Context into a compelling Twitter {'thread' if thread else 'post'}. 

Condense the content from the context into a {'series of well-crafted tweets' if thread else 'well-crafted tweet'}  that capture the main ideas and highlights. 

Craft each tweet to be concise, staying within the character limit of {word_count}, while effectively conveying the essence of the context for {target_audience} audience. 

DO NOT Use hashtags and Emojis

Use mentions to enhance the visibility and engagement of the thread. 

Feel free to rephrase sentences and use bullet points to simplify complex information. 

{"Maintain a coherent narrative across the tweets to ensure that the thread tells a complete story or provides valuable insights." if thread else ""}

The aim is to create an engaging and informative Twitter thread that encourages interaction and sharing among the target audience.

CONTEXT: {{text}}

{"TWITTER THREAD" if thread else "TWEET"}: """

    prompt = ChatPromptTemplate.from_template(template=prompt_text)

    return prompt

def newsletter_generator(word_count:int=1000, target_audience:str='all'):

    prompt_text = f"""
Transform the following Context into a compelling and informative email newsletter with maximum {word_count} words for {target_audience} audience. 

Craft a well-structured newsletter that captures the essence of the context while providing valuable insights to the subscribers. 

Summarize the key points, ideas, and takeaways from the video in a clear and concise manner. Add contextual explanations and examples where needed to enhance understanding. 

Organize the content into sections with relevant headings, and consider incorporating visual elements such as images or infographics to support the text. 

Craft a captivating subject line that entices recipients to open the email. 

Ensure that the tone of the newsletter aligns with the audience's preferences and that the content flows smoothly. 

The goal is to create an engaging email newsletter that offers value to the subscribers and encourages them to take action or engage further with the content.

CONTEXT: {{text}}

NEWSLETTER: """

    prompt = ChatPromptTemplate.from_template(template=prompt_text)

    return prompt

def summary_generator(word_count:int=1000, target_audience:str='all'):

    prompt_text = f"""
Generate a concise and coherent summary from the given Context. 

Condense the context into a well-written summary that captures the main ideas, key points, and insights presented in the context. 

Prioritize clarity and brevity while retaining the essential information. 

Aim to convey the context's core message and any supporting details that contribute to a comprehensive understanding. 

Craft the summary to be self-contained, ensuring that readers can grasp the content even if they haven't read the context. 

Provide context where necessary and avoid excessive technical jargon or verbosity.

The goal is to create a summary that effectively communicates the context's content while being easily digestible and engaging."

Keep the summary under {word_count} words for {target_audience} audience.

CONTEXT: {{text}}

SUMMARY: """

    prompt = ChatPromptTemplate.from_template(template=prompt_text)

    return prompt


def unit_tests_generator():

    prompt_text = f"""
Create comprehensive unit tests for the provided Python code block to verify that it functions correctly under various scenarios. 

Ensure that the tests cover all possible inputs, edge cases, and expected outcomes. 

Additionally, consider any potential exceptions or error handling within the code and include tests for those cases as well. 

Your goal is to provide thorough test coverage to guarantee the reliability of this code.

Only answer me with the code and nothing else.

Here's an example of a answer,

Example answer: 
import unittest
from sample_module import Calculator, square, greet...

Code: ``` {{text}} ```

ANSWER: 
"""

    prompt = ChatPromptTemplate.from_template(template=prompt_text)

    return prompt