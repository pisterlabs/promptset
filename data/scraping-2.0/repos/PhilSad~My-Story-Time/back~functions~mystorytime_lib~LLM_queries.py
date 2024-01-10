import re
import openai

from .prompt_templates import prompt_template_translate, prompt_template_describe, prompt_template_split, prompt_template_writestory


# Convert a list of panel descriptions with fields to dict
def to_description_dict(chatgpt_reponse):
    pattern = r"Panel [^:]+:(.*?)(?=\nPanel|$)"
    descriptions = re.findall(pattern, chatgpt_reponse, re.DOTALL)

    splitted_descriptions = []
    for description in descriptions:
        pattern = r"\n- ([^:]+): (.*?)(?=\n-|$)"
        matches = re.findall(pattern, description, re.DOTALL)

        splitted_descriptions.append(
            {
                key: value for key, value in matches 
            }
        )

    return splitted_descriptions


# Split chapter into sections using ChatGPT
def LLM_split_chapter(chapter, n_panels):
    prompt = prompt_template_split(chapter, n_panels)

    response_caption = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user", "content": prompt}]
    ).choices[0].message.content

    pattern = r"Panel [^:]+:(.*?)(?=\nPanel|$)"
    captions = re.findall(pattern, response_caption, re.DOTALL)
    captions = [caption.strip() for caption in captions]

    return captions


# Describe every caption passed to the function while keeping the context
def LLM_describe_chapter(captions):
    response_caption = "\n\n".join([f"Panel {i}: {text}" for i, text in enumerate(captions)])

    image_desc_prompt = prompt_template_describe(response_caption)

    response_description = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user", "content": f"{response_caption}\n\n{image_desc_prompt}"}]
    ).choices[0].message.content

    description_dict = to_description_dict(response_description)

    description_prompts = [
        " ".join([v.strip() for v in desc_dict.values()])
        for desc_dict in description_dict
    ]

    return description_prompts


# returns a paragraph dictionnary with text and description for a given chapter
def LLM_split_and_describe(chapter, n_panels=3):
    """Split a chapter into its parts and describe every part for image generation

    Keyword arguments:  
    - chapter: content of the chapter   
    - n_panels: number of panels to create (default 4)  

    Returns:  
    A list of dictionnary containing text and image_desc for each panel  
    """
    captions = LLM_split_chapter(chapter, n_panels)
    description_prompts = LLM_describe_chapter(captions)

    paragraphs = [{
        "text": text,
        "image_desc": description
    } for text, description in zip(captions, description_prompts)]

    return paragraphs


# write the story from the idea and the hero name using chatgpt
def LLM_write_story(story_idea, hero_name):
    prompt = prompt_template_writestory(story_idea, hero_name)
    response = openai.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role":"user", "content": prompt}]
    )
    
    full_text = response.choices[0].message.content

    return full_text


# translate a given text using chatgpt
def LLM_translate_text(text, language):
    print('called LLM_translate_text')
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user", "content": prompt_template_translate(text, language)}],
        max_tokens=100
    )
    translation = response.choices[0].message.content
    return translation
