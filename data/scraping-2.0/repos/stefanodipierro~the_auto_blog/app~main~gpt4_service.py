# gpt4_service.py
import openai
from app import db
from app.main.models import Post
from flask import session, jsonify, current_app, flash
from .sender import Sender
from .receiver import Receiver
import os




def generate_titles(num_articles, topic):
    openai.api_key = current_app.config['OPENAI_API_KEY']

    prompt = f"Generate {num_articles} number of titles for blog posts about {topic}"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use the GPT-3.5 model
        messages=[
            {"role": "system", "content": "You are an advanced AI assistant specialized in generating creative and unique blog post titles. Consider the topic at hand and think about engaging, relevant titles that would attract readers. Try to vary the structure and style of each title for diversity."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000
    )

    # The generated titles are in the 'choices' list in the response. We split them by newline character.
    titles = response['choices'][0]['message']['content'].split('\n')
    # Remove bullet points
    titles = [title.split('. ', 1)[-1] for title in titles if title.strip() != '']
    titles = [title.strip('"') for title in titles]
    if num_articles == 1:
        titles = [titles[0]]
    return titles

def create_post(title, description, image_path_list, images_prompt):
    post = Post()
    post.from_dict({'title': title, 'description': description, 'images': image_path_list, 'images_prompt': images_prompt})
    db.session.add(post)
    db.session.commit()
    response = jsonify({"message": "Post created successfully", "id": post.id})
    response.status_code = 201
    return response

def generate_article(title):
    openai.api_key = current_app.config['OPENAI_API_KEY']

    prompt = f"Write an article about {title}. The article should be at least 2000 words long."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",  # Use the GPT-3.5 model
        messages=[
            {"role": "system", "content": "You are an intelligent AI assistant with expertise in generating informative, engaging, and well-structured blog articles. Each article should provide value to the reader, be coherent and well-organized, and use a style and tone appropriate for a blog audience. Remember to include a strong introduction and conclusion."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=10000
    )

    # The generated article is in the 'choices' list in the response
    article = response['choices'][0]['message']['content']
    return article

def wrap_paragraphs(article):
    paragraphs = article.split("\n")  # Split the text into paragraphs at newline characters
    wrapped_paragraphs = [f"<p>{p.strip()}</p>" for p in paragraphs if p.strip()]  # Wrap each paragraph in <p> tags
    return "\n".join(wrapped_paragraphs)  # Join the paragraphs back together, separated by newlines

  

def generate_images(title):
    openai.api_key = current_app.config['OPENAI_API_KEY']

    prompt = f"Generate image description for an article titled '{title}'"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",  # Use the GPT-3.5 model
        messages=[
            {"role": "system", "content": "You are an advanced AI assistant, specializing in creating detailed and comprehensive descriptions of images. Your task is to capture all the significant elements in the image, from the colors and shapes to the emotions and actions depicted. Be as specific and descriptive as possible, conveying the overall mood and atmosphere of the image. Remember to consider the potential cultural, historical, or symbolic significance of elements in the image. Your descriptions should help someone who cannot see the image understand its content and feel its impact."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50
        
    )

    # The generated image descriptions are in the 'choices' list in the response.
    image_descriptions = [choice['message']['content'] for choice in response['choices']]
    
    # Return only the first description
    print('Image description' + image_descriptions[0])
    return image_descriptions[0] if image_descriptions else None




def generate_and_save_articles(num_articles, topic):
    
    

    prompt = f"Create {num_articles} titles for articles of a blog on the topic {topic}"
    # Qui invii il prompt a GPT-3.5 e ottieni una lista di titoli
    titles = generate_titles(num_articles , topic)

    for title in titles:
        string_description = generate_article(title)
        description = wrap_paragraphs(string_description)

        images_prompt = generate_images(title)
        print('image prompt generated' + images_prompt )
        sender = Sender()
        sender.send(prompt=images_prompt)
        print('sent to mid api')
        receiver = Receiver(directory='app/static')
        try:
            url, filename = receiver.collecting_result(image_prompt= images_prompt)
            images_path_list = receiver.download_image(url, filename)
            print('images downloaded')
            print(images_path_list)

            create_post(title, description, images_path_list, images_prompt)
            print('post created')
        except Exception as e:
            flash(f"Error: {str(e)}")
            # You might want to break the loop here, or continue with the next iteration.
            # It depends on how you want your application to behave in case of error.
    return "Articles generated!"