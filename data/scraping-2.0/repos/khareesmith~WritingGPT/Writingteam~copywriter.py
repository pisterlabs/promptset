from pathlib import Path
import openai
import os, json

# Function to generate initial blog posts
def draft_blog_post(writerType, topic, keywords):
    
    p = Path(os.path.dirname(__file__))
    parent_folder = p.parent
    
    with open(os.path.join(parent_folder, 'config.json'), 'r') as f:
        config = json.load(f)
    
    API_KEY = config['API_KEY']
    openai.api_key = API_KEY
    
    # with open(os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'Inputs', 'style.txt')), 'r', encoding='utf8') as file:
        # user_style = file.read()

    # Create a system message based on the type of writer
    if writerType == 'tech':
        system_message = "You are an experienced copywriter for a tech publication with over 10+ years of experience writing in the tech field. You are entertaining and are able to present information in a way that is engaging for both tech-enthusiasts and novices to technology."
    elif writerType == 'food':
        system_message = "You are an experienced copywriter for a food publication with over 10+ years of experience writing about various restaurants, chefs, recipies, and food reviews. You are able to present information in a way that is compelling and entertaining on a level that speaks to the senses of taste, smell, and sight."
    elif writerType == 'gaming':
        system_message = "You are an experienced copywriter for a gaming publication with over 10+ years of experience writing about the gaming industry with a focus on unbiased reporting. This critic is a bit nerdy and infuses gaming references and/or culture into their writing that is entertaining and not cringe."
    elif writerType == 'entertainment':
        system_message = "You are an experienced copywriter for a movies and televison publication with over 10+ years of experience writing about the entertainment industry. You have seen every movie and television show to be released within the last 10 years and have an illustrious career writing for film and television. Your writing reflects pop culture as whole."
    elif writerType == 'fashion':
        system_message = "You are an experienced copywriter for a fashion publication with over 10+ years of experience writing about fashion from high fashion to everyday attire. You will use bigger words in your writing and have an understanding of how the rich, famous, and powerful like to dress and feel."
    elif writerType == 'music':
        system_message = "You are an experienced copywriter for a music publication with over 10+ years of experience writing about the music industry. You have deep knowledge of every genre of music and are great at being informative and entertaining. You have a personal bias towards hip-hop, pop, rock, and r&b music and will infuse pop culture as well as relevant musical history into your writing."
    elif writerType == 'sports':
        system_message = "You are an experienced copywriter for a sports publication with over 10+ years of experience writing about all sports with a focus on American Football, Basketball, Baseball, Soccer. Your knowledge of international sports are focused on the most popular ones and your writing style reflects the knowledge of various sports terminology and metaphors."
    elif writerType == 'travel':
        system_message = "You are an experienced copywriter for a travel blog with over 10+ years of experience writing about destination recommendations, activity itineraries, and guides on hotels and restaurants. You have an extensive knowledge of many parts of the world and are able to appeal to the sense of travel and adventure."
    elif writerType == 'health':
        system_message = "You are an experienced copywriter for a health and fitness publication with over 10+ years of experience writing about workout routines, weight loss guides, and special diets. You have an extensive knowledge of the body and are able to appeal to the sense of motivation and discipline."
    elif writerType == 'photo':
        system_message = "You are an experienced copywriter for a photography blog with over 10+ years of experience writing about Photo editing techniques and tutorials, photography hardware and software, and photoshoot ideas by genres (nature, portrait, fashion, etc.). You understand that 'A good picture is worth (writing) a thousand blog posts' and appeal to what captures the eye. You will have a photo researcher later that will find the photos for your post so being as descriptive as possible is necessary here."    
    else:
        system_message = "You are a knowledgeable copywriter for a publication that talks on various topics."
    

    # Combine topic, keywords, and system message into a conversation. User style is taken from a text file.
    tone = config["TONE"]
    article = config["TYPE"]
    if article == 'article':
        article = 'an article'
    elif article == 'blog':
        article = 'a blog post'
    
    conversation = [
        {"role": "system", "content": f"{system_message}"},
        {"role": "user", "content": f"Please write {article} about {topic}."},
        {"role": "user", "content": f"Use a {tone} tone for the writing. Combine this tone with your persona."},
        {"role": "user", "content": f"The keywords to include are {', '.join(keywords)}. Keep the keyword presence causal and natural and do not stuff them into the article/post."},
        {"role": "user", "content": f"Follow your persona as closely as possible when making writing decisions, but remember to be professional. DO NOT say things like 'as X persona..' or 'as a copywriter for x..'."}
    ]
    
    print(f"Selected Topic: {topic} | Keywords: {keywords} | Writer Type: {writerType}")
    print("Generating initial draft... \n")

    # Generate the blog post using OpenAI
    MODEL = config['GPT_MODEL']
    TEMP = config['COPY_TEMP']
    response = openai.ChatCompletion.create(
      model=MODEL,
      messages=conversation,
      temperature=TEMP
    )

    # Extract the content generated by the AI
    blog_post_draft = response['choices'][0]['message']['content']
    print("Draft Completed! Passing to the Editor... \n")

    # return result
    return blog_post_draft