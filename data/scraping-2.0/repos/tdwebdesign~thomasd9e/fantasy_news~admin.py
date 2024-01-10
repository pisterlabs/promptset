from django.contrib import admin
from django.utils.text import slugify
from .models import GameRecap
from .services import FantasyLeague
from blog.models import Post
from accounts.models import CustomUser

import markdown
import os
import openai
import spacy

# Constants and initializations
league = FantasyLeague("985036837548855296")
nlp = spacy.load("en_core_web_sm")
openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_prompt_template(owned_players, recap_body):
    return f"""
    Provide a summary with a touch of humor of the following game recap, structured with the title 
    'Fantasy Football Highlights: <game score>'. <game score> is the score of the game found in the recap. 
    Example: "Giants 45 - Panthers 26". Follow this with three distinct sections titled 'Game Summary', 
    'McAlister's Deli Quick Bites', and 'Free Agent Spotlight'. In the 'McAlister's Deli Quick Bites' section, 
    give quick takes on these owned fantasy players: {', '.join(owned_players)}. Highlight a potential free 
    agent from the recap that had a notable performance in the 'Free Agent Spotlight' section. Ensure each 
    section is clearly separated and formatted with bold headings. Do not include any messages after the output.
    ###
    {recap_body}
    ###
    """


def extract_owned_players(game_recap_body):
    doc = nlp(game_recap_body)
    player_names = {entity.text for entity in doc.ents if entity.label_ == "PERSON"}
    return [
        f"{player} - {league.get_owner_for_player(player)}"
        for player in player_names
        if league.get_owner_for_player(player)
    ]


def create_post_from_response(response):
    message = response["choices"][0]["message"]["content"]
    lines = message.split("\n")
    title = lines[0]
    slug = slugify(title)
    message = "\n".join(lines[1:])
    html_message = markdown.markdown(message)
    user = CustomUser.objects.get(username="AI Writer")
    post = Post(
        title=title, slug=slug, content=html_message, author=user, post_status="draft"
    )
    post.save()


def process_game_recap(game_recap):
    owned_players = extract_owned_players(game_recap.body)
    print(owned_players)
    prompt = generate_prompt_template(owned_players, game_recap.body)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    create_post_from_response(response)


def process_selected_recaps(modeladmin, request, queryset):
    for recap in queryset:
        process_game_recap(recap)
        recap.processed = True
        recap.save()


@admin.register(GameRecap)
class GameRecapAdmin(admin.ModelAdmin):
    list_display = ["title", "date", "processed"]
    actions = [process_selected_recaps]
