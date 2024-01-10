import os
import openai
import inspect
from nextcord.ext import commands
import nextcord
import datetime
from src.discord.cogs.openAI.components.tasks import wait_for_idx_reaction, wait_for_options_reaction
from src.discord.helpers.embedding.recipe_embed import RecipeEmbedding
from src.discord.bot import DiscordBot
from src.discord.helpers import parser
from src.discord.helpers import json_manager


openai.organization = os.getenv("OPEN_AI_ORG")
openai.api_key = os.getenv("OPEN_AI_TOKEN")


class AICog(commands.Cog):

    def __init__(self, bot):
        self.recipe_embedding = RecipeEmbedding()
        self.bot: DiscordBot = bot
        self.name = "Admin Commands"
        print("AICog connected")


    def generate_response(self, messages:list[dict], token_limit:int) -> str:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Using GPT-3.5-turbo
            messages=messages,
            max_tokens=token_limit  # Maximum length of the output
        )

        return response['choices'][0]['message']['content']

    def determine_difficulty(self, recipe_info):
        steps = len(recipe_info.get('steps', []))
        ingredients = len(recipe_info.get('ingredients', []))
        
        total = steps + ingredients

        if total <= 5:
            return "Beginner"
        elif 5 < total <= 10:
            return "Intermediate"
        elif 10 < total <= 15:
            return "Advanced"
        else:
            return "Expert"




    # =====================================================================================================
    @nextcord.slash_command(dm_permission=False, name="contains", description="type an ingredient or list of ingredients")
    async def contains(self, interaction: nextcord.Interaction, ingredient: str, allergies: str=None) -> None:
        print("INFO", f"{interaction.user.name} used {self}.{inspect.currentframe().f_code.co_name} at {datetime.datetime.now()}")
        """use AI to find recipes that contain certain ingredients"""
        await interaction.response.defer()
        embed_title = f"Recipes that contain {ingredient}."
        embed = nextcord.Embed(title=embed_title)

        user_message = f"Create a list of 5 recipe names that contain {ingredient}, list only the names"
        if allergies:
            embed.description = f"Excluding: {allergies}"
            user_message += f" exclude recipes that contain {allergies}"

        messages = [
                {"role": "system", "content": "You are a helpful sous chef."},
                {"role": "user", "content":  user_message}
            ]

        response = self.generate_response(messages, 150)

        embed.add_field(name="="*len(embed_title), value=response)
        embed.set_footer(text="Click the numbers below to view recipe details")

        message = await interaction.followup.send(embed=embed)

        # Add reactions to the message
        for reaction in self.bot.IDX_REACTIONS:
            await message.add_reaction(reaction)

        # Start a background task to watch for reactions
        self.bot.loop.create_task(wait_for_idx_reaction(self, interaction, response, message))
        

    # =====================================================================================================
    @nextcord.slash_command(dm_permission=False, name="get_recipe", description="use AI to find recipes that contain certain ingredients")
    async def get_recipe(self, interaction: nextcord.Interaction, dish_name: str, serving_count: int = 1) -> None:
        print("INFO", f"{interaction.user.name} used {self}.{inspect.currentframe().f_code.co_name} at {datetime.datetime.now()}")

        try:
            await interaction.response.defer()
        except nextcord.errors.InteractionResponded:
            pass

        if f"{dish_name}.json" in os.listdir("src/recipes"):
            recipe_info = json_manager.open_json(f"src/recipes/{dish_name}.json")
            messages = []
            response = ""
        else:
            messages = [
                {"role": "system", "content": f"You are a helpful sous chef preparing a concise recipe.\n===\nPart 1: List the Ingredients for {serving_count} servings\n- ingredient 1\n- ingredient 2\n===\nPart 2: Write concise Instructions\n1.\n2.\n3.\n===\nPart 3: short Description of dish\nPart 4: carefully consider a spice factor integer between one and ten"},
                {"role": "user", "content": f'Generate a step by step recipe for {dish_name}'}
            ]
            response = self.generate_response(messages, 1000)
            recipe_info = parser.recipe_parser(dish_name, response)
            json_manager.save_json(f"src/recipes/{dish_name}.json", recipe_info)

        difficulty = self.determine_difficulty(recipe_info)

        head_embed, instructions_embed = self.recipe_embedding.create_embeds(dish_name, recipe_info)

        head_embed.description = f"Difficulty: **{difficulty}**"

        await interaction.followup.send(embed=head_embed)
        message = await interaction.followup.send(embed=instructions_embed)

        # Add reactions to the message
        for reaction in self.bot.OPTIONS_REACTIONS:
            await message.add_reaction(reaction)

        # Start a background task to watch for reactions
        self.bot.loop.create_task(wait_for_options_reaction(self.bot, interaction, recipe_info, message))


# =====================================================================================================
    @nextcord.slash_command(dm_permission=False, name="help", description="use AI to find solutions to your culinary problems")
    async def help(self, interaction: nextcord.Interaction, query:str) -> None:
        print("INFO", f"{interaction.user.name} used AICog.contains at {datetime.datetime.now()}")
        """use AI to find solutions to your culinary problems"""
        await interaction.response.defer()

        messages = [
                {"role": "system", "content": "You are a helpful sous chef. Please help me with my culinary problem. do not respond to non culinary questions"},
                {"role": "user", "content":  query + "do not respond to non culinary questions"}
            ]
        response = self.generate_response(messages, 500)

        embed = nextcord.Embed(title=query.capitalize(), description=response, color=nextcord.Color.blurple())

        await interaction.followup.send(embed=embed)



# =====================================================================================================
    @nextcord.slash_command(dm_permission=False, name="cuisine", description="use AI to find recipes of a certain cuisine")
    async def cuisine(self, interaction: nextcord.Interaction, cuisine_name: str, allergies: str=None) -> None:
        print("INFO", f"{interaction.user.name} used AICog.contains at {datetime.datetime.now()}")
        """use AI to find recipes of a certain cuisine"""
        await interaction.response.defer()
        embed_title = f"{cuisine_name} Recipes."
        embed = nextcord.Embed(title=embed_title)

        user_message = f"Create a list of 5 recipe names that are a {cuisine_name} dish"
        if allergies:
            user_message += f" exclude recipes that contain {allergies}"

        messages = [
                {"role": "system", "content": "You are a helpful sous chef."},
                {"role": "user", "content":  user_message}
            ]

        response = self.generate_response(messages, 150)

        embed.add_field(name="="*len(embed_title), value=response)
        embed.set_footer(text="Click the numbers below to view recipe details")

        message = await interaction.followup.send(embed=embed)

        # Add reactions to the message
        for reaction in self.IDX_REACTIONS:
            await message.add_reaction(reaction)

        # Start a background task to watch for reactions
        self.bot.loop.create_task(wait_for_idx_reaction(interaction, messages, response, message))
        
        

# =====================================================================================================
#    @nextcord.slash_command(dm_permission=False, name="get_nutrition", description="use AI to find nutrition facts for dish")
#    async def get_nutrition(self, interaction: nextcord.Interaction, dish_name: str, serving_count: int=2) -> None:
#        print("INFO", f"{interaction.user.name} used AICog.contains at {datetime.datetime.now()}")
#        """use AI to find nutrition facts for dish"""
#        await interaction.response.defer()
        # messages = [
        #         {"role": "system", "content": f"You are a helpful sous chef preparing a concise recipe.\n===\nPart 1: List the Ingredients for {serving_count} servings\n- ingredient 1\n- ingredient 2\n===\nPart 2: Give nutritional facts for calories, fats, carbohydrates, and protein\n1.\n2.\n3.\n===\nPart 3: short Description of dish\n"},
        #         {"role": "user", "content": f'Generate nutritional information for {dish_name}'}
        #     ]
        # response = self.generate_response(messages, 500)

        # r_embed, i_embed = self.recipe_embedding.create_embeds(title=f"Recipe for {dish_name}", message=response)

        # await interaction.followup.send(embed=r_embed)
#        await interaction.send("Coming soon!")



def setup(bot: commands.Bot):
    bot.add_cog(AICog(bot))

