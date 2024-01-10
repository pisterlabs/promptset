import nextcord

from nextcord.ext import commands
import openai
from difflib import SequenceMatcher
import asyncio
openai_model_engine = "text-davinci-003"

class LanguageQuizCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.quiz_sessions = {}

    @nextcord.slash_command()
    async def quiz1(self, interaction: nextcord.Interaction):
        pass

    @quiz1.subcommand()
    async def language_quiz(self,interaction: nextcord.Interaction, target_language: str):
        await interaction.response.send_message("Type 'quit' at any time to end the quiz.")

        while True:
            # Request a quiz question from GPT-3.5-turbo
            prompt = f"Create a language learning quiz question for {target_language}:"
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates language learning quiz questions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                n=1,
                stop=None,
                temperature=0.8,
            )
            question = completion.choices[0].message["content"].strip()

            # Request the answer to the quiz question
            prompt = f"What is the answer to the following language learning quiz question: '{question}'? Provide a simple and clear answer."
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides answers to language learning quiz questions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                n=1,
                stop=None,
                temperature=1,
            )
            answer = completion.choices[0].message["content"].strip()

            # Send the question to the user
            await interaction.channel.send(question)

            # Check the user's answer and provide feedback
            def check(m):
                return m.author.id == interaction.user.id and m.channel.id == interaction.channel.id

            try:
                user_answer = await self.bot.wait_for("message", check=check, timeout=60)
                if user_answer.content.lower() == "quit":
                    await interaction.channel.send("Quiz ended. Goodbye!")
                    break

                similarity = SequenceMatcher(None, user_answer.content.lower(), answer.lower()).ratio()
                if similarity > 0.10:  # Set the similarity threshold (0.9 means 90% similarity)
                    await interaction.channel.send("Correct! Good job! 🎉")
                else:
                    await interaction.channel.send(f"Almost! The correct answer is:\n\n{answer}")
            except asyncio.TimeoutError:
                await interaction.channel.send("Time's up! Please try again.")
                
                
                
    @quiz1.subcommand()
    async def multiple_choice_quiz(self,interaction: nextcord.Interaction, target_language: str):
        await interaction.response.send_message("Type 'quit' at any time to end the quiz.")

        while True:
            # Request a multiple-choice quiz question from GPT-3.5-turbo
            prompt = f"Create a multiple-choice language learning quiz question for {target_language}:"
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates multiple-choice language learning quiz questions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                n=1,
                stop=None,
                temperature=0.8,
            )
            question = completion.choices[0].message["content"].strip()

            # Request the answer to the quiz question
            prompt = f"What is the correct answer to the following multiple-choice language learning quiz question: '{question}'? Provide the letter of the correct answer."
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides answers to multiple-choice language learning quiz questions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                n=1,
                stop=None,
                temperature=1,
            )
            answer = completion.choices[0].message["content"].strip().lower()

            # Send the question to the user
            await interaction.channel.send(question)

            # Check the user's answer and provide feedback
            def check(m):
                return m.author.id == interaction.user.id and m.channel.id == interaction.channel.id

            try:
                user_answer = await self.bot.wait_for("message", check=check, timeout=60)
                if user_answer.content.lower() == "quit":
                    await interaction.channel.send("Quiz ended. Goodbye!")
                    break

                if user_answer.content.lower() == answer.lower():
                    await interaction.channel.send("Correct! Good job! 🎉")
                else:
                    await interaction.channel.send(f"Almost! The correct answer is: {answer.upper()}")
            except asyncio.TimeoutError:
                await interaction.channel.send("Time's up! Please try again.")          
                
                
                
    @quiz1.subcommand()
    async def fill_in_the_blank(self,interaction: nextcord.Interaction, target_language: str):
        # Generate a sentence with a blank in the target language
        prompt = f"Generate a sentence with a blank in {target_language}:"
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates fill-in-the-blank sentences in different languages."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.8,
        )
        sentence = completion.choices[0].message["content"].strip()

        # Get the answer to the blank
        prompt = f"What is the missing word in the following sentence: '{sentence}'?"
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides answers to fill-in-the-blank sentences in different languages."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            n=1,
            stop=None,
            temperature=0.8,
        )
        answer = completion.choices[0].message["content"].strip()

        # Send the sentence with a blank to the user
        await interaction.channel.send(f"💬 **Fill in the Blank** 💬\n\nComplete the following {target_language} sentence:\n\n**{sentence}**")

        # Check the user's answer and provide feedback
        def check(m):
            return m.author.id == interaction.user.id and m.channel.id == interaction.channel.id

        try:
            user_answer = await self.bot.wait_for("message", check=check, timeout=100)
            similarity = SequenceMatcher(None, user_answer.content.lower(), answer.lower()).ratio()
            if similarity > 0.75:  # Set the similarity threshold (0.75 means 75% similarity)
                await interaction.channel.send("🎉 Correct! Good job! 🎉")
            else:
                await interaction.channel.send(f"Almost! The correct missing word is: **{answer}**")
        except asyncio.TimeoutError:
            await interaction.channel.send("⌛ Time's up! Please try again.")
                
                
                
    @quiz1.subcommand()
    async def vocabulary_practice(self,interaction: nextcord.Interaction, target_language: str):
        await interaction.response.send_message("Type 'quit' at any time to end the practice.")

        while True:
            # Request a vocabulary question from GPT-3.5-turbo
            prompt = f"Create a vocabulary practice question for {target_language}:"
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates vocabulary practice questions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                n=1,
                stop=None,
                temperature=0.8,
            )
            question = completion.choices[0].message["content"].strip()

            # Request the answer to the vocabulary question
            prompt = f"What is the correct answer to the following vocabulary practice question: '{question}'?"
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides answers to vocabulary practice questions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                n=1,
                stop=None,
                temperature=1,
            )
            answer = completion.choices[0].message["content"].strip()

            # Send the question to the user
            await interaction.channel.send(question)

            # Check the user's answer and provide feedback
            def check(m):
                return m.author.id == interaction.user.id and m.channel.id == interaction.channel.id

            try:
                user_answer = await self.bot.wait_for("message", check=check, timeout=60)
                if user_answer.content.lower() == "quit":
                    await interaction.channel.send("Practice ended. Goodbye!")
                    break

                if user_answer.content.lower() == answer.lower():
                    await interaction.channel.send("Correct! Good job! 🎉")
                else:
                    await interaction.channel.send(f"Almost! The correct answer is: {answer}")
            except asyncio.TimeoutError:
                await interaction.channel.send("Time's up! Please try again.")
                
                
                
                
    @quiz1.subcommand()
    async def translate(self,interaction: nextcord.Interaction, source_language: str, target_language: str, text: str):
        """
        Translates text from one language to another using OpenAI.
        Usage: /translate2 <source_language> <target_language> <text to translate>
        """
        # Call the OpenAI API to translate the text
        completions = openai.Completion.create(
            engine=openai_model_engine,
            prompt=f"Translate from {source_language} to {target_language}: {text}",
            max_tokens=64,
            n=1,
            stop=None,
            temperature=0.5,
        )
        translated_text = completions.choices[0].text.strip()

        # Send the translated text back to the user
        await interaction.response.send_message(f"{translated_text}") 
        
        
        
        
    @quiz1.subcommand(description="Learn a new language with the Language Learning Bot! 🌍")
    async def learning(self, interaction: nextcord.Interaction):
        languages = "- Spanish\n- French\n- German\n- Chinese\n- Arabic\n- Portuguese\n- Russian\n- Italian\n- Japanese\n- Korean\n- Dutch\n- Polish\n- Turkish\n- Swedish\n- Norwegian\n- Danish\n- Finnish\n- Greek\n- Czech\n- Hungarian\n- Romanian\n- Bulgarian\n- Slovak\n- Slovenian\n- Lithuanian\n- Latvian\n- Estonian\n- Croatian\n- Serbian\n- Albanian\n- Macedonian"
        embed = nextcord.Embed(title="Welcome to the Language Learning Bot! 🌍", description="Let's start by choosing a language to learn:")
        embed.add_field(name="Languages", value=languages, inline=False)
        await interaction.response.send_message(embed=embed)  
    @quiz1.subcommand()
    async def word_scramble(self,interaction: nextcord.Interaction, target_language: str):
        await interaction.response.send_message("Type 'quit' at any time to end the game.")

        while True:
            # Request a scrambled word from GPT-3.5-turbo
            prompt = f"Create a scrambled word for a word scramble game in {target_language}:"
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates word scramble challenges."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                n=1,
                stop=None,
                temperature=0.8,
            )
            scrambled_word = completion.choices[0].message["content"].strip()

            # Request the unscrambled word
            prompt = f"What is the unscrambled version of the following scrambled word: '{scrambled_word}'?"
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that unscrambles words."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                n=1,
                stop=None,
                temperature=1,
            )
            unscrambled_word = completion.choices[0].message["content"].strip()

            # Send the scrambled word to the user
            await interaction.channel.send(f"Unscramble this word: {scrambled_word}")

            # Check the user's answer and provide feedback
            def check(m):
                return m.author.id == interaction.user.id and m.channel.id == interaction.channel.id

            try:
                user_answer = await self.bot.wait_for("message", check=check, timeout=60)
                if user_answer.content.lower() == "quit":
                    await interaction.channel.send("Game ended. Goodbye!")
                    break

                if user_answer.content.lower() == unscrambled_word.lower():
                    await interaction.channel.send("Correct! Great job! 🎉")
                else:
                    await interaction.channel.send(f"Oops! The correct answer is:\n\n{unscrambled_word}")
            except asyncio.TimeoutError:
                await interaction.channel.send("Time's up! Please try again.")
    
    @quiz1.subcommand()
    async def conversation_partner(self,interaction: nextcord.Interaction, scenario: str, language: str):
        scenarios = {
            "coffee": "You are a coffee enthusiast discussing the various aspects of coffee.",
        "dinner": "You are discussing what to eat for dinner tonight.",
        "sports": "You are talking about your favorite sports and teams.",
        "travel": "You are sharing your travel experiences and future plans.",
        "movies": "You are discussing your favorite movies and genres.",
        "music": "You are talking about your favorite music artists and genres.",
        "books": "You are discussing your favorite books and authors.",
        "gaming": "You are talking about video games and gaming platforms.",
        "technology": "You are discussing the latest technology trends and gadgets.",
        "history": "You are talking about historical events and figures.",
        "science": "You are discussing scientific discoveries and theories.",
        "art": "You are talking about different art styles and famous artists.",
        "fitness": "You are discussing fitness routines and exercises.",
        "cooking": "You are talking about cooking techniques and recipes.",
        "pets": "You are discussing pet care and your favorite pets.",
        "gardening": "You are talking about gardening tips and plants.",
        "photography": "You are discussing photography techniques and equipment.",
        "fashion": "You are talking about fashion trends and styles.",
        "cars": "You are discussing cars and automotive technology.",
        "space": "You are talking about space exploration and astronomy.",
        "politics": "You are discussing current political events and issues.",
        "environment": "You are talking about environmental conservation and climate change.",
        "education": "You are discussing the education system and learning techniques.",
        "parenting": "You are talking about parenting tips and experiences.",
        "shopping": "You are discussing shopping habits and favorite stores.",
        "finances": "You are talking about personal finances and budgeting.",
        "career": "You are discussing career choices and professional development.",
        "hobbies": "You are talking about your favorite hobbies and pastimes.",
        "relationships": "You are discussing relationship advice and experiences.",
        "health": "You are talking about health and wellness tips.",
        "home": "You are discussing home improvement and interior design.",
        "languages": "You are talking about learning new languages and language barriers.",
        "tv_shows": "You are discussing your favorite TV shows and series.",
        "celebrities": "You are talking about celebrities and their lives.",
        "news": "You are discussing recent news events and their impact.",
        "philosophy": "You are talking about philosophical theories and ideas.",
        "economy": "You are discussing the global economy and financial markets.",
        "wildlife": "You are talking about wildlife and animal behavior.",
        "outdoors": "You are discussing outdoor activities and adventures.",
        "beauty": "You are talking about beauty tips and skincare.",
        "social_media": "You are discussing the impact of social media on society.",
        "mythology": "You are talking about mythology and legends.",
        "psychology": "You are discussing psychological theories and human behavior.",
        "architecture": "You are talking about famous buildings and architectural styles.",
        "human_rights": "You are discussing human rights and social justice issues.",
        "religion": "You are talking about different religions and beliefs.",
        "volunteering": "You are discussing volunteering opportunities and experiences.",
        "startups": "You are talking about startups and entrepreneurship.",
        "investing": "You are discussing investing strategies and stock market.",
        "festivals": "You are talking about cultural festivals and celebrations.",
        "food": "You are discussing different types of cuisine and food preferences.",
    "writing": "You are talking about writing techniques and storytelling.",
    "poetry": "You are discussing various forms of poetry and famous poets.",
    "public_speaking": "You are talking about public speaking tips and overcoming stage fright.",
    "magic": "You are discussing magic tricks and illusions.",
    "comedy": "You are talking about comedy styles and favorite comedians.",
    "theater": "You are discussing theatrical plays and performances.",
    "dance": "You are talking about dance styles and famous dancers.",
    "painting": "You are discussing painting techniques and famous painters.",
    "sculpture": "You are talking about sculpture techniques and famous sculptors.",
    "pottery": "You are discussing pottery and ceramic art.",
    "origami": "You are talking about origami and paper folding techniques.",
    "knitting": "You are discussing knitting and crochet patterns.",
    "sewing": "You are talking about sewing techniques and fashion design.",
    "woodworking": "You are discussing woodworking techniques and projects.",
    "meditation": "You are talking about meditation and mindfulness practices.",
    "yoga": "You are discussing yoga poses and benefits.",
    "astronomy": "You are talking about astronomy and celestial events.",
    "geography": "You are discussing geographic locations and landmarks.",
    "anthropology": "You are talking about anthropology and human cultures.",
    "archaeology": "You are discussing archaeological discoveries and ancient civilizations.",
    "sociology": "You are talking about social issues and human behavior.",
    "geology": "You are discussing geology and the Earth's formation.",
    "meteorology": "You are talking about weather patterns and climate change.",
    "oceanography": "You are discussing ocean currents and marine life.",
    "cryptography": "You are talking about cryptography and code breaking.",
    "robotics": "You are discussing robotics and artificial intelligence.",
    "aviation": "You are talking about aviation and aircraft technology.",
    "marine_biology": "You are discussing marine biology and ocean ecosystems.",
    "paleontology": "You are talking about paleontology and prehistoric life.",
    "quantum_physics": "You are discussing quantum physics and the nature of reality.",
    "genetics": "You are talking about genetics and the study of heredity.",
    "nanotechnology": "You are discussing nanotechnology and its applications.",
    "particle_physics": "You are talking about particle physics and the fundamental forces of nature.",
    "chemistry": "You are discussing chemical reactions and the properties of matter.",
    "alternative_energy": "You are talking about alternative energy sources and sustainability.",
    "forensics": "You are discussing forensic science and crime scene investigation.",
    "virtual_reality": "You are talking about virtual reality and its impact on society.",
    "animation": "You are discussing animation techniques and styles.",
    "graphic_design": "You are talking about graphic design principles and software.",
    "web_development": "You are discussing web development and programming languages.",
    "cyber_security": "You are talking about cyber security and protecting online information.",
    "networking": "You are discussing computer networking and internet infrastructure.",
    "software_engineering": "You are talking about software engineering and development methodologies.",
    "hardware_engineering": "You are discussing hardware engineering and computer components.",
    "data_science": "You are talking about data science and analytics.",
    "machine_learning": "You are discussing machine learning algorithms and applications.",
    "bioinformatics": "You are talking about bioinformatics and the analysis of biological data.",
    "neuroscience": "You are discussing neuroscience and the study of the nervous system.",
    "immunology": "You are talking about immunology and the immune system.",
    "pharmacology": "You are discussing pharmacology and drug development.",
    "toxicology": "You are talking about toxicology and the study of harmful substances.",
    "epidemiology": "You are discussing epidemiology and the study of diseases.",
    "nutrition": "You are talking about nutrition and healthy eating habits.",
    "microbiology": "You are discussing microbiology and the study of microorganisms.",
    "zoology": "You are talking about zoology and the study of animals.",
    "botany": "You are discussing botany and the study of plants.",
    "ecology": "You are talking about ecology and the study of ecosystems.",
    "conservation": "You are discussing wildlife conservation and habitat preservation.",
    "sustainability": "You are talking about sustainable living practices and reducing waste.",
    "ethics": "You are discussing ethical dilemmas and moral principles.",
    "law": "You are talking about legal issues and the justice system.",
    "criminology": "You are discussing criminology and criminal behavior.",
    "human_resources": "You are talking about human resources and employee management.",
    "marketing": "You are discussing marketing strategies and consumer behavior.",
    "sales": "You are talking about sales techniques and building customer relationships.",
    "logistics": "You are discussing logistics and supply chain management.",
    "real_estate": "You are talking about real estate investing and property management.",
    "personal_development": "You are discussing personal development and self-improvement.",
        
        }

        # Check if the given scenario is valid
        if scenario not in scenarios:
            await interaction.channel.send("Invalid scenario. Please choose a valid one.")
            return

        # Translate the system message to the chosen language
        prompt = f"Translate the following text to {language}: '{scenarios[scenario]}'"
        completion = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=500,
            n=1,
            stop=None,
            temperature=0.8,
        )
        translated_system_message = completion.choices[0].text.strip()

        await interaction.response.send_message(f"Engaging in a conversation about {scenario} in {language}. Type 'quit' at any time to end the conversation.")

        # Initialize the conversation
        conversation_history = [
            {"role": "system", "content": translated_system_message}
        ]

        while True:
            # Wait for user input
            def check(m):
                return m.author.id == interaction.user.id and m.channel.id == interaction.channel.id

            user_message = await self.bot.wait_for("message", check=check)
            if user_message.content.lower() == "quit":
                await interaction.channel.send("Conversation ended. Goodbye!")
                break

            # Add user input to conversation history
            conversation_history.append({"role": "user", "content": user_message.content})

            # Generate bot response
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=conversation_history,
                max_tokens=500,
                n=1,
                stop=None,
                temperature=0.8,
            )
            bot_response = completion.choices[0].message["content"].strip()

            # Add bot response to conversation history
            conversation_history.append({"role": "assistant", "content": bot_response})

            # Send the response to the user
            await interaction.channel.send(bot_response)
            
            
            
    @quiz1.subcommand()
    async def translation_practice(self,interaction: nextcord.Interaction, target_language: str):
        await interaction.response.send_message("Type 'quit' at any time to end the translation practice.")

        while True:
            # Request a sentence to translate from GPT-3.5-turbo
            prompt = f"Generate a simple sentence in English for translation practice:"
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates text."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                n=1,
                stop=None,
                temperature=0.8,
            )
            sentence = completion.choices[0].message["content"].strip()

            # Request translation of the sentence
            prompt = f"Translate the following sentence from English to {target_language}: '{sentence}'"
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that translates text."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                n=1,
                stop=None,
                temperature=0.8,
            )
            response = completion.choices[0].message["content"].strip()

            # Send the sentence to the user and ask for their translation
            await interaction.channel.send(f"Translate the following sentence to {target_language}:\n\n{sentence}")

            # Check the user's translation and provide feedback
            def check(m):
                return m.author.id == interaction.user.id and m.channel.id == interaction.channel.id

            try:
                user_translation = await self.bot.wait_for("message", check=check, timeout=60)
                if user_translation.content.lower() == "quit":
                    await interaction.channel.send("Translation practice ended. Goodbye!")
                    break

                if user_translation.content.lower() == response.lower():
                    await interaction.channel.send("Correct! Good job! 🎉")
                else:
                    await interaction.channel.send(f"Almost! The correct translation is:\n\n{response}")
            except asyncio.TimeoutError:
                await interaction.channel.send("Time's up! Please try again.")
                
def setup(bot):
    bot.add_cog(LanguageQuizCog(bot))