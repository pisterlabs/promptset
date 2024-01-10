import os
from dotenv import load_dotenv
load_dotenv()
import disnake
from disnake.ext import commands
from disnake import TextInputStyle
import aiohttp
from typing import List



key = os.environ.get('YOUR_OPENAI_KEY')

from openai import OpenAI
from openai.types.beta.assistant import Assistant

class AssistantCOG(commands.Cog):
    def __init__(self, bot):
        self.bot=bot
        self.client=OpenAI(api_key=key)
       

    @commands.slash_command()
    async def assistant(self, inter):
        pass



    @assistant.sub_command()
    async def create(self, inter:disnake.AppCmdInter):
     

        await inter.response.send_modal(modal=AssistantModal(self.bot, self.client))


    @assistant.sub_command()
    async def work(self, inter:disnake.AppCmdInter, assistant_id, thread_id:str):
        """Pick up where you left off with your agent."""
        await inter.response.defer(ephemeral=False)

        await inter.edit_original_message(view=AssistantView(self.bot, assistant_id=assistant_id,client=self.client, thread_id=thread_id))





# Subclassing the modal.
class AssistantModal(disnake.ui.Modal):
    def __init__(self,bot, client :OpenAI):
        self.bot=bot
        self.client=client
        # The details of the modal, and its components
        components = [
            disnake.ui.TextInput(
                label="Name",
                placeholder="Give your assistant a name.",
                custom_id="assistant_name",
                style=TextInputStyle.short,
            ),

        ]
        super().__init__(title="Create Assistant:", components=components)

    # The callback received when the user input is completed.
    async def callback(self, inter: disnake.ModalInteraction):
        await inter.response.defer(with_message=True, ephemeral=False)

        assistant = self.client.beta.assistants.create(
        name=inter.text_values.get('assistant_name'),
        instructions=f"You are a helpful assistant. You will help the user to the best of your abilities. You will send back downloadable files to the user.",
        tools=[{"type": "code_interpreter"}],
        model="gpt-4-1106-preview"
    )
        embed = disnake.Embed(title="Assistant Created!", description=f"```py\nYour assistant has been created with the name {inter.text_values.get('name')}. Save the ids below this message, and use /assistants work to continue where you left off. ALL AGENTS will have long-term memory as in you can pick up right where you left off so long as you have your thread_id and assistant IDs handy.```")
        embed.add_field(name=f"Your assistant ID:", value=f"{assistant.id} || Save this for your records.")
        embed.set_footer(text=f"{assistant.id}")

        await inter.edit_original_message(embed=embed, view=AssistantView(self.bot, client=self.client,assistant_id=assistant.id))
    


# Subclassing the modal.
class TaskModal(disnake.ui.Modal):
    def __init__(self,bot, client :OpenAI, assistant_id, thread_id = None):
        self.thread_id = thread_id
        self.assistant_id = assistant_id
        self.bot=bot
        self.client=client
        # The details of the modal, and its components
        components = [
            disnake.ui.TextInput(
                label="Task 1",
                placeholder="The first task. Can be a part of a larger task.",
                custom_id="task1",
                style=TextInputStyle.short,
            ),
            disnake.ui.TextInput(
                label="Task 2",
                placeholder="The second task - builds on the first one.",
                custom_id="task2",
                style=TextInputStyle.paragraph,
                required=False
            ),
            disnake.ui.TextInput(
                label="Task 3",
                placeholder="The third task - builds on the first two.",
                custom_id="task3",
                style=TextInputStyle.paragraph,
                required=False
            ),

            disnake.ui.TextInput(
                label="Task 4",
                placeholder="The fourth task - builds on the first three.",
                custom_id="task4",
                style=TextInputStyle.paragraph,
                required=False
            ),
            disnake.ui.TextInput(
                label="Task 5",
                placeholder="The second task - builds on the first four.",
                custom_id="task5",
                style=TextInputStyle.paragraph,
                required=False
            ),
        ]
        super().__init__(title="Create a job:", components=components)

    async def callback(self, inter: disnake.ModalInteraction):
      

        # Start with an empty prompt
        prompt = ""

        # Iterate through the text inputs
        for i in range(1, 6):
            task_key = f"task{i}"
            task_value = inter.text_values.get(task_key)

            # Append the task to the prompt if it's not empty
            if task_value:
                prompt += f"Task {i}: {task_value}\n"

        # Now `prompt` contains all the entered tasks, separated by new lines
        # Continue with your logic to handle this prompt

        # Example: Create a thread and send the message
        if self.thread_id == None:
            thread = self.client.beta.threads.create()
            thread_id = thread.id

        else:

            thread_id = self.thread_id

        message = self.client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=prompt
        )

        embed = disnake.Embed(title=f"Work Given!", description=f"```py\nYou've just created a thread, and created work for your assistant. Use the 'HISTORY' button to check for its' response. NOTE: Response can take up to 1 minute.```")
        embed.add_field(name=f"IMPORTANT:", value=f"> Keep these IDs for your records to use this agent again in the future:\n\n> Assistant ID: **{self.assistant_id}**\n> Thread ID: **{thread_id}**")
        run = self.client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=self.assistant_id,
        instructions=prompt
        )
        if run.file_ids:
            embed.add_field(name=f"File IDs:", value=f"> **{run.file_ids}**")
        await inter.response.edit_message(embed=embed, view=AssistantView(bot=self.bot, assistant_id=self.assistant_id, client=self.client, thread_id=thread_id))




class AssistantView(disnake.ui.View):
    def __init__(self, bot: commands.Bot, assistant_id, client: OpenAI, thread_id=None, file_id=None, embeds: List[disnake.Embed] = None):
        self.file_id = file_id
        self.thread_id = thread_id
        self.bot=bot
        self.assistant_id = assistant_id
        self.client = client
        self.embeds = embeds
        self.embed_count = 0
        self.count = 0
        super().__init__(timeout=None)

        if self.thread_id is not None:
            self.ass12.disabled = False
            self.ass11.disabled = True
    @disnake.ui.button(style=disnake.ButtonStyle.gray, row=0, custom_id='assistant1', emoji="<a:_:963244979063517184>")
    async def ass1(self, button:disnake.ui.Button, inter:disnake.AppCmdInter):
        pass


    @disnake.ui.button(style=disnake.ButtonStyle.gray, row=0, custom_id='assistant2', emoji="<a:_:963244979063517184>")
    async def ass2(self, button:disnake.ui.Button, inter:disnake.AppCmdInter):
        pass

    @disnake.ui.button(style=disnake.ButtonStyle.gray, row=0, custom_id='assistant3', emoji="<a:_:963244979063517184>")
    async def ass3(self, button:disnake.ui.Button, inter:disnake.AppCmdInter):
        pass


    @disnake.ui.button(style=disnake.ButtonStyle.green, row=1, custom_id='assistant11', label="WORK")
    async def ass11(self, button:disnake.ui.Button, inter:disnake.AppCmdInter):
        await inter.response.send_modal(TaskModal(self.bot, self.client, assistant_id=self.assistant_id))


    @disnake.ui.button(style=disnake.ButtonStyle.gray, row=1, custom_id='assistant12', emoji="<a:_:1044653187212247133>", disabled=True)
    async def ass12(self, button:disnake.ui.Button, inter:disnake.AppCmdInter):
        messages = self.client.beta.threads.messages.list(
        thread_id=self.thread_id
        )
        id = messages.data[0].thread_id
        await inter.response.send_modal(TaskModal(self.bot, self.client, assistant_id=self.assistant_id, thread_id=id))


    @disnake.ui.button(style=disnake.ButtonStyle.gray, row=2, custom_id='assistant4', label='HISTORY')
    async def ass4(self, button:disnake.ui.Button, inter:disnake.AppCmdInter):
        await inter.response.defer(ephemeral=False)
        messages = self.client.beta.threads.messages.list(
        thread_id=self.thread_id
        )

        data = messages.data

        msg_content = [i.content for i in data]
        text = [i[0].text.value for i in msg_content]  
        
        embeds = []

        for i in text:
            embed = disnake.Embed(title=f"Agent History", description=f"```py\n{i}```")

            embed.add_field(name=f"IDs:", value=f"> Assistant ID: **{self.assistant_id}**\n> Thread ID: **{self.thread_id}**")
            embeds.append(embed)

        
        await inter.edit_original_message(embed=embeds[0], view=AssistantView(embeds=embeds,bot=self.bot,assistant_id=self.assistant_id, client=self.client, thread_id = self.thread_id))



        


    @disnake.ui.button(
        emoji="<a:_:1042677512284680321>",
        style=disnake.ButtonStyle.red,
        custom_id=f"persistent_view:prevqasdasdwfpage_{str(disnake.Member)}aq2wfwa",
        row=4,
        label=f"🇵 🇷 🇪 🇻"

    )
    async def prev_page(  # pylint: disable=W0613
        self,
        button: disnake.ui.Button,
        interaction: disnake.MessageInteraction,
    ):
        # Decrements the embed count.
        self.embed_count -= 1

        # Gets the embed object.
        embed = self.embeds[self.embed_count]

        # Enables the next page button and disables the previous page button if we're on the first embed.
        self.next_page.disabled = False

        await interaction.response.edit_message(embed=embed, view=self)


    @disnake.ui.button(
        emoji="<a:_:1042677591196319765>",
        style=disnake.ButtonStyle.red,
        custom_id=f"persistent_view:asdasdnextpage_{str(disnake.Member)}awfawwa",
        label=f"🇳 🇪 🇽 🇹",
        row=4
    )
    async def next_page(
        self,
        button: disnake.ui.Button,
        interaction: disnake.MessageInteraction,
    ):
        # Checks if self.embed_count is within the valid range
        if 0 <= self.embed_count < len(self.embeds):
            # Increments the embed count
            self.embed_count += 1

            # Gets the embed object
            embed = self.embeds[self.embed_count]

            # Enables the previous page button and disables the next page button if we're on the last embed
            self.prev_page.disabled = False
            if self.embed_count == len(self.embeds) - 1:
                self.next_page.disabled = True

            await interaction.response.edit_message(embed=embed, view=self)





def setup(bot:commands.Bot):
    bot.add_cog(AssistantCOG(bot))

    print(f"assistant READY!")