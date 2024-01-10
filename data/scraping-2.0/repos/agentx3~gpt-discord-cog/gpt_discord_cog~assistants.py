import discord
from openai.types.beta.assistant import Assistant
from .lib.types import GPTConfig


class ModifyAssistantModal(discord.ui.Modal):
    def __init__(self, *args, config: GPTConfig, assistant: Assistant, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        current_instructions = assistant.instructions
        current_description = assistant.description
        self.add_item(
            discord.ui.InputText(
                label="Instructions",
                placeholder="Enter instructions",
                style=discord.InputTextStyle.long,
                value=current_instructions,
            )
        )
        self.add_item(
            discord.ui.InputText(
                label="Description",
                placeholder="Enter instructions",
                style=discord.InputTextStyle.long,
                value=current_description,
            )
        )

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer()
        try:
            instructions = self.children[0].value
            description = self.children[1].value
            await self.config["client"].beta.assistants.update(
                self.config["assistant_id"],
                instructions=instructions,
                description=description,
            )
            await interaction.followup.send(content="Updated assistant instructions")
        except Exception as e:
            await interaction.followup.send(f"Error: {e}")


async def modify_assistant(ctx: discord.ApplicationContext, config: GPTConfig):
    """Modify the assistant's instructions"""
    # The assistant needs to be retrieved here because it is asynchronous and we cannot
    # use it in the modal constructor
    current_assistant = await config["client"].beta.assistants.retrieve(
        config["assistant_id"]
    )
    modal = ModifyAssistantModal(
        config=config,
        assistant=current_assistant,
        title="Change Assistant Instructions",
    )
    await ctx.send_modal(modal)
