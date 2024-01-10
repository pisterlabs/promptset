from functools import lru_cache
from pytz import timezone
import interactions
import datetime
import config
import openai
import uuid
from interactions import ActionRow, Button, ButtonStyle, Client, ComponentContext, Embed, Extension, Guild, Modal, ModalContext, ParagraphText, Permissions, ShortText, StringSelectMenu, StringSelectOption, component_callback, listen
from discord_typings import SelectMenuComponentData, SelectMenuOptionData
from repository.macro_repository import MacroRepository
from entities.macro_entity import Macro
from interactions.ext.paginators import Paginator
import logging

logger = logging.getLogger(__name__)

EMBED_COLOR = 0x71368a
MAX_MACRO_LENGTH = 900
MAX_EMBED_FIELD_LENGTH = 1024
EMBED_FIELD_COUNT = 5
OPENAPI_TOKEN = config.OPENAPI_TOKEN

openai.api_key = OPENAPI_TOKEN


class Macros(Extension):
    def __init__(self, client: Client) -> None:
        self.client: Client = client
        self.macro_repository = MacroRepository()
        self.guild_macros = {}

    """Class for operations around macros.

    NOTE: The LRU Cache in the get_all_macros function prevents the bot from 
    constantly querying the database to retrieve all macros. It stores up to a 
    maximum of 50 macros in cache. This is not intended to reduce the bot's 
    memory footprint (as the memory usage in this case is quite low), but to 
    minimize database requests.

    """

    @interactions.listen('GUILD_CREATE')
    async def on_guild_create(self, guild: Guild):
        await self.update_guild_macros(guild.id)

    async def update_guild_macros(self, guild_id: int):
        self.guild_macros[guild_id] = self.get_all_macros(guild_id)

    @lru_cache(maxsize=50)  # Adjust the maxsize based on your own requeriments
    def get_all_macros(self, guild_id: int):
        try:
            return list(self.macro_repository.find_all(guild_id))
        except Exception as e:
            logger.error(f"Failed to get all macros for guild {guild_id}: {e}")
            return []
        
    @interactions.slash_command(
        name="macro-add", 
        description="Adiciona uma macro.",
        default_member_permissions=interactions.Permissions.ADMINISTRATOR
    )
    async def macro_add(self, ctx: interactions.SlashContext):
        modal = Modal(
            ShortText(label="Nome da Macro", custom_id="macro-title", placeholder="Insira um título", required=True),
            ParagraphText(label="Texto da Macro", custom_id="macro-text", placeholder="Insira uma descrição", required=True, min_length=1, max_length=2000),
            title="Adicionar Macro",
            custom_id="macro_add_form"
        )
        await ctx.send_modal(modal)        
        modal_ctx: ModalContext = await ctx.bot.wait_for_modal(modal)
        macro_title = modal_ctx.responses["macro-title"]
        macro_text = modal_ctx.responses["macro-text"]
        await modal_ctx.defer()
        macro_created = self.create_macro(ctx.guild_id, macro_title, macro_text)
        if macro_created:
            await self.update_guild_macros(ctx.guild_id)
            await modal_ctx.send(f"Macro **{macro_title}** adicionada.")
        else:
            await modal_ctx.send(f"Macro **{macro_title}** já existe!")

    @interactions.slash_command(
        name="macro-edit", 
        description="Edita uma macro.",
        default_member_permissions=interactions.Permissions.ADMINISTRATOR,
        options=[
            interactions.SlashCommandOption(
                name="macro",
                description="macro para editar",
                type=interactions.OptionType.STRING,
                required=True,
            )
        ],
    )
    async def macro_edit(self, ctx: interactions.SlashContext, macro: str):
        updated_macro = self.get_macro_by_title_and_guild_id(macro, ctx.guild_id)
        if updated_macro is None:
            await ctx.send(f"A macro **{macro}** não existe neste servidor!")
        else:
            modal = Modal(
                ParagraphText(label="Texto da Macro", custom_id="macro-text", placeholder="Edite o texto da macro", value=updated_macro.text, required=True, min_length=1, max_length=2000),
                title="Edite a Macro",
                custom_id="macro_edit_form"
            )
            await ctx.send_modal(modal)
            modal_ctx: ModalContext = await ctx.bot.wait_for_modal(modal)
            await modal_ctx.defer()
            
            macro_new_text = modal_ctx.responses["macro-text"]
            old_macro_text = updated_macro.text
            
            updated_macro.text = macro_new_text
            self.update_macro(updated_macro)

            macro_formated_old_text = self.trim_text(old_macro_text)
            macro_formated_new_text = self.trim_text(updated_macro.text)
            embed = Embed(
                title=f"Macro \"{macro}\" editada!",
                color=0x71368a,
                timestamp=datetime.datetime.now(timezone('UTC'))
            )
            macro_formated_old_text = f"```diff\n- {macro_formated_old_text}```"  
            macro_formated_new_text = f"```diff\n+ {macro_formated_new_text}```" 

            embed.add_field(name="Antes:", value=macro_formated_old_text, inline=False)
            embed.add_field(name="Depois:", value=macro_formated_new_text, inline=False)

            await self.update_guild_macros(ctx.guild_id)
            await modal_ctx.send(embed=embed)
    
    @interactions.slash_command(
        name="macro-delete", 
        description="Remove uma macro.",
        default_member_permissions=interactions.Permissions.ADMINISTRATOR,
        options=[
            interactions.SlashCommandOption(
                name="macro",
                description="macro para remover",
                type=interactions.OptionType.STRING,
                required=True,
            )
        ],
    )
    async def macro_delete(self, ctx: interactions.SlashContext, macro: str):
        await ctx.defer()
        find_macro = self.get_macro_by_title_and_guild_id(macro, ctx.guild_id)
        if find_macro is None:
            await ctx.send(f"A macro **{macro}** não existe neste servidor!")
        else:
            self.delete_macro(find_macro)
            await self.update_guild_macros(ctx.guild_id)
            await ctx.send(f"A macro **{macro}** foi deletada!")

    @interactions.listen()
    async def on_message_create(self, message_create: interactions.events.MessageCreate):
        message = message_create.message
        guild_id = message_create.message.channel.guild.id

        if guild_id not in self.guild_macros:
            macro_repository = MacroRepository()
            self.guild_macros[guild_id] = list(macro_repository.find_all(guild_id))

        for macro in self.guild_macros[guild_id]:
            if macro.title.strip() == message.content.strip():
                await message.channel.send(macro.text)
                break  



    @interactions.slash_command(
        name="macro-list", 
        description="Lista todas macros do servidor.",
        default_member_permissions=interactions.Permissions.ADMINISTRATOR
    )
    async def macro_list(self, ctx: interactions.SlashContext):
        guild_id = ctx.guild_id
        await self.update_guild_macros(guild_id)
        macros = self.guild_macros.get(guild_id, [])

        if not macros:
            await ctx.send("Ainda não há macros nesse servidor!")
        else:
            embeds = []
            embed = Embed(
                title="Macros do Servidor",
                description="Todas as macros deste servidor:",
                color=0x71368a, 
                timestamp=datetime.datetime.now(timezone('UTC'))
            )
            count = 0
            for macro in macros:
                if macro.text.startswith("http") and macro.text.rsplit('.', 1)[-1] in {'png', 'jpg', 'jpeg', 'gif'}:
                    value = f"[Imagem]({macro.text})"
                else:
                    if len(macro.text) > 1024:
                        value = macro.text[:50] + "..."
                    else:
                        value = macro.text
                embed.add_field(name=macro.title, value=value, inline=False)
                count += 1
                if count == 5:
                    embeds.append(embed)
                    count = 0
                    embed = Embed(
                        title="Macros do Servidor",
                        description="Todas as macros deste servidor:",
                        color=0x71368a, 
                        timestamp=datetime.datetime.now(timezone('UTC'))
                    )
            if count > 0:
                embeds.append(embed)
            paginator = Paginator.create_from_embeds(ctx.bot, *embeds)
            paginator.default_button_color = ButtonStyle.BLUE
            await paginator.send(ctx)

    @interactions.slash_command(
        name="macro-delete-all", 
        description="Deleta todas as macros do servidor.",
        default_member_permissions=Permissions.ADMINISTRATOR
    )
    async def macro_delete_all(self, ctx: interactions.SlashContext):
        guild_id = ctx.guild_id
        if guild_id not in self.guild_macros:
            await self.update_guild_macros(guild_id)
        macros = self.guild_macros.get(guild_id, [])

        if not macros:
            await ctx.send("Ainda não há macros nesse servidor!")
        else:
            components = [
                ActionRow(
                    Button(
                        custom_id="confirm_delete_all_macro_button",
                        style=ButtonStyle.GREEN,
                        label="Confirm",
                    ),
                    Button(
                        custom_id="cancel_delete_all_macro_button",
                        style=ButtonStyle.RED,
                        label="Cancel",
                    ),
                ),
            ]
            await ctx.send("Você tem certeza que quer deletar todas as macros?", components=components, ephemeral=True)

    @component_callback("confirm_delete_all_macro_button")
    @interactions.auto_defer()
    async def confirm_callback(self, ctx: ComponentContext):
        guild_id = ctx.guild_id
        macro_repository = MacroRepository()
        macro_repository.delete_all_macros(guild_id)
        await self.update_guild_macros(guild_id)
        await ctx.send(f"O usuario {ctx.author.mention} deletou todas as macros.")

    @component_callback("cancel_delete_all_macro_button")
    @interactions.auto_defer()
    async def cancel_callback(self, ctx: ComponentContext):
        await ctx.send("Operação cancelada.", ephemeral=True)

    """Repository and format utilies"""

    def create_macro(self, guild_id: int, macro_title: str, macro_text: str):
        macro = self.get_macro_by_title_and_guild_id(macro_title, guild_id)
        if macro is None:
            new_macro = Macro(macro_id=uuid.uuid4(), guild_id=guild_id, title=macro_title, text=macro_text)
            try:
                self.macro_repository.create_macro(new_macro)
                logger.info(f"Created new macro {macro_title} for guild {guild_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to create macro {macro_title} for guild {guild_id}: {e}")
                raise
        else:
            logger.warning(f"Macro {macro_title} for guild {guild_id} already exists")
            return False

    def update_macro(self, macro: Macro):
        try:
            self.macro_repository.update_macro(macro.macro_id, macro)
            return macro
        except Exception as e:
            logger.error(f"Failed to update macro {macro.title} for guild {macro.guild_id}: {e}")
            return None

    def delete_macro(self, macro: Macro):
        try:
            self.macro_repository.delete_macro(macro.macro_id)
            return True
        except Exception as e:
            logger.error(f"Failed to delete macro {macro.title} for guild {macro.guild_id}: {e}")
            return False

    def get_macro_by_title_and_guild_id(self, macro_title: str, guild_id: int):
        try:
            return self.macro_repository.get_macro_by_title_and_guild_id(macro_title, guild_id)
        except Exception as e:
            logger.error(f"Failed to get macro {macro_title} for guild {guild_id}: {e}")
            raise

    def trim_text(self, text):
        if len(text) > 900:
            return text[:850] + "..."
        return text
        
