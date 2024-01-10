import interactions
from interactions import Client, Extension, Guild, Modal, ModalContext, ParagraphText
import openai
import config

openai.api_key = config.OPENAPI_TOKEN

class MemberManage(Extension):
    def __init__(self, client: Client) -> None:
        self.client: Client = client
        
    def split_and_parse_member_ids(self, member_ids: str):
        return [int(a) for a in member_ids.split('\n') if a]
    
    async def ban_members(self, guild: Guild, member_ids: list[int]):
        for member_id in member_ids:
            member = await guild.fetch_member(member_id)
            if member is not None:
                await guild.ban(member)

    @interactions.slash_command(
        name="massban", 
        description="Banir membros em massa.",
        default_member_permissions=interactions.Permissions.ADMINISTRATOR
    )
    async def mass_ban(self, ctx: interactions.SlashContext):
        modal = Modal(
            ParagraphText(label="ID dos Membros", custom_id="members"),
            title="Banir Membros em Massa",
            custom_id="mass_ban_form"
        )        
        await ctx.send_modal(modal)        

        modal_ctx: ModalContext = await ctx.bot.wait_for_modal(modal)
        member_str = modal_ctx.responses["members"]
        member_ids = self.split_and_parse_member_ids(member_str)
        await self.ban_members(ctx.guild, member_ids)
        await modal_ctx.send(f"{len(member_ids)} usuários banidos.")
    
    async def kick_members(self, guild: Guild, member_ids: list[int]):
        for member_id in member_ids:
            member = await guild.fetch_member(member_id)
            if member is not None:
                await guild.kick(member)

    @interactions.slash_command(
        name="masskick", 
        description="Expulsar membros em massa.",
        default_member_permissions=interactions.Permissions.ADMINISTRATOR
    )
    async def mass_kick(self, ctx: interactions.SlashContext):
        modal = Modal(
            ParagraphText(label="ID dos Membros", custom_id="members"),
            title="Expulsar Membros em Massa",
            custom_id="mass_kick_form"
        )        
        await ctx.send_modal(modal)        

        modal_ctx: ModalContext = await ctx.bot.wait_for_modal(modal)
        member_str = modal_ctx.responses["members"]
        member_ids = self.split_and_parse_member_ids(member_str)
        await self.kick_members(ctx.guild, member_ids)
        await modal_ctx.send(f"{len(member_ids)} usuários expulsos.")

def setup(client):
    MemberManage(client)