import asyncio
import logging
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Union

import discord
from httpx import delete
import openai
import tiktoken
import unidecode
from discord import app_commands
from discord.ext import commands

from common import dataio
from common.utils import fuzzy, pretty

logger = logging.getLogger(f'Wanderlust.{__name__.capitalize()}')

AI_MODEL = 'gpt-3.5-turbo'
MAX_TOKENS = 400
MAX_CONTEXT_SIZE = 4096
DEFAULT_CONTEXT_SIZE = 1024
EMBED_DEFAULT_COLOR = 0x2b2d31

class ConfirmationView(discord.ui.View):
    """Ajoute un bouton de confirmation et d'annulation à un message"""
    def __init__(self, *, custom_labels: tuple[str, str] | None = None, timeout: float | None = 60):
        super().__init__(timeout=timeout)
        self.value = None
        
        if custom_labels is None:
            custom_labels = ('Confirmer', 'Annuler')
        self.confirm.label = custom_labels[0]
        self.cancel.label = custom_labels[1]
        
    @discord.ui.button(style=discord.ButtonStyle.green)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.value = True
        self.stop()
        
    @discord.ui.button(style=discord.ButtonStyle.red)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.value = False
        self.stop()
        
class ContinueButtonView(discord.ui.View):
    """Ajoute un bouton pour indiquer au Chatbot qu'il doit fournir la suite de sa réponse"""
    def __init__(self, *, timeout: float | None = 60, author: discord.User | discord.Member | None = None):
        super().__init__(timeout=timeout)
        self.author = author
        self.value = None
        
    @discord.ui.button(style=discord.ButtonStyle.gray, label='Continuer', emoji='<:iconContinue:1135604595624783953>')
    async def continue_(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.value = True
        self.stop()

    async def on_timeout(self) -> None:
        self.value = False
        self.stop()
        
    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        return interaction.user.id == self.author.id if self.author else True
    
class AddAskContextModal(discord.ui.Modal):
    def __init__(self, chatbot: Union['CustomChatbot', 'TempChatbot'], initial_message: discord.Message, *, timeout: float | None = None) -> None:
        super().__init__(title=f"Demander à {chatbot}", timeout=timeout)
        self.chatbot = chatbot 
        self.initial_message = initial_message
        
        self.context = discord.ui.TextInput(label='Introduire la citation', placeholder="Entrez une demande concernant cette citation (facultatif)", max_length=200, required=False)
        self.add_item(self.context)
        
    async def on_submit(self, interaction: discord.Interaction) -> None:
        self.value = self.context.value if self.context.value else None
        if self.value:
            await interaction.response.send_message(f"**Demande à l'IA** · Le message de *{self.initial_message.author.display_name}* a été envoyé au chatbot **{self.chatbot}** avec comme demande `{self.value}`", delete_after=120)
        else:
            await interaction.response.send_message(f"**Demande à l'IA** · Le message de *{self.initial_message.author.display_name}* a été envoyé au chatbot **{self.chatbot}**", delete_after=30)
    
class ChatbotList(discord.ui.View):
    """Affiche les détails sur les chatbots d'un serveur"""
    def __init__(self, chatbots: List['CustomChatbot'], *, timeout: float | None = 60, starting_page: int = 0, user: discord.User | discord.Member | None = None):
        super().__init__(timeout=timeout)
        self.chatbots = sorted(chatbots, key=lambda c: c.name)
        self.current_page = starting_page
        self.user = user
        
        self.select_chatbot.options = self.add_options()
        if len(self.chatbots) == 1:
            self.select_chatbot.disabled = True
    
    def _get_page(self) -> discord.Embed:
        """Récupère la page actuelle."""
        chatbot = self.chatbots[self.current_page]
        em = chatbot.embed
        em.set_footer(text='Utilisez la liste ci-dessous pour naviguer entre les Chatbots')
        return em
        
    async def start(self, interaction: discord.Interaction) -> None:
        await interaction.followup.send(embed=self._get_page(), view=self)
        self.initial_interaction = interaction
            
    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if self.user is None:
            return True
        return interaction.user.id == self.user.id
    
    async def on_timeout(self) -> None:
        self.clear_items()
        if self.initial_interaction.message:
            em = self.initial_interaction.message.embeds[0]
            em.set_footer(text=None)
            await self.initial_interaction.edit_original_response(embed=em, view=None)
        else:
            await self.initial_interaction.edit_original_response(view=None)
        
    def add_options(self):
        options = []
        for c in self.chatbots:
            options.append(discord.SelectOption(label=c.name, value=str(c.id)))
        return options
        
    @discord.ui.select(placeholder='Sélectionnez un Chatbot', min_values=1, max_values=1)
    async def select_chatbot(self, interaction: discord.Interaction, select: discord.ui.Select):
        await interaction.response.defer()
        chatbot_id = int(select.values[0])
        for i, c in enumerate(self.chatbots):
            if c.id == chatbot_id:
                self.current_page = i
                break
        await self.initial_interaction.edit_original_response(embed=self._get_page())

class CustomChatbot:
    def __init__(self, cog: 'Chatter', guild: discord.Guild, profile_id: int, *, resume: bool = True, debug: bool = False):
        """Représente un chatbot IA personnalisé.

        :param cog: Module Chatter
        :param guild: Serveur sur lequel le chatbot est utilisé
        :param profile_id: ID du profil
        :param resume: Si True, charge la dernière session de messages du chatbot
        """
        self._cog = cog
        self.guild = guild
        self.id = profile_id
        
        self.stats = ChatbotStats(self)
        self.logs = ChatbotLogs(self, resume=resume)
        
        self._debug = debug
        self.__load()
        
    def __repr__(self) -> str:
        return f'<CustomChatbot id={self.id}>'
    
    def __str__(self) -> str:
        name = self.name.title()
        if self._debug:
            name += ' [DEBUG]'
        return name
    
    def __load(self) -> None:
        """Charge les données du profil."""
        query = """SELECT * FROM profiles WHERE id = ?"""
        data = self._cog.data.fetchone(self.guild, query, (self.id,))
        if not data:
            raise ValueError(f'Profil {self.id} introuvable.')
        
        # Données du profil
        self.name = data['name']
        self.description = data['description']
        self.avatar_url = data['avatar_url'] or self._cog._get_default_avatar(data['system_prompt'])
        self.system_prompt = data['system_prompt']
        self.temperature = float(data['temperature'])
        self.context_size = int(data['context_size'])
        self.features = data['features'].split(';') if data['features'] else []
        self.blacklist = [int(i) for i in data['blacklist'].split(';')] if data['blacklist'] else []
        self.author_id = int(data['author_id'])
        self.created_at = float(data['created_at'])
    
    def save(self) -> None:
        """Enregistre les données du profil."""
        query = """UPDATE profiles SET
            name = ?,
            description = ?,
            avatar_url = ?,
            system_prompt = ?,
            temperature = ?,
            context_size = ?,
            features = ?,
            blacklist = ?,
            author_id = ?,
            created_at = ?
            WHERE id = ?"""
        self._cog.data.execute(self.guild, query, (
            self.name,
            self.description,
            self.avatar_url,
            self.system_prompt,
            self.temperature,
            self.context_size,
            ';'.join(self.features),
            ';'.join([str(i) for i in self.blacklist]),
            self.author_id,
            self.created_at,
            self.id
        ))
        
    def _get_embed(self):
        """Récupère un embed représentant le chatbot."""
        em = discord.Embed(title=f'***{str(self)}***', description=f'*{self.description}*', color=EMBED_DEFAULT_COLOR)
        em.set_thumbnail(url=self.avatar_url)
        em.add_field(name="Prompt d'initialisation", value=f'```{pretty.troncate_text(self.system_prompt, 1000)}```', inline=False)
        em.add_field(name="Température", value=f'`{self.temperature}`')
        em.add_field(name="Taille du contexte", value=f'`{self.context_size} tokens`')
        creator = self.guild.get_member(self.author_id)
        date = datetime.fromtimestamp(self.created_at).strftime('%d/%m/%Y %H:%M')
        if creator:
            em.add_field(name="Création", value=f'`{date}`\npar {creator.mention}')
        else:
            em.add_field(name="Création", value=f'`{date}`\npar {self.author_id}')
        if self.features:
            em.add_field(name="Fonctions activées", value='\n'.join(f'`{f}`' for f in self.features))
        if self.blacklist:
            em.add_field(name="Nb. d'élements blacklistés", value=f'`{len(self.blacklist)}`')
        if self._debug:
            em.add_field(name="Mode debug", value="Activé")
            
        # Stats
        text = f"- ***Utilisations*** · `{self.stats.uses}`\n- ***Messages*** · `{self.stats.messages}`\n- ***Tokens*** · `{self.stats.tokens}`\n- ***Moy. tokens/réponse*** · `{self.stats.average_tokens:.2f}`"
        em.add_field(name="--- Statistiques ----", value=text, inline=False)
        return em
    
    @property
    def embed(self) -> discord.Embed:
        """Récupère un embed représentant le chatbot."""
        return self._get_embed()
    
    @property
    def context(self) -> List[dict]:
        """Récupère le contexte du chatbot (derniers messages dans la limite de la taille du contexte)."""
        return self.logs._get_context(self.context_size)
                
class ChatbotLogs:
    """Représente les logs d'un chatbot IA."""
    def __init__(self, chatbot: CustomChatbot, *, resume: bool = True):
        self._chatbot = chatbot
        self.guild = chatbot.guild
        self.id = chatbot.id
        
        self._cog = chatbot._cog
        self._session_id = self.get_last_session_id() + 1 if not resume else self.get_last_session_id()
        
        self.messages : List[dict] = self.__load_messages()
        
    def __load_messages(self) -> List[Dict[str, Any]]:
        """Charge les messages du chatbot."""
        query = """SELECT * FROM messages WHERE profile_id = ? AND session_id = ? ORDER BY timestamp ASC"""
        data = self._cog.data.fetchall(self.guild, query, (self.id, self._session_id))
        return [dict(m) for m in data] if data else []
        
    def save(self) -> None:
        """Enregistre les messages du chatbot."""
        last_save = self.__load_messages()
        if self.messages == last_save:
            return
        
        if len(self.messages) < len(last_save):
            # Si des messages ont été supprimés, on nettoie la base de données de la session actuelle pour la remettre au propre
            query = """DELETE FROM messages WHERE profile_id = ? AND session_id = ?"""
            self._cog.data.execute(self.guild, query, (self.id, self._session_id))

        query = """INSERT OR REPLACE INTO messages VALUES (?, ?, ?, ?, ?, ?)"""
        self._cog.data.executemany(self.guild, query, [(m['timestamp'], self.id, self._session_id, m['role'], m['content'], m['username']) for m in self.messages])
        
    def get_last_session_id(self) -> int:
        """Récupère l'ID de la dernière session du chatbot."""
        query = """SELECT MAX(session_id) AS session_id FROM messages WHERE profile_id = ?"""
        data = self._cog.data.fetchone(self.guild, query, (self.id,))
        return data['session_id'] or 0
        
    def add_message(self, timestamp: float, role: str, content: str, username: str | None, *, save: bool = True) -> None:
        """Ajoute un message au chatbot."""
        self.messages.append({
            'timestamp': timestamp,
            'role': role,
            'content': content,
            'username': username
        })
        if save:
            self.save()
        
    def remove_message_by_timestamp(self, timestamp: float) -> None:
        """Supprime un message du chatbot."""
        for message in self.messages:
            if message['timestamp'] == timestamp:
                self.messages.remove(message)
                break
        self.save()
        
    def remove_message_by_content(self, content: str) -> None:
        """Supprime un message du chatbot."""
        for message in self.messages:
            if message['content'] == content:
                self.messages.remove(message)
                break
        self.save()
        
    def remove_messages_by_role(self, role: str, limit: int = 1) -> None:
        """Supprime X messages d'un rôle du chatbot."""
        for message in self.messages:
            if message['role'] == role:
                self.messages.remove(message)
                limit -= 1
                if limit == 0:
                    break
        self.save()
        
    def remove_last_message(self) -> None:
        """Supprime le dernier message du chatbot."""
        self.messages.pop()
        self.save()
        
    # Exploitation
    
    def _get_sanitized_messages(self) -> List[dict]:
        """Formate les messages tels qu'ils doivent être envoyés à l'API."""
        sanitized = []
        for m in self.messages:
            if m['role'] == 'user':
                sanitized.append({'role': m['role'], 'content': m['content'], 'name': m['username']})
            else:
                sanitized.append({'role': m['role'], 'content': m['content']})
        return sanitized
    
    def _get_context(self, context_size: int) -> List[dict]:
        """Récupère le contexte du chatbot (derniers messages dans la limite de la taille du contexte)."""
        encoding = tiktoken.encoding_for_model(AI_MODEL)
        init_size = len(encoding.encode(self._chatbot.system_prompt))
        messages = self._get_sanitized_messages()
        if not messages:
            return [{'role': 'system', 'content': self._chatbot.system_prompt}]
        
        last_message = messages.pop()
        last_message_size = len(encoding.encode(last_message['content']))
        tokens = init_size + last_message_size
        context = []
        for message in reversed(messages):
            if tokens + len(encoding.encode(message['content'])) > context_size:
                break
            context.append(message)
            tokens += len(encoding.encode(message['content']))
        return [{'role': 'system', 'content': self._chatbot.system_prompt}] + context[::-1] + [last_message]

    
class ChatbotStats:
    """Représente les statistiques d'un chatbot IA."""
    def __init__(self, chatbot: CustomChatbot):
        self._chatbot = chatbot
        self.guild = chatbot.guild
        self.id = chatbot.id
        
        self._cog = chatbot._cog
        
        self.__load_stats()
    
    def __load_stats(self) -> None:
        """Charge les statistiques du chatbot."""
        query = """SELECT * FROM stats WHERE profile_id = ?"""
        data = self._cog.data.fetchone(self.guild, query, (self.id,))
        if not data:
            self._uses = 0
            self._messages = 0
            self._tokens = 0
            self._last_use = 0
        else:
            self._uses = data['uses']
            self._messages = data['messages']
            self._tokens = data['tokens']
            self._last_use = data['last_use']
            
    def save_stats(self) -> None:
        """Enregistre les statistiques du chatbot."""
        query = """INSERT OR REPLACE INTO stats (profile_id, uses, messages, tokens, last_use) VALUES (?, ?, ?, ?, ?)"""
        self._cog.data.execute(self.guild, query, (
            self.id,
            self._uses,
            self._messages,
            self._tokens,
            self._last_use
        ))
        
    @property
    def uses(self) -> int:
        """Récupère le nombre d'utilisations du chatbot."""
        return self._uses
    
    @uses.setter
    def uses(self, value: int) -> None:
        """Modifie le nombre d'utilisations du chatbot."""
        self._uses = value
        self.save_stats()
        
    @property
    def messages(self) -> int:
        """Récupère le nombre de messages du chatbot."""
        return self._messages

    @messages.setter
    def messages(self, value: int) -> None:
        """Modifie le nombre de messages du chatbot."""
        self._messages = value
        self.save_stats()
        
    @property
    def tokens(self) -> int:
        """Récupère le nombre de tokens du chatbot."""
        return self._tokens
    
    @tokens.setter
    def tokens(self, value: int) -> None:
        """Modifie le nombre de tokens du chatbot."""
        self._tokens = value
        self.save_stats()
        
    @property
    def average_tokens(self) -> float:
        """Récupère le nombre moyen de tokens par message."""
        return self._tokens / self._messages if self._messages else 0
    
    @property
    def last_use(self) -> float:
        """Récupère le timestamp de la dernière utilisation du chatbot."""
        return self._last_use
    
    def update_last_use(self) -> None:
        """Met à jour le timestamp de la dernière utilisation du chatbot."""
        self._last_use = time.time()
        self.save_stats()
        
class TempChatbot:
    """Représente un chatbot temporaire."""
    def __init__(self, cog: 'Chatter', system_prompt: str, temperature: float, context_size: int, *, author_id: int | None = None, debug: bool = False):
        self._cog = cog
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.context_size = context_size
        
        self.name = cog._get_default_name(system_prompt)
        self.avatar_url = cog._get_default_avatar(system_prompt)
        self.author_id = author_id
        self.created_at = time.time()
        
        self.description = f'Chatbot temporaire'
        self.features = []
        self.blacklist = []
        
        self._debug = debug
        self._context = []
        
    def __repr__(self) -> str:
        return f'<TempChatbot system_prompt={self.system_prompt}>'
    
    def __str__(self) -> str:
        return f'T.{self.name.title()} [DEBUG]' if self._debug else f'T.{self.name.title()}'
    
    def _get_avatar_color(self) -> discord.Color:
        return discord.Color(EMBED_DEFAULT_COLOR)
    
    def _get_context(self, context_size: int) -> List[dict]:
        """Récupère le contexte du chatbot (derniers messages dans la limite de la taille du contexte)."""
        encoding = tiktoken.encoding_for_model(AI_MODEL)
        init_size = len(encoding.encode(self.system_prompt))
        
        last_message = self._context[-1]
        last_message_size = len(encoding.encode(last_message['content']))
        tokens = init_size + last_message_size
        context = []
        for message in reversed(self._context[:-1]):
            if tokens + len(encoding.encode(message['content'])) > context_size:
                break
            context.append(message)
            tokens += len(encoding.encode(message['content']))
        return [{'role': 'system', 'content': self.system_prompt}] + context[::-1] + [last_message]
    
    @property
    def context(self) -> List[dict]:
        """Récupère le contexte du chatbot (derniers messages dans la limite de la taille du contexte)."""
        return self._get_context(self.context_size)
    
    def _get_embed(self) -> discord.Embed:
        em = discord.Embed(title=f"*{str(self)}*", description=f'*{self.description}*', color=EMBED_DEFAULT_COLOR, timestamp=datetime.fromtimestamp(self.created_at))
        em.set_thumbnail(url=self.avatar_url)
        em.add_field(name="Prompt d'initialisation", value=f'```{pretty.troncate_text(self.system_prompt, 1000)}```', inline=False)
        em.add_field(name="Température", value=f'`{self.temperature}`')
        em.add_field(name="Taille du contexte", value=f'`{self.context_size} tokens`')
        if self._debug:
            em.add_field(name="Mode debug", value="Activé")
        return em
    
    @property
    def embed(self) -> discord.Embed:
        """Récupère un embed représentant le chatbot temporaire."""
        return self._get_embed()
    
class PassiveChatbot(TempChatbot):
    """Représente un chatbot temporaire passif qui tourne constamment en fond sur un salon"""
    def __init__(self, cog: 'Chatter', system_prompt: str, temperature: float, context_size: int, *, author_id: int | None = None, debug: bool = False):
        super().__init__(cog, system_prompt, temperature, context_size, author_id=author_id, debug=debug)
        self.description = f'Chatbot temporaire en mode passif\n**⚠ Ce chatbot lit activement tous les messages postés sur ce salon et les utilise comme contexte de réponse.**'

    def __repr__(self) -> str:
        return f'<PassiveChatbot system_prompt={self.system_prompt}>'
    
    def __str__(self) -> str:
        return f'TP.{self.name.title()} [DEBUG]' if self._debug else f'TP.{self.name.title()}'
                
class AIChatSession:
    """Représente une session de chat avec un chatbot IA."""
    def __init__(self, channel: discord.TextChannel | discord.Thread, chatbot: CustomChatbot | TempChatbot, *, auto_answer: bool = False):
        self.channel = channel
        self.chatbot = chatbot
        self.auto_answer = auto_answer # Si True, le chatbot répond automatiquement aux messages des utilisateurs sans mention (seulement sur les threads)
        
        self._cog = chatbot._cog
        
    def __repr__(self) -> str:
        return f'<AIChatSession channel={self.channel} chatbot={self.chatbot}>'
    
    def __str__(self) -> str:
        return f'Chatbot {self.chatbot} sur {self.channel}'
    
    def check_blacklist(self, object_id: int) -> bool:
        """Vérifie si un ID est blacklisté."""
        return object_id in self.chatbot.blacklist
    
    # Utilisation de l'API ----------------------------------------------------------------------------------------------------
    
    async def _get_completion(self, content: str, username: str) -> Dict[str, str] | None:
        """Envoie le prompt demandé à ChatGPT ainsi que le contexte du chatbot."""
        username = unidecode.unidecode(username)
        username = ''.join([c for c in username if c.isalnum()]).rstrip()
        
        if isinstance(self.chatbot, CustomChatbot):
            self.chatbot.logs.add_message(time.time(), 'user', content, username)
        else:
            self.chatbot._context.append({'role': 'user', 'content': content, 'name': username})
            
        messages = self.chatbot.context
        payload = {
            'model': AI_MODEL,
            'temperature': self.chatbot.temperature,
            'max_tokens': MAX_TOKENS,
            'messages': messages
        }
        try:
            response = await openai.ChatCompletion.acreate(**payload)
        except Exception as e:
            logger.error(f'Erreur lors de la requête à l\'API OpenAI : {e}')
            await self.channel.send(f"**Erreur dans la requête à l'API OpenAI** · `{e}`", delete_after=30)
            raise commands.CommandError('Une erreur est survenue lors de la requête à l\'API OpenAI. Veuillez réessayer plus tard.')

        if not response or not response['choices']:
            raise commands.CommandError('Une erreur est survenue lors de la requête à l\'API OpenAI. Veuillez réessayer plus tard.')

        text = response['choices'][0]['message']['content']
        is_finished = response['choices'][0]['finish_reason'] == 'stop'
        tokens = response['usage']['total_tokens']
        
        if len(text) > 2000:
            text = text[:1997] + '...'
            is_finished = False
        
        timestamp = time.time()
        if isinstance(self.chatbot, CustomChatbot):
            self.chatbot.logs.add_message(timestamp, 'assistant', text, None)
            self.chatbot.stats.messages += 1
            self.chatbot.stats.tokens += tokens
            self.chatbot.stats.update_last_use()
        else:
            self.chatbot._context.append({'role': 'assistant', 'content': text})

        return {
            'content': text,
            'tokens_used': tokens,
            'stop': is_finished
        } # type: ignore

    async def handle_message(self, message: discord.Message, *, send_continue: bool = False, override_mention: bool = False, custom_content: str | None = None) -> bool:
        """Gère un message envoyé sur le salon."""
        botuser = self._cog.bot.user
        channel = message.channel
        if not botuser:
            return False
        
        content = message.content if custom_content is None else custom_content
            
        if not content:
            return False

        if content.startswith('w!'): # Ignore les commandes
            return False
        
        comp = None
        
        if send_continue: # Si le chatbot n'a pas fini de parler
            content = 'Suite'
            async with channel.typing():
                comp = await self._get_completion(content, message.author.display_name)
        elif botuser.mentioned_in(message) or override_mention:
            async with channel.typing():
                comp = await self._get_completion(content, message.author.display_name)
        elif isinstance(self.chatbot, PassiveChatbot): # Si le chatbot est en mode passif on enregistre tous les messages d'utilisateurs qui ne mentionnent pas le bot
            username = unidecode.unidecode(message.author.display_name)
            username = ''.join([c for c in username if c.isalnum()]).rstrip()
            self.chatbot._context.append({'role': 'user', 'content': content, 'name': username})
            
        if comp: # Si le chatbot a répondu
            text = comp['content']
            tokens_used = comp['tokens_used']
            is_finished = comp['stop']
            
            if self.chatbot._debug:
                text = f'```{text}```'
                
                encoder = tiktoken.encoding_for_model(AI_MODEL)
                context_text = '\n'.join([m['content'] for m in self.chatbot.context])
                current_context_size = len(encoder.encode(context_text))
                text += f'**Tokens** : {tokens_used}\n**Contexte** : {len(self.chatbot.context)} messages ({current_context_size}/{self.chatbot.context_size} tokens)'
            
            # Si le bot a fini de parler on envoie juste la réponse
            if is_finished:
                await message.reply(text, mention_author=False, suppress_embeds=True, allowed_mentions=discord.AllowedMentions(users=False, roles=False, everyone=False, replied_user=True))
                return True
            
            # Sinon on envoie la réponse avec un bouton pour continuer
            view = ContinueButtonView(author=message.author)
            resp = await message.reply(text, view=view, mention_author=False, suppress_embeds=True, allowed_mentions=discord.AllowedMentions(users=False, roles=False, everyone=False, replied_user=True))
            await view.wait()
            
            if view.value is True:
                # Si l'utilisateur a cliqué sur le bouton, on continue
                await resp.edit(view=None)
                await self.handle_message(message, send_continue=True)
            else:
                await resp.edit(view=None)
            return True
        return False

class Chatter(commands.Cog):
    """Parlez avec le bot et customisez son comportement !"""
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.data = dataio.get_cog_data(self)
        openai.api_key = self.bot.config['OPENAI_APIKEY'] #type: ignore
        
        self.use_msg_as_prompt = app_commands.ContextMenu(
            name="Demander à l'IA",
            callback=self.ctx_use_as_prompt,
        )
        self.bot.tree.add_command(self.use_msg_as_prompt)
        
        self.sessions = {}
        
    @commands.Cog.listener()
    async def on_ready(self):
        self.__init_global()
        self.__init_guilds()
        
    @commands.Cog.listener()
    async def on_guild_join(self, guild: discord.Guild):
        self.__init_guilds([guild])
        
    def __init_guilds(self, guilds: List[discord.Guild] | None = None) -> None:
        guilds = guilds or list(self.bot.guilds)
        for guild in guilds:
            # Profils d'IA
            profiles = """CREATE TABLE IF NOT EXISTS profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                description TEXT,
                avatar_url TEXT DEFAULT NULL,
                system_prompt TEXT,
                temperature REAL DEFAULT 0.8,
                context_size INTEGER DEFAULT 1024,
                features TEXT DEFAULT '',
                blacklist TEXT DEFAULT '',
                author_id INTEGER,
                created_at REAL
                )"""
            self.data.execute(guild, profiles)
        
            # Statistiques
            stats = """CREATE TABLE IF NOT EXISTS stats (
                profile_id INTEGER PRIMARY KEY,
                uses INTEGER,
                messages INTEGER,
                tokens INTEGER,
                last_use REAL,
                FOREIGN KEY(profile_id) REFERENCES profiles(id)
                )"""
            self.data.execute(guild, stats)
            
            # Messages
            messages = """CREATE TABLE IF NOT EXISTS messages (
                timestamp REAL PRIMARY KEY,
                profile_id INTEGER,
                session_id INTEGER,
                role TEXT,
                content TEXT,
                username TEXT,
                FOREIGN KEY(profile_id) REFERENCES profiles(id)
                )"""
            self.data.execute(guild, messages)
            
    def __init_global(self):
        # Crédits des serveurs
        query = """CREATE TABLE IF NOT EXISTS credits (
            guild_id INTEGER PRIMARY KEY,
            credits INTEGER DEFAULT 0,
            last_update REAL DEFAULT 0,
            renew BOOLEAN DEFAULT 1 CHECK (renew IN (0, 1))
            )"""
        self.data.execute('global', query)
        
    # Utils ------------------------------------------------------------------------------------------------------
    
    def _get_default_name(self, system_prompt: str) -> str:
        """Détermine un nom par défaut pour un prompt donné."""
        r = random.Random(system_prompt)
        names = [
            'GladOS',
            'HAL 9000',
            'TARS',
            'C3PO',
            'MAGI',
            'Skynet',
            'Cortana',
            'Jarvis',
            'Mother',
            'NERON',
            'Bender',
            'Ava',
            'T-800'
        ]
        return r.choice(names)
    
    def _get_default_avatar(self, system_prompt: str) -> str:
        """Détermine un avatar par défaut pour un prompt donné."""
        # r = random.Random(system_prompt)
        # n = r.randint(0, 4)
        # default_avatars = [
        #     'https://i.imgur.com/HZCpxN9.png',
        #     'https://i.imgur.com/skKSIWz.png',
        #     'https://i.imgur.com/pXMIUTz.png',
        #     'https://i.imgur.com/nEKs6vq.png',
        #     'https://i.imgur.com/eBhrPU2.png'
        # ]
        # return default_avatars[n]
        if not self.bot.user:
            return ''
        return self.bot.user.display_avatar.url

    
    # Chatbots customs -----------------------------------------------------------------------------------------------
    
    def get_chatbot(self, guild: discord.Guild, profile_id: int, *, resume: bool = True, debug: bool = False) -> CustomChatbot:
        """Récupère un profil d'IA."""
        # On met à jour la couleur du chatbot
        return CustomChatbot(self, guild, profile_id, resume=resume, debug=debug)
    
    def get_chatbot_by_name(self, guild: discord.Guild, name: str) -> CustomChatbot:
        """Récupère un profil d'IA par son nom."""
        query = """SELECT id FROM profiles WHERE name = ?"""
        data = self.data.fetchone(guild, query, (name,))
        if not data:
            raise ValueError(f'Profil {name} introuvable.')
        return self.get_chatbot(guild, data['id'])
    
    def get_chatbots(self, guild: discord.Guild) -> List[CustomChatbot]:
        """Récupère tous les profils d'IA d'une guilde."""
        query = """SELECT id FROM profiles"""
        data = self.data.fetchall(guild, query)
        return [self.get_chatbot(guild, profile['id']) for profile in data]
    
    def get_chatbots_by_author(self, guild: discord.Guild, author_id: int) -> List[CustomChatbot]:
        """Récupère tous les profils d'IA créés par un auteur."""
        query = """SELECT id FROM profiles WHERE author_id = ?"""
        data = self.data.fetchall(guild, query, (author_id,))
        return [self.get_chatbot(guild, profile['id']) for profile in data]
    
    # Sessions ---------------------------------------------------------------------------------------------------
    
    def get_session(self, channel: discord.TextChannel | discord.Thread) -> AIChatSession:
        """Récupère une session de chat."""
        return self.sessions[channel.id]
    
    def set_session(self, channel: discord.TextChannel | discord.Thread, chatbot: CustomChatbot | TempChatbot, *, auto_answer: bool = False) -> None:
        """Définit une session de chat."""
        self.sessions[channel.id] = AIChatSession(channel, chatbot, auto_answer=auto_answer)
        
    async def ask_replace_session(self, interaction: discord.Interaction, channel: discord.TextChannel | discord.Thread, chatbot: CustomChatbot | TempChatbot) -> bool:
        """Demande à l'utilisateur s'il veut remplacer la session de chat actuelle."""
        current_chatbot = self.get_session(channel).chatbot
        if isinstance(current_chatbot, TempChatbot):
            return True # On ne demande pas pour les chatbots temporaires
        
        if current_chatbot == chatbot:
            return True # On ne demande pas si c'est le même chatbot
        
        view = ConfirmationView()
        text = f"Le chatbot **{current_chatbot}** est déjà en cours d'utilisation sur ce salon. Voulez-vous le remplacer par **{chatbot}** ?"
        await interaction.followup.send(text, view=view, ephemeral=True)
        await view.wait()
        await interaction.delete_original_response()
        if view.value is None:
            return False
        return view.value
    
    # COMMANDES ==================================================================================================
    
    # Chat ---------------------------------------------------------------------------------------------------
    chat_group = app_commands.Group(name='chat', description="Commandes permettant de discuter avec l'IA", guild_only=True)
    
    @chat_group.command(name='temp')
    @app_commands.rename(system_prompt='initialisation', temperature='température', context_size='taille_contexte')
    async def _chat_temp(self, interaction: discord.Interaction, system_prompt: str, temperature: app_commands.Range[float, 0.1, 2.0] = 0.8, context_size: app_commands.Range[int, 1, MAX_CONTEXT_SIZE] = DEFAULT_CONTEXT_SIZE, debug: bool = False):
        """Créer un chatbot temporaire pour discuter sur le salon courant
        
        :param system_prompt: Prompt d'initialisation de l'IA
        :param temperature: Température de l'IA (entre 0.1 et 2.0)
        :param context_size: Taille du contexte de l'IA en tokens (par défaut 1024)
        :param debug: Si True, affiche des informations supplémentaires à la fin des messages
        """
        channel = interaction.channel
        if not isinstance(channel, (discord.TextChannel, discord.Thread)):
            raise commands.BadArgument('Cette commande ne peut être utilisée que sur un salon textuel ou un thread.')

        await interaction.response.defer()
        chatbot = TempChatbot(self, system_prompt, temperature, context_size, author_id=interaction.user.id, debug=debug)
        if channel.id in self.sessions:
            if not await self.ask_replace_session(interaction, channel, TempChatbot(self, system_prompt, temperature, context_size, author_id=interaction.user.id)):
                return await interaction.followup.send("Vous avez annulé la création du chatbot temporaire.", ephemeral=True)
            
        self.set_session(channel, chatbot)
        await interaction.followup.send(f"Le chatbot temporaire **{chatbot}** a été attaché à ce salon.", embed=chatbot.embed)
        
    @chat_group.command(name='passive')
    @app_commands.rename(system_prompt='initialisation', temperature='température', context_size='taille_contexte')
    async def _chat_passive(self, interaction: discord.Interaction, system_prompt: str, temperature: app_commands.Range[float, 0.1, 2.0] = 0.8, context_size: app_commands.Range[int, 1, MAX_CONTEXT_SIZE] = DEFAULT_CONTEXT_SIZE, debug: bool = False):
        """Créer un chatbot temporaire passif qui lit tous les messages du salon courant
        
        :param system_prompt: Prompt d'initialisation de l'IA
        :param temperature: Température de l'IA (entre 0.1 et 2.0)
        :param context_size: Taille du contexte de l'IA en tokens (par défaut 1024)
        :param debug: Si True, affiche des informations supplémentaires à la fin des messages
        """
        channel = interaction.channel
        if not isinstance(channel, (discord.TextChannel, discord.Thread)):
            raise commands.BadArgument('Cette commande ne peut être utilisée que sur un salon textuel ou un thread.')
        
        await interaction.response.defer()
        chatbot = PassiveChatbot(self, system_prompt, temperature, context_size, author_id=interaction.user.id, debug=debug)
        if channel.id in self.sessions:
            if not await self.ask_replace_session(interaction, channel, chatbot):
                return await interaction.followup.send("Vous avez annulé la création du chatbot temporaire passif.", ephemeral=True)
        
        self.set_session(channel, chatbot)
        await interaction.followup.send(f"Le chatbot temporaire passif **{chatbot}** a été attaché à ce salon.", embed=chatbot.embed)
        
    @chat_group.command(name='wipe')
    async def _chat_wipe(self, interaction: discord.Interaction):
        """Efface la mémoire (contexte) du chatbot attaché au salon et démarre une nouvelle session de chat"""
        channel = interaction.channel
        if not isinstance(channel, (discord.TextChannel, discord.Thread)):
            raise commands.BadArgument('Cette commande ne peut être utilisée que sur un salon textuel ou un thread.')
        
        if channel.id not in self.sessions:
            return await interaction.response.send_message("**Erreur** · Il n'y a pas de chatbot attaché à ce salon.", ephemeral=True)
        
        chatbot = self.get_session(channel).chatbot
        if isinstance(chatbot, CustomChatbot):
            newchatbot = CustomChatbot(self, channel.guild, chatbot.id, resume=False)
            self.set_session(channel, newchatbot)
        else:
            chatbot._context = []
        await interaction.response.send_message(f"**Succès** · Le contexte du chatbot **{chatbot}** a été effacé.\nCelui-ci ne se souviendra plus de vos messages précédents.")
        
    @chat_group.command(name='remove')
    async def _chat_remove(self, interaction: discord.Interaction):
        """Retire tout chatbot attaché au salon courant"""
        channel = interaction.channel
        if not isinstance(channel, (discord.TextChannel, discord.Thread)):
            raise commands.BadArgument('Cette commande ne peut être utilisée que sur un salon textuel ou un thread.')
        
        if channel.id not in self.sessions:
            return await interaction.response.send_message("**Erreur** · Il n'y a pas de chatbot attaché à ce salon.", ephemeral=True)
        
        self.sessions.pop(channel.id)
        await interaction.response.send_message("**Succès** · Le chatbot actuel a été détaché de ce salon.")
        
    @chat_group.command(name='load')
    @app_commands.rename(chatbot_id='chatbot', resume='reprendre')
    async def _chat_load(self, interaction: discord.Interaction, chatbot_id: int, resume: bool = True, debug: bool = False):
        """Charge un chatbot personnalisé sur le salon courant

        :param chatbot_id: Identifiant unique du chatbot à charger
        :param resume: Si True, reprend la discussion à partir de la dernière session (par défaut)
        :param debug: Si True, affiche des informations supplémentaires à la fin des messages
        """
        channel = interaction.channel
        if not isinstance(channel, (discord.TextChannel, discord.Thread)):
            raise commands.BadArgument('Cette commande ne peut être utilisée que sur un salon textuel ou un thread.')
        
        await interaction.response.defer()
        chatbot = self.get_chatbot(channel.guild, chatbot_id, resume=resume, debug=debug)
        if channel.id in self.sessions:
            if not await self.ask_replace_session(interaction, channel, chatbot):
                return await interaction.followup.send("Vous avez annulé le chargement du chatbot.", ephemeral=True)
        
        chatbot.stats.uses += 1
        self.set_session(channel, chatbot)
        await interaction.followup.send(f"Le chatbot **{chatbot}** a été chargé sur ce salon.", embed=chatbot.embed)
        await asyncio.sleep(60)
        await interaction.edit_original_response(embed=None)
        
    @chat_group.command(name='current')
    async def _chat_current(self, interaction: discord.Interaction):
        """Affiche les détails sur le chatbot actuel du salon"""
        channel = interaction.channel
        if not isinstance(channel, (discord.TextChannel, discord.Thread)):
            raise commands.BadArgument('Cette commande ne peut être utilisée que sur un salon textuel ou un thread.')
        
        if channel.id not in self.sessions:
            return await interaction.response.send_message("**Aucun** · Il n'y a pas de chatbot actuellement attaché à ce salon.")
        
        chatbot = self.get_session(channel).chatbot
        await interaction.response.send_message(embed=chatbot.embed)
        
    async def ctx_use_as_prompt(self, interaction: discord.Interaction, message: discord.Message):
        """Menu contextuel permettant d'utiliser le message visé comme prompt pour le chatbot du salon"""
        channel = interaction.channel
        if not isinstance(channel, (discord.TextChannel, discord.Thread)):
            return await interaction.response.send_message("**Impossible** · Cette fonctionnalité n'est disponible que sur les salons textuels et les threads.", ephemeral=True)

        if channel.id not in self.sessions:
            return await interaction.response.send_message("**Aucun chatbot n'est chargé** · Il n'y a pas de chatbot actuellement attaché à ce salon.", ephemeral=True)
        
        session = self.get_session(channel)
        
        ctx_user = interaction.user
        if session.check_blacklist(ctx_user.id) or session.check_blacklist(message.channel.id):
            return await interaction.response.send_message("**Impossible** · Vous ne pouvez pas utiliser ce message comme prompt car vous avez été blacklisté par le chatbot ou ce salon n'est pas autorisé à utiliser ce chatbot.", ephemeral=True)
        
        modal = AddAskContextModal(session.chatbot, message, timeout=120)
        await interaction.response.send_modal(modal)
        
        cancel_modal = await modal.wait()
        if not cancel_modal and modal.value:
            new_content = modal.value + '\n' + message.content
            return await session.handle_message(message, override_mention=True, custom_content=new_content)
        
        # await interaction.response.send_message(f"**Demande à l'IA** · Le message de *{message.author.display_name}* a été envoyé au chatbot **{session.chatbot}**", delete_after=5)
        await session.handle_message(message, override_mention=True)
        
    # TODO: Ajouter une fois que les crédits seront implémentés et obligatoires
    
    # @chat_group.command(name='auto')
    # @app_commands.rename(auto_answer='auto')
    # async def _chat_auto(self, interaction: discord.Interaction, auto_answer: bool):
    #     """Activer/Désactiver le mode de réponse automatique du chatbot (seulement sur les threads)
        
    #     :param auto_answer: Activer/désactiver le mode de réponse automatique
    #     """
    #     channel = interaction.channel
    #     if not isinstance(channel, discord.Thread):
    #         return await interaction.response.send_message("**Impossible** · Cette fonctionnalité n'est disponible que sur les threads.")
        
    #     if channel.id not in self.sessions:
    #         return await interaction.response.send_message("**Erreur** · Il n'y a pas de chatbot actuellement attaché à ce salon.")
        
    #     session = self.get_session(channel)
    #     session.auto_answer = auto_answer
    #     await interaction.response.send_message(f"**Succès** · Le mode de réponse automatique du chatbot **{session.chatbot}** a été **{'activé' if auto_answer else 'désactivé'}**.")
        
    
    # Profils ---------------------------------------------------------------------------------------------------
    chatbot_group = app_commands.Group(name='chatbot', description="Commandes de gestion des Chatbot customisés", guild_only=True)

    @chatbot_group.command(name='setup')
    @app_commands.rename(name='nom', system_prompt='initialisation', temperature='température', avatar_url='avatar', context_size='taille_contexte')
    async def _chatbot_setup(self, 
                             interaction: discord.Interaction, 
                             name: str, 
                             description: str, 
                             system_prompt: str,
                             avatar_url: str = '',
                             temperature: app_commands.Range[float, 0.1, 2.0] = 0.8,
                             context_size: app_commands.Range[int, 1, MAX_CONTEXT_SIZE] = DEFAULT_CONTEXT_SIZE):
        """Créer ou modifier un chatbot personnalisé
        
        :param name: Nom du chatbot
        :param description: Description du chatbot
        :param system_prompt: Prompt d'initialisation de l'IA
        :param avatar_url: URL de l'avatar du chatbot
        :param temperature: Température de l'IA (entre 0.1 et 2.0)
        :param context_size: Taille du contexte (en tokens)
        """
        guild = interaction.guild
        if not isinstance(guild, discord.Guild):
            raise commands.BadArgument('Cette commande ne peut être utilisée que sur un serveur.')
        
        if len(name) > 32:
            return await interaction.response.send_message("**Erreur** · Le nom du chatbot ne peut pas dépasser 32 caractères.", ephemeral=True)
        if len(description) > 200:
            return await interaction.response.send_message("**Erreur** · La description du chatbot ne peut pas dépasser 200 caractères.", ephemeral=True)
        
        await interaction.response.defer()
        edit = False
        # Vérifier un conflit de nom
        chatbots = self.get_chatbots(guild)
        if any(c.name.lower() == name.lower() for c in chatbots):
            confview = ConfirmationView()
            msg = await interaction.followup.send(f"**Conflit de nom** · Un chatbot nommé **{name}** existe déjà sur ce serveur.\n**Voulez-vous l'écraser ?**", ephemeral=True, view=confview, wait=True)
            await confview.wait()
            await msg.edit(content=f"**Conflit de nom** · Le chatbot **{name}** sera écrasé.", view=None)
            if confview.value is None or not confview.value:
                return await interaction.followup.send("Vous avez annulé la modification du chatbot.", ephemeral=True)
            edit = True

        # Vérifier la taille du prompt d'initialisation
        sysprompt_tokens = len(tiktoken.encoding_for_model(AI_MODEL).encode(system_prompt))
        if sysprompt_tokens >= MAX_CONTEXT_SIZE:
            return await interaction.followup.send(f"**Erreur** · Le prompt d'initialisation est trop long ({MAX_CONTEXT_SIZE} tokens maximum).", ephemeral=True)
        
        if context_size > MAX_CONTEXT_SIZE / 2:
            confview = ConfirmationView()
            msg = await interaction.followup.send(f"**Consommation de crédit potentiellement élevé** · Vous allez créer un chatbot dont le nombre de tokens alloués à la mémoire (contexte) est élevé (>{MAX_CONTEXT_SIZE / 2}) ce qui peut coûter cher en crédits d'API lors de son utilisation.\nIl est déconseillé d'utiliser une taille de contexte aussi élevée à moins que le chatbot ait besoin de garder en mémoire plusieurs dizaines de messages afin de répondre à vos attentes.\n**Vouls-vous continuer sa création ?**", ephemeral=True, view=confview, wait=True)
            await confview.wait()
            await msg.edit(content=f"**Consommation de crédit potentiellement élevé** · La taille du contexte spécifiée sera utilisée.", view=None)
            if confview.value is None or not confview.value:
                return await interaction.followup.send("Vous avez annulé la création/modification du chatbot.", ephemeral=True)
        
        if sysprompt_tokens >= round(context_size / 2):
            confview = ConfirmationView()
            msg = await interaction.followup.send("**Prompt de grande taille** · Le prompt d'initialisation représente plus de la moitié du contexte et pourrait grandement restreindre les capacités du Chatbot à garder en mémoire les messages précédents lors de vos interactions.\n**Continuer ?**", ephemeral=True, view=confview, wait=True)
            await confview.wait()
            await msg.edit(content=f"**Prompt de grande taille** · Le prompt spécifié sera utilisé.", view=None)
            if confview.value is None or not confview.value:
                return await interaction.followup.send("Vous avez annulé la création/modification du chatbot.", ephemeral=True)
        
        # Créer le chatbot
        if len(chatbots) >= 20:
            return await interaction.followup.send("**Erreur** · Vous avez atteint la limite de 20 chatbots par serveur.", ephemeral=True)
        
        query = """INSERT OR REPLACE INTO profiles VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        self.data.execute(guild, query, (
            name,
            description,
            avatar_url,
            system_prompt,
            temperature,
            context_size,
            '',
            '',
            interaction.user.id,
            time.time()
        ))
        chatbot = self.get_chatbot_by_name(guild, name)
        await interaction.followup.send(f"Le chatbot **{chatbot}** a été {'modifié' if edit else 'créé'} avec succès.", embed=chatbot.embed)
        
    @chatbot_group.command(name='delete')
    @app_commands.rename(chatbot_id='chatbot')
    async def _chatbot_delete(self, interaction: discord.Interaction, chatbot_id: int):
        """Supprime un chatbot personnalisé

        :param chatbot_id: Identifiant unique du chatbot à supprimer
        """
        guild = interaction.guild
        if not isinstance(guild, discord.Guild):
            raise commands.BadArgument('Cette commande ne peut être utilisée que sur un serveur.')
        
        chatbot = self.get_chatbot(guild, chatbot_id)
        if not chatbot:
            return await interaction.response.send_message("**Erreur** · Ce chatbot n'existe pas.", ephemeral=True)
        
        confview = ConfirmationView()
        await interaction.response.send_message(f"**Effacer un chatbot** · Vous êtes sur le point de supprimer le chatbot **{chatbot}**. Cette action est __irréversible__.\n**Voulez-vous continuer ?**", ephemeral=True, view=confview)
        await confview.wait()
        await interaction.delete_original_response()
        if confview.value is None or not confview.value:
            return await interaction.response.send_message("Vous avez annulé la suppression du chatbot.", ephemeral=True)
        
        # Supprimer le chatbot
        query = """DELETE FROM profiles WHERE id = ?"""
        self.data.execute(guild, query, (chatbot.id,))
        
        # Supprimer les messages
        query = """DELETE FROM messages WHERE profile_id = ?"""
        self.data.execute(guild, query, (chatbot.id,))
        
        # Supprimer les stats
        query = """DELETE FROM stats WHERE profile_id = ?"""
        self.data.execute(guild, query, (chatbot.id,))
        await interaction.response.send_message(f"Le chatbot **{chatbot}** a été supprimé avec succès.")
        
    @chatbot_group.command(name='list')
    async def _chatbot_list(self, interaction: discord.Interaction):
        """Liste les chatbots personnalisés du serveur et affiche leurs informations"""
        guild = interaction.guild
        if not isinstance(guild, discord.Guild):
            raise commands.BadArgument('Cette commande ne peut être utilisée que sur un serveur.')
        
        await interaction.response.defer()
        chatbots = self.get_chatbots(guild)
        if not chatbots:
            return await interaction.followup.send("**Aucun chatbot** · Il n'y a aucun chatbot personnalisé sur ce serveur.")
        
        menu = ChatbotList(chatbots, user=interaction.user)
        await menu.start(interaction)
        
    # Dev ---------------------------------------------------------------------------------------------------
    modchat_group = app_commands.Group(name='modchat', description="Commandes avancées de gestion des Chatbots", guild_only=True, default_permissions=discord.Permissions(manage_messages=True))
        
    @modchat_group.command(name='edit')
    @app_commands.rename(chatbot_id='chatbot', key='clé', value='valeur')
    async def _modchat_edit(self, interaction: discord.Interaction, chatbot_id: int, key: str, value: str):
        """Modifie un paramètre d'un chatbot personnalisé sans avoir à le refaire entièrement

        :param chatbot_id: Identifiant unique du chatbot
        :param key: Clé du paramètre à modifier
        :param value: Valeur du paramètre à modifier
        """
        guild = interaction.guild
        if not isinstance(guild, discord.Guild):
            raise commands.BadArgument('Cette commande ne peut être utilisée que sur un serveur.')
        
        chatbot = self.get_chatbot(guild, chatbot_id)
        if not chatbot:
            return await interaction.response.send_message("**Erreur** · Ce chatbot n'existe pas.", ephemeral=True)
        
        if key not in chatbot.__dict__:
            return await interaction.response.send_message(f"**Erreur** · Le paramètre `{key}` n'existe pas.", ephemeral=True)
        
        if key == 'context_size':
            val = int(value)
            if val > MAX_CONTEXT_SIZE:
                return await interaction.response.send_message(f"**Erreur** · La taille du contexte ne peut pas dépasser {MAX_CONTEXT_SIZE} tokens.", ephemeral=True)
        elif key == 'temperature':
            try:
                val = float(value)
            except:
                return await interaction.response.send_message(f"**Erreur** · La température doit être un nombre compris entre 0.1 et 2.0.", ephemeral=True)
            if val < 0.1 or val > 2.0:
                return await interaction.response.send_message(f"**Erreur** · La température doit être comprise entre 0.1 et 2.0.", ephemeral=True)
        
        chatbot.__dict__[key] = value
        chatbot.save()
        await interaction.response.send_message(f"**Modification effectuée** · Le paramètre `{key}` a été modifié pour `{value}`.", embed=chatbot.embed, ephemeral=True)
        
    @modchat_group.command(name='resolve')
    async def _modchat_resolve(self, interaction: discord.Interaction):
        """Règle tous les conflits de noms entre les chatbots du serveur"""
        guild = interaction.guild
        if not isinstance(guild, discord.Guild):
            raise commands.BadArgument('Cette commande ne peut être utilisée que sur un serveur.')
        
        chatbots = self.get_chatbots(guild)
        conflicts : List[CustomChatbot] = []
        for chatbot in chatbots:
            if any(c.name.lower() == chatbot.name.lower() for c in chatbots if c.id != chatbot.id):
                conflicts.append(chatbot)
        
        if not conflicts:
            return await interaction.response.send_message("**Aucun conflit** · Il n'y a aucun conflit de nom entre les chatbots de ce serveur.")
        
        em = discord.Embed(title="Conflits de noms", description="Il y a des conflits de noms entre les chatbots de ce serveur. **Voulez-vous supprimer les plus anciens ?**", color=discord.Color.red())
        em.add_field(name="Chatbots", value='\n'.join([f"**{c}** ({c.id})" for c in conflicts]))
        
        await interaction.response.defer()
        confview = ConfirmationView()
        await interaction.followup.send(embed=em, view=confview, ephemeral=True)
        await confview.wait()
        await interaction.delete_original_response()
        if confview.value is None or not confview.value:
            return await interaction.followup.send("Vous avez annulé la suppression des chatbots.", ephemeral=True)

        # Supprimer les chatbots sauf le plus récent
        conflicts = sorted(conflicts, key=lambda c: c.created_at)
        
        for chatbot in conflicts[:-1]:
            query = """DELETE FROM profiles WHERE id = ?"""
            self.data.execute(guild, query, (chatbot.id,))
            
            # Supprimer les messages
            query = """DELETE FROM messages WHERE profile_id = ?"""
            self.data.execute(guild, query, (chatbot.id,))
            
            # Supprimer les stats
            query = """DELETE FROM stats WHERE profile_id = ?"""
            self.data.execute(guild, query, (chatbot.id,))
        
        await interaction.followup.send(f"**Succès** · {len(conflicts) - 1} chatbots ont été supprimés.", ephemeral=True)
        
    @modchat_group.command(name='cancellast')
    async def _modchat_cancel_last(self, interaction: discord.Interaction):
        """Permet de supprimer du contexte le dernier message envoyé ou reçu par le chatbot sur le salon courant"""
        guild = interaction.guild
        if not isinstance(guild, discord.Guild):
            raise commands.BadArgument('Cette commande ne peut être utilisée que sur un serveur.')
        
        channel = interaction.channel
        if not isinstance(channel, (discord.TextChannel, discord.Thread)):
            raise commands.BadArgument('Cette commande ne peut être utilisée que sur un salon textuel ou un thread.')
        
        if not channel.id in self.sessions:
            return await interaction.response.send_message("**Erreur** · Il n'y a pas de chatbot attaché à ce salon.", ephemeral=True)
        
        chatbot = self.get_session(channel).chatbot
        if not chatbot.context:
            return await interaction.response.send_message("**Erreur** · Le contexte est vide.", ephemeral=True)
        
        last_message = chatbot.context[-1]
        # On affiche le message et on demande confirmation
        em = discord.Embed(title="Supprimer le dernier message du contexte", description="Êtes-vous sûr de vouloir supprimer le dernier message du contexte ? Cette action est __irréversible__.", color=discord.Color.red())
        em.add_field(name="Message", value=f"```{pretty.troncate_text(last_message['content'], 500)}```")
        await interaction.response.defer()
        confview = ConfirmationView()
        await interaction.followup.send(embed=em, view=confview, ephemeral=True)
        await confview.wait()
        await interaction.delete_original_response()
        if confview.value is None or not confview.value:
            return await interaction.followup.send("Vous avez annulé la suppression du message.", ephemeral=True)
        
        if isinstance(chatbot, CustomChatbot):
            chatbot.logs.remove_last_message()
        else:
            chatbot._context.pop()
        
        await interaction.delete_original_response()
        await interaction.followup.send(f"**Succès** · Le dernier message du chatbot **{chatbot}** a été supprimé du contexte.", ephemeral=True)
        
    # Blacklists ---------------------------------------------------------------------------------------------------
    blacklists_group = app_commands.Group(name='blacklist', description="Gestion des blacklists des Chatbots", guild_only=True, parent=chatbot_group, default_permissions=discord.Permissions(manage_messages=True))
    
    @blacklists_group.command(name='add')
    @app_commands.rename(chatbot_id='chatbot')
    async def _blacklist_add(self, interaction: discord.Interaction, chatbot_id: int, user: discord.User | None = None, channel: discord.TextChannel | discord.Thread | None = None):
        """Ajoute un utilisateur ou un salon à la blacklist d'un chatbot

        :param chatbot_id: Identifiant unique du chatbot
        :param user: Utilisateur à ajouter à la blacklist
        :param channel: Salon à ajouter à la blacklist
        """
        guild = interaction.guild
        if not isinstance(guild, discord.Guild):
            raise commands.BadArgument('Cette commande ne peut être utilisée que sur un serveur.')
        
        author = interaction.user
        comchan = interaction.channel
        if not isinstance(channel, (discord.TextChannel, discord.Thread)) or not comchan:
            raise commands.BadArgument('Cette commande ne peut être utilisée que sur un salon textuel ou un thread.')                     
        if not comchan.permissions_for(author).manage_messages: #type: ignore
            return await interaction.response.send_message("**Permissions insuffisantes** · Vous devez avoir la permission de gérer les messages pour utiliser cette commande.", ephemeral=True)
        
        chatbot = self.get_chatbot(guild, chatbot_id)
        if not chatbot:
            return await interaction.response.send_message("**Erreur** · Ce chatbot n'existe pas.", ephemeral=True)
        
        if not user and not channel:
            return await interaction.response.send_message("**Erreur** · Vous devez spécifier un utilisateur ou un salon à ajouter à la blacklist.", ephemeral=True)
        
        if user:
            if user.id in chatbot.blacklist:
                return await interaction.response.send_message("**Erreur** · Cet utilisateur est déjà dans la blacklist de ce chatbot.", ephemeral=True)
            chatbot.blacklist.append(user.id)
        elif channel:
            if channel.id in chatbot.blacklist:
                return await interaction.response.send_message("**Erreur** · Ce salon est déjà dans la blacklist de ce chatbot.", ephemeral=True)
            chatbot.blacklist.append(channel.id)
        chatbot.save()
        
        if user:
            await interaction.response.send_message(f"**Succès** · L'utilisateur **{user}** a été ajouté à la blacklist du chatbot **{chatbot}**.")
        else:
            await interaction.response.send_message(f"**Succès** · Le salon **{channel}** a été ajouté à la blacklist du chatbot **{chatbot}**.")
        
    @blacklists_group.command(name='remove')
    @app_commands.rename(chatbot_id='chatbot')
    async def _blacklist_remove(self, interaction: discord.Interaction, chatbot_id: int, user: discord.User | None = None, channel: discord.TextChannel | discord.Thread | None = None):
        """Retire un utilisateur ou un salon de la blacklist d'un chatbot

        :param chatbot_id: Identifiant unique du chatbot
        :param user: Utilisateur à retirer de la blacklist
        :param channel: Salon à retirer de la blacklist
        """
        guild = interaction.guild
        if not isinstance(guild, discord.Guild):
            raise commands.BadArgument('Cette commande ne peut être utilisée que sur un serveur.')
        
        author = interaction.user
        comchan = interaction.channel
        if not isinstance(channel, (discord.TextChannel, discord.Thread)) or not comchan:
            raise commands.BadArgument('Cette commande ne peut être utilisée que sur un salon textuel ou un thread.')                     
        if not comchan.permissions_for(author).manage_messages: #type: ignore
            return await interaction.response.send_message("**Permissions insuffisantes** · Vous devez avoir la permission de gérer les messages pour utiliser cette commande.", ephemeral=True)
        
        chatbot = self.get_chatbot(guild, chatbot_id)
        if not chatbot:
            return await interaction.response.send_message("**Erreur** · Ce chatbot n'existe pas.", ephemeral=True)
        
        if not user and not channel:
            return await interaction.response.send_message("**Erreur** · Vous devez spécifier un utilisateur ou un salon à retirer de la blacklist.", ephemeral=True)
        
        if user:
            if user.id not in chatbot.blacklist:
                return await interaction.response.send_message("**Erreur** · Cet utilisateur n'est pas dans la blacklist de ce chatbot.", ephemeral=True)
            chatbot.blacklist.remove(user.id)
        elif channel:
            if channel.id not in chatbot.blacklist:
                return await interaction.response.send_message("**Erreur** · Ce salon n'est pas dans la blacklist de ce chatbot.", ephemeral=True)
            chatbot.blacklist.remove(channel.id)
        chatbot.save()
        
        if user:
            await interaction.response.send_message(f"**Succès** · L'utilisateur **{user}** a été retiré de la blacklist du chatbot **{chatbot}**.")
        else:
            await interaction.response.send_message(f"**Succès** · Le salon **{channel}** a été retiré de la blacklist du chatbot **{chatbot}**.")
            
    @blacklists_group.command(name='list')
    @app_commands.rename(chatbot_id='chatbot')
    async def _blacklist_list(self, interaction: discord.Interaction, chatbot_id: int):
        """Liste les utilisateurs et salons blacklistés d'un chatbot

        :param chatbot_id: Identifiant unique du chatbot
        """
        guild = interaction.guild
        if not isinstance(guild, discord.Guild):
            raise commands.BadArgument('Cette commande ne peut être utilisée que sur un serveur.')
        
        chatbot = self.get_chatbot(guild, chatbot_id)
        if not chatbot:
            return await interaction.response.send_message("**Erreur** · Ce chatbot n'existe pas.", ephemeral=True)
        
        if not chatbot.blacklist:
            return await interaction.response.send_message(f"**Blacklist vide** · Il n'y a aucun utilisateur ou salon blacklisté pour le chatbot **{chatbot}**.")
        
        blacklist_users = [guild.get_member(uid) for uid in chatbot.blacklist if guild.get_member(uid)]
        blacklist_channels = [guild.get_channel(cid) for cid in chatbot.blacklist if guild.get_channel(cid)]
        blacklist = [b for b in blacklist_users + blacklist_channels if b]
        
        if not blacklist:
            return await interaction.response.send_message(f"**Blacklist vide** · Il n'y a aucun utilisateur ou salon blacklisté pour le chatbot **{chatbot}**.")
        
        em = discord.Embed(title=f"Blacklist du chatbot **{chatbot}**", color=EMBED_DEFAULT_COLOR)
        em.description = '\n'.join(f'• {c.mention}' for c in blacklist)
        await interaction.response.send_message(embed=em)
        
    @_chat_load.autocomplete('chatbot_id')
    @_chatbot_delete.autocomplete('chatbot_id')
    @_modchat_edit.autocomplete('chatbot_id')
    @_blacklist_add.autocomplete('chatbot_id')
    @_blacklist_remove.autocomplete('chatbot_id')
    @_blacklist_list.autocomplete('chatbot_id')
    async def chatbot_id_autocomplete(self, interaction: discord.Interaction, current: str):
        if not isinstance(interaction.guild, discord.Guild):
            return []
        chatbots = self.get_chatbots(interaction.guild)
        r = fuzzy.finder(current, chatbots, key=lambda c: c.name)
        return [app_commands.Choice(name=c.name, value=c.id) for c in r]
    
    @_modchat_edit.autocomplete('key')
    async def chatbot_key_autocomplete(self, interaction: discord.Interaction, current: str):
        return [app_commands.Choice(name=k, value=k) for k in ['name', 'description', 'avatar_url', 'system_prompt', 'temperature', 'context_size']]
    
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Répond aux messages des utilisateurs avec un chatbot IA si on le mentionne, lui répond, ou si on est sur un thread avec le mode de réponse automatique activé."""
        if message.channel.id not in self.sessions:
            if not self.bot.user:
                return
            if message.channel.type not in (discord.ChannelType.text, discord.ChannelType.private_thread, discord.ChannelType.public_thread):
                return
            if not self.bot.user.mentioned_in(message):
                return
            return await message.reply("**Aucun chatbot** · Il n'y a pas de chatbot actuellement attaché à ce salon.\nUtilisez </chat load:1105603534310879345> ou </chat temp:1105603534310879345> pour en attacher un.", delete_after=10, mention_author=False)
        
        if message.author.bot:
            return
        
        session = self.get_session(message.channel) # type: ignore
        if session.check_blacklist(message.author.id) or session.check_blacklist(message.channel.id):
            return
        
        await session.handle_message(message)
            

async def setup(bot):
    await bot.add_cog(Chatter(bot))