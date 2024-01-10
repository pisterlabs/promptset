import graphene
from graphene import relay
from graphene_mongo import MongoengineObjectType
from .core.user.models import User
from .core.settings.models as Settings
import openai

from .core.chat.models import ChatMessage  # Ensure this model is defined in your models

# OpenAI API key setup
openai.api_key = 'your-openai-api-key'

# Existing UserType and SettingsType definitions...
class UserType(MongoengineObjectType):
    class Meta:
        model = User
        interfaces = (relay.Node, )

class SettingsType(MongoengineObjectType):
    class Meta:
        model = Settings
        interfaces = (relay.Node, )

class ChatMessageType(MongoengineObjectType):
    class Meta:
        model = ChatMessage
        interfaces = (relay.Node, )

class Query(graphene.ObjectType):
    node = relay.Node.Field()
    all_users = graphene.List(UserType)
    user = graphene.Field(UserType, id=graphene.String(required=True))
    user_settings = graphene.Field(SettingsType, user_id=graphene.String(required=True))
    all_chat_messages = graphene.List(ChatMessageType)

    def resolve_all_users(self, info):
        return list(User.objects.all())

    def resolve_user(self, info, id):
        return User.objects.get(id=id)

    def resolve_user_settings(self, info, user_id):
        return Settings.objects(user=user_id).first()

    def resolve_all_chat_messages(self, info):
        return list(ChatMessage.objects.all())

class CreateUser(graphene.Mutation):
    class Arguments:
        email = graphene.String(required=True)
        name = graphene.String(required=True)

    user = graphene.Field(UserType)

    def mutate(self, info, email, name):
        user = User(email=email, name=name)
        user.save()
        return CreateUser(user=user)

class CreateChatMessage(graphene.Mutation):
    class Arguments:
        message = graphene.String(required=True)

    chat_message = graphene.Field(ChatMessageType)

    def mutate(self, info, message):
        # Save user's message
        user_message = ChatMessage(message=message)
        user_message.save()

        # Interact with OpenAI GPT-4
        gpt_response = openai.Completion.create(
            engine="text-davinci-004",
            prompt=message,
            max_tokens=150
        )

        # Save and return GPT-4's response
        gpt_message = ChatMessage(message=gpt_response.choices[0].text.strip())
        gpt_message.save()

        return CreateChatMessage(chat_message=gpt_message)

class Mutation(graphene.ObjectType):
    create_user = CreateUser.Field()
    create_chat_message = CreateChatMessage.Field()

schema = graphene.Schema(query=Query, mutation=Mutation)
