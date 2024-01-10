from bot.chat import OpenAIBot,ExplorerBot,GPT4Bot,LoveBot,SimpleBot,AssistantBot,AssistantBotV2

BotsMapping = {
    'OpenAIBot': OpenAIBot,
    'ExplorerBot': ExplorerBot,
    'GPT4Bot': GPT4Bot,
    'LoveBot': LoveBot,
    'SimpleBot':SimpleBot,
    'AssistantBot':AssistantBot,
    'AssistantBotV2':AssistantBotV2
}


def if_support_memory(botname):
    return botname in ['GPT4Bot']

def get_bots_list():
    return [
        ('OpenAIBot', '基础'),
        ('GPT4Bot', 'GPT4（支持长期记忆哦）'),
        ('AssistantBot','智能助理'),
        ('AssistantBotV2','智能助理2号'),
        ('ExplorerBot', '首页引导员'),
        ('LoveBot', '爱情脑'),
        ('SimpleBot','测试')
    ]

def load_bot(bot_name, description, messages, caller_id, context, username=None, profilename=None):
    return BotsMapping[bot_name](description, messages, caller_id,context, username, profilename)

def load_bot_by_profile(profile, caller_id, context={}, username=None):
    context = {**context,**{"profile":profile}}
    return BotsMapping[profile.bot](profile.description, profile.message, caller_id, context, username, profile.name)
