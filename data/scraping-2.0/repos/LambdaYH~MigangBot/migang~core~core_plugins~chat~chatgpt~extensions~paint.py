import random
import traceback

import openai
from nonebot.log import logger

from migang.core import get_config

from ..extension import extension_manager

style_preset = (
    "Chinese style, ink painting",
    "anime style, colored-pencil",
    "anime style, colored crayons",
)
custom_size = (512, 512)


@extension_manager(
    name="paint",
    description="paint a picture，使用/#paint&CONTENT#/，其中CONTENT是用逗号分隔的描述性词语。(例如：/#paint&兔子,草地,彩虹#/)",
    refer_word=["paint", "画", "图"],
)
async def _(content: str):
    proxy = await get_config("proxy", "chat_chatgpt")
    if proxy:
        if not proxy.startswith("http"):
            proxy = "http://" + proxy
        openai.proxy = proxy
    # style = "anime style, colored-pencil"
    style = random.choice(style_preset)
    try:
        response = await openai.Image.acreate(
            prompt=content + ", " + style,
            n=1,
            size=f"{custom_size[0]}x{custom_size[1]}",
        )
    except Exception:
        logger.error(f"调用openai绘图错误：{traceback.format_exc()}")
        return {
            "text": "画笔没墨了...",
        }
    image_url = response["data"][0]["url"]

    if image_url is None:
        return {
            "text": "画笔没墨了...",
        }
    elif "rejected" in response:
        # 返回的信息将会被发送到会话中
        return {
            "text": "抱歉，这个图违反了ai生成规定，可能是太色了吧",  # 文本信息
        }
    else:
        # 返回的信息将会被发送到会话中
        return {
            "image": image_url,  # 图片url
        }
