import logging
from typing import Optional

from injector import inject, noninjectable, singleton
from langchain.chat_models import ChatOpenAI
from langchain.llms.base import BaseLanguageModel
from langchain.llms.fake import FakeListLLM
from langchain_deepread.settings.settings import Settings


logger = logging.getLogger(__name__)


class LLMComponent:
    @inject
    def __init__(self, settings: Settings) -> None:
        llm_mode = settings.llm.mode
        logger.info("Initializing the LLM in mode=%s", llm_mode)
        self.modelname = settings.openai.modelname
        match settings.llm.mode:
            case "local":
                ...  # Todo
            case "openai":
                openai_settings = settings.openai
                self._llm = ChatOpenAI(
                    temperature=openai_settings.temperature,
                    model_name=openai_settings.modelname,
                    api_key=openai_settings.api_key,
                    openai_api_base=openai_settings.api_base,
                    model_kwargs={"response_format": {"type": "json_object"}},
                )
            case "mock":
                self._llm_dict = {
                    "summary": FakeListLLM(
                        responses=[
                            '{"summary":"LAC空间学部与设计行业知识传播与分享有关","content":["LAC空间学部专注于设计行业前沿知识传播与分享，联合全球顶尖院校设计师、艺术家和学者。","设计组今年的主题关键词是Idiosyncrasy，意为“怪癖”、“特质”或“个人特有的气质或癖好”，并要寻找一部小说或电影来诠释这一关键词。","项目选择了《百年孤独》作为叙事灵感，探究了怪诞故事中的轮回主题和家族特质。","项目以“同理心”为核心策略，在项目中体现出对关键词的理解，推动设计方向。","选择老挝的石缸平原作为项目地点，转向调查老挝越战中的历史事件。","项目涉及情绪或物品的象征，致力于与观众产生共鸣。","在设计中使用了特定的配色，用色将恶与善、张力与谐和表达出来。","考虑了建筑的功能布局，主要是基于场地高差和当地建筑聚落形式设计。","作品集整体流程采取了循序渐进的研究和修改，最终呈现出设计的构思过程。","未来不会继续推进该项目。"],"title":"LAC空间","outline":[{"section_title":"LAC空间学部","section_list":["设计行业知识传播与分享","联合全球顶尖院校设计师、艺术家、学者","致力于设计行业前沿知识传播与分享"]},{"section_title":"关键信息点","section_list":["设计组今年主题关键词是Idiosyncrasy","选择了《百年孤独》作为叙事灵感","项目以“同理心”为核心策略","选择老挝的石缸平原作为项目地点","项目涉及情绪或物品的象征"]},{"section_title":"空间布局","section_list":["主要空间包括主会场、悬挂飞机雕塑、活竹屋顶","考虑了建筑的功能布局"]},{"section_title":"设计思考流程","section_list":["循序渐进的研究和修改","作品集整体流程采取了循序渐进的研究和修改"]},{"section_title":"未来展望","section_list":["未来不会继续推进该项目"]}],"tags":["设计","艺术","学术","项目","创意"],"qa":["Q: 为什么项目选择了《百年孤独》作为叙事灵感？\\nA: 项目通过探究怪诞故事中的轮回主题和家族特质来回应相似的命运。","Q: 为什么在整理了《百年孤独》的故事情节且捕捉了情绪之后，选择位于老挝的基地以及越南战争这一事件？\\nA: 项目从小说中总结出了关键词，并根据这些词定义寻找了符合定义的场地，然后转向调查相关历史事件。","Q: 整体画面用色上与项目有无联系？\\nA: 用色将恶与善、张力与谐和表达出来，暖色代表恶，冷色代表善。"],"recommends":["项目充分展现了设计团队对于关键词的深入诠释，令人深思。","设计中使用的配色方案充分表现了张力和谐，值得借鉴和学习。","效果图的构成借鉴了老挝的版画，呈现出东南亚炎热潮湿的夏季氛围，具有独特的韵味。"]}'
                        ]
                    ),
                    "qa": FakeListLLM(responses=['{"answer":"你好,帝阅DeepRead"}']),
                }

    @property
    def llm(self) -> BaseLanguageModel:
        return self._llm

    @llm.setter
    def llm(self, choice):
        try:
            self._llm = self._llm_dict[choice]
        except Exception as e:
            raise ValueError(f"Not supported for mock large models {e}")
