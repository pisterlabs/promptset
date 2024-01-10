# 机器翻译
import json
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
)
from tqdm import tqdm
from units.load_data import load_data


async def tranlate(pages):
    model = "Qwen-14B-Chat-Int4"
    examples = [
        {
            "input": '''A boy named Li Ming is reading a science fiction novel. He is attracted by the peculiar world within and the breathtaking imagination. Every character, every scene, seems to vividly present itself in his mind. Li Ming enjoys immersing himself in the world of the novel, as if he could travel to another universe filled with mystery.''',
            "output": '''{"result": "一位名叫李明的男孩正在读一本科幻小说。他被其中的奇特世界和令人叹为观止的想象力所吸引。每个角色，每个场景，似乎都在他的头脑中形象地展现出来。李明喜欢这样沉浸在小说的世界里，仿佛他可以穿越到另一个充满神秘的宇宙。"}'''
        },
        {
            "input": '''おはようございます、李さん。今日は天気がいいですね、散歩に出かけるのに適しています。公園で散歩するのに興味はありますか？''',
            "output": '''{"result": "早上好，李先生。今天天气真好，适合出去散步。你对在公园散步有兴趣吗？"}'''
        },
        {
            "input": '''Dans cette belle ville, il y a de grands parcs verts et de vastes plaines. Chaque aube et chaque coucher de soleil sont extraordinaires. Se promener dans le parc, on peut entendre le chant des oiseaux, ressentir les cadeaux de la nature. Les habitants de cette ville sont très amicaux, ils accueillent toujours chaleureusement les visiteurs du monde entier. Cette ville, c'est chez moi, j'aime chaque brin d'herbe, chaque montagne et chaque goutte d'eau ici.''',
            "output": "{'result': '在这座美丽的城市里，有大片绿色的公园和广袤的平原。每一抹日出和日落都分外壮丽。去公园里散步，可以听到鸟儿的歌唱，感受自然的馈赠。这座城市的居民非常友好，他们总是热情欢迎世界的访客。这座城市，就是我的家，我爱这里的一草一木，一山一水'}"
        }

    ]
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}")
        ]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", '''你现在需要完成一个**机器翻译**的任务，你需要将输入的**其他语言**的文本翻译为**中文**文本''' +
             '''翻译的结构请用json的形式展示，除了这个json以外请不要输出别的多余的话。'''),
            few_shot_prompt,
            ("human", "{input}")
        ]
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048, chunk_overlap=16)
    chain = LLMChain(
        prompt=final_prompt,
        # 温度调为0，可以保证输出的结果是确定的
        llm=ChatOpenAI(
            temperature=0,
            model_name=model,
            openai_api_key="EMPTY",
            openai_api_base="http://localhost:8000/v1")
        # output_parser=output_parser
    )

    res = {"result": ""}
    with tqdm(total=len(pages)) as pbar:
        pbar.set_description('Processing:')
        for page in pages:
            texts = text_splitter.split_text(page.page_content)
            for text in texts:
                tmp = await chain.arun(input=text, return_only_outputs=True)
                try:
                    json_object = json.loads(tmp)
                    res["result"] += json_object["result"]
                except Exception as e:
                    continue

            pbar.update(1)
    return res
