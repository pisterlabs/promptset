import re
import time
from typing import List
from langchain.schema.document import Document
from langchain.schema.language_model import BaseLanguageModel
from utils.stream_chain import Stream_Chain, load_summarize_chain
from utils.translation import Translation_Baidu
from sdxl.sd_refiner_model import SD_Refiner_Model
from sdxl.sd_base_model import SD_Base_Model
from PIL import Image
from abc import abstractmethod
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from embeddings.embedding_helper import EmbeddingHelper


class Note_Generate_Utils:
    llm: BaseLanguageModel
    max_image_count: int = 10,
    num_per_prompt_image: int = 2

    def __init__(self,
                 llm,
                 base_file_path: str = None,
                 max_image_count: int = 10,
                 num_per_prompt_image: int = 2) -> None:
        self.llm = llm
        self.max_image_count = max_image_count
        self.num_per_prompt_image = num_per_prompt_image
        self.base_file_path = base_file_path
        self.split_count = 3

    @abstractmethod
    def read_note_list(self, **kwargs):
        pass

    @abstractmethod
    def summary_response(self, details: list = None) -> str:
        pass

    def generate_image_sentence(self, content: str):
        prompt_template = """'{content}',你的任务是根据上述文章内容，尽量生成多条描述自然景色的句子，句与句之间用~进行分割"""
        PROMPT = PromptTemplate(template=prompt_template,
                                input_variables=["content"])
        # 重新设置模型参数
        chain = LLMChain(llm=self.llm,
                         prompt=PROMPT,
                         llm_kwargs={
                             'temperature': 0.95,
                             'top_p': 0.7
                         })

        result = chain.predict(content=content)
        return result

    def get_image_flags(self, content: str, style: dict, loras: list):
        images = re.findall('【(.*?)】', content)
        text_to_image_list = []
        if images and len(images) > 0:
            for i in range(0, len(images)):
                text_to_image_list.append(images[i])
        else:
            content = self.generate_image_sentence(content)
            text_to_image_list = content.split("~")

        # 翻译成英文
        texts_result: List[str] = []
        # texts_result.extend(text_to_image_list)
        for item in text_to_image_list[:self.max_image_count]:
            time.sleep(3)
            texts_result.append(Translation_Baidu.excute_translation(item))
        return {"texts": texts_result, "style": style, "loras": loras}

    def generate_images_by_image(self, image_url: str, style: dict, text: str):
        # 关闭LLM模型，释放显卡资源
        if self.llm.model is not None:
            self.llm.unload_model()
            time.sleep(5)
        text_english = " "
        if text:
            text_english = Translation_Baidu.excute_translation(text)
        sd_model = SD_Refiner_Model().instance(is_combine_base=False)
        sd_model.load_model()
        prompt = style["prompt"].format(prompt=text_english).lower()
        negative_prompt = style["negative_prompt"]
        images = sd_model.get_image_to_image_single_prompt(
            query=prompt,
            image_url=image_url,
            image_count=4,
            negative_prompt=negative_prompt)
        # 关闭SDXL模型，释放显卡资源
        sd_model.unload_model()
        return images

    def generate_images(self, texts: list, style: dict, loras: list):
        # 关闭LLM模型，释放显卡资源
        if self.llm.model is not None:
            self.llm.unload_model()
            time.sleep(5)
        # 加载了Lora,只使用basemodel，效果最好
        if len(loras) > 0:
            sd_model = SD_Base_Model.instance()
            sd_model.load_model()
            sd_model.fuse_lora(loras=loras)
        # 不加载lora 使用refiner+base，效果最好
        else:
            sd_model = SD_Refiner_Model.instance(is_combine_base=True)
            sd_model.load_model()
        image_addr = []
        for item in texts:
            try:
                name = self._name_image(item)
                prefix = ", ".join([(i["tag_words"]) for i in loras
                                    ]) + ", " if len(loras) > 0 else ""
                prompt = prefix + style["prompt"].format(prompt=item).lower()
                negative_prompt = style["negative_prompt"]
                target_image = sd_model.get_image_to_image_single_prompt(
                    query=prompt,
                    image_count=self.num_per_prompt_image,
                    negative_prompt=negative_prompt)
                for i in range(len(target_image)):
                    target_image[i].save(
                        f"{self.base_file_path}\\{name}_{i}.jpg", "JPEG")
                    image_addr.append(f"{self.base_file_path}\\{name}_{i}.jpg")
            except Exception:
                pass

        # 关闭SDXL模型，释放显卡资源
        sd_model.unload_model()
        return image_addr

    def _name_image(self, sentence: str):
        words = sentence.split()
        initials = [word[0] for word in words]
        return "".join(initials)

    def generate_article_content(self,
                                 text: str,
                                 prompt_template: str,
                                 is_only_return_result: bool = True):
        PROMPT = PromptTemplate(template=prompt_template,
                                input_variables=["text"])
        # 将文本进行拆分成段
        texts, text = self._get_middle_partal_chapter(
            text=text, split_count=self.split_count)
        # 定义分段生成方法
        # 内容长度过长，则采用分段参考生成的策略
        if len(text) > 1024 * 2 and self.split_count > 3:
            content = self._summarize_docs(texts=texts, PROMPT=PROMPT)
        else:
            # 重新设置模型参数
            chain = Stream_Chain(llm=self.llm,
                                 prompt=PROMPT,
                                 llm_kwargs={
                                     "temperature": 0.95,
                                     "top_p": 0.7
                                 })
            content = chain.predict(text=text)
        if is_only_return_result is False:
            return {"original": text, "content": content}
        return {"content": content}

    # 定义分段生成方法
    def _summarize_docs(self, texts: List[str], PROMPT: PromptTemplate):
        if texts:
            combine_prompt = PromptTemplate(template="""已知信息：'{text}'
                你的任务将已知信息，改编成一篇散文式的新文章，文章的用词，
                造句必须丰富，而且文章里面对场景的描写一定要具体，要细致""",
                                            input_variables=["text"])
            chain = load_summarize_chain(self.llm,
                                         chain_type="map_reduce",
                                         return_intermediate_steps=True,
                                         map_prompt=PROMPT,
                                         combine_prompt=combine_prompt,
                                         verbose=True,
                                         llm_kwargs={
                                             "temperature": 0.8,
                                             "top_p": 0.6
                                         })

            docs = [Document(page_content=text) for text in texts]
            summ = chain.stream({"input_documents": docs},
                                return_only_outputs=True)
            return summ

    def _get_middle_partal_chapter(self, text: str, split_count: int = 1):
        texts = EmbeddingHelper.splitText(text=text,
                                          chunk_size=1024 * 2,
                                          overlap=0)
        middle = len(texts) // 2
        if len(texts) % 2 == 0:
            texts = texts[middle - 1:middle + (split_count - 1)]
        else:
            texts = texts[middle:middle + (split_count - 1)]
        new_text = "".join(texts)
        return texts, new_text
