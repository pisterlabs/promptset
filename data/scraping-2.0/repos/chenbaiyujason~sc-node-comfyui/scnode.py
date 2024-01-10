import torch
import numpy as np
from PIL import Image, ImageEnhance
import os
import openai
import re
import global_table

#总输出节点，返回给请求的内容
class PostOutputAll:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("ASCII", ),
            }
        }

    RETURN_TYPES = ("ASCII",)
    FUNCTION = "text_string"
    OUTPUT_NODE = True
    CATEGORY = "SC/Gradio"

    def output(self,source):
        gt = global_table.Global()
        gt.set_update(True)

#输出到gradio表格

class GradioOutput:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("ASCII", {"multiline": False}),
                "target": ("ASCII",),
            }
        }

    RETURN_TYPES = ("ASCII",)
    FUNCTION = "text_string"
    OUTPUT_NODE = True
    CATEGORY = "SC/Gradio"

    def text_string(self, source, target):
        message = {source,target}
        gt = global_table.Global()
        gt.add_data(source,target)
        print(f"输出到gradio\n{gt.get_data().values.tolist()}")
        gt.get_data().to_csv('new_data.csv', index=False)
        return (message,)

#清除表格
class GradioClean:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Data": ("ASCII", {"multiline": False}),
            }
        }

    RETURN_TYPES = ("ASCII",)
    FUNCTION = "Clean_data"

    CATEGORY = "SC/Gradio"

    def Clean_data(self, Data):
        gt = global_table.Global()
        gt.clean_data()
        return (Data,)

#获取表格内容
class GradioGet:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": ("ASCII", {"multiline": False}),
            }
        }

    RETURN_TYPES = ("ASCII",)
    FUNCTION = "Get_data"

    CATEGORY = "SC/Gradio"

    def Get_data(self,):
        gt = global_table.Global()
        df = gt.get_data().head(2)
        result =df.values.tolist()
        print(','.join(result))
        return (','.join(result),)


# 变量版postGPT

class OnePostGPT:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "key": ("ASCII", {"multiline": False}),
                "role": ("ASCII", {"multiline": False}),
                "text": ("ASCII",),
            }
        }

    RETURN_TYPES = ("ASCII",)
    FUNCTION = "text_string"

    CATEGORY = "SC/GPT"

    def text_string(self, text, key, role):
        openai.api_key = key
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": role, "content": text}
            ]
        )
        return (completion.choices[0].message,)


# 输入版postGPT

class VerbOnePostGPT:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "key": ("STRING", {"multiline": False}),
                "role": ("STRING", {"multiline": False}),
                "text": ("ASCII",),
            }
        }

    RETURN_TYPES = ("ASCII",)
    FUNCTION = "text_string"

    CATEGORY = "SC/GPT"

    def text_string(self, text, key, role):
        openai.api_key = key
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": role, "content": text}
            ]
        )
        return (completion.choices[0].message,)


class MultiplePostGPT:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "key": ("ASCII", {"multiline": False}),
                "Messages": ("ASCII", {"multiline": False}),

            }
        }

    RETURN_TYPES = ("ASCII",)
    FUNCTION = "text_string"

    CATEGORY = "SC/GPT"

    def text_string(self, key, Messages):
        openai.api_key = key
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=Messages,
        )
        return (completion.choices[0].message,)


class OneGPTBuilder:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "role": ("ASCII", {"multiline": False}),
                "text": ("ASCII",),
            }
        }

    RETURN_TYPES = ("ASCII",)
    FUNCTION = "text_string"

    CATEGORY = "SC/GPT"

    def text_string(self, text, role):
        message = {"role": role, "content": text}
        return (message,)


class CombineGPTBuilder:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "PromptA": ("ASCII",),
                "PromptB": ("ASCII",),
            }
        }

    RETURN_TYPES = ("ASCII",)
    FUNCTION = "text_string"

    CATEGORY = "SC/GPT"

    def text_string(self, PromptA, PromptB):
        messages = [PromptA, PromptB]
        return (messages,)


class MultipleCombineGPTBuilder:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "PromptA": ("ASCII",{"default": ''}),
                "PromptB": ("ASCII",{"default": ''}),
            },
            "optional": {
                "PromptC": ("ASCII", {"default": ''}),
                "PromptD": ("ASCII", {"default": ''}),
                "PromptE": ("ASCII", {"default": ''}),
                "PromptF": ("ASCII", {"default": ''}),
            },
        }

    RETURN_TYPES = ("ASCII",)
    FUNCTION = "text_string"
    # MULTIPLE_NODE = True
    CATEGORY = "SC/GPT"

    def text_string(self, PromptA, PromptB, PromptC="", PromptD="", PromptE="", PromptF=""):
        #遍历每一个变量，如果该元素不为空，则将他添加到列表中
        messages = []
        for i in [PromptA, PromptB, PromptC, PromptD, PromptE, PromptF]:
            if i:
                messages.append(i)
        return (messages,)


# Prompt Preview
class PromptPreview:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "text": ("ASCII",), }
        }

    RETURN_TYPES = ("ASCII",)
    FUNCTION = "display"

    CATEGORY = "SC/Text"

    def display(self, text):
        text = str(text)
        text1 = text.split("\": \"")[1:]
        text2 = ''.join(str(t) for t in text1).split("\",")[:-1]
        result = ''.join(str(t) for t in text2).encode().decode('unicode_escape')
        # print(f"GPT返回预览：\n{result}")
        return (result,)


class String_TO_ASCII:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "text": ("STRING",), }
        }

    RETURN_TYPES = ("ASCII",)
    FUNCTION = "S2A"

    CATEGORY = "SC/Text"

    def S2A(self, text):
        return (text,)


class SCCLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("ASCII",), "clip": ("CLIP",)}}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "SC/Img"

    def encode(self, clip, text):
        return ([[clip.encode(text), {}]],)


# Multiple Text String Node

class Multiple_Text_String:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": '', "multiline": True}),
            }
        }

    RETURN_TYPES = ("ASCII",)
    FUNCTION = "text_string"

    CATEGORY = "SC/Text"

    def text_string(self, text):
        return (text,)


# Text String Node

class Single_Text_String:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": '', "multiline": False}),
            }
        }

    RETURN_TYPES = ("ASCII",)
    FUNCTION = "text_string"

    CATEGORY = "SC/Text"

    def text_string(self, text):
        return (text,)


# Text Combine

class Combine_Text_String:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text1": ("ASCII", {"default": '', "multiline": False}),
                "text2": ("ASCII", {"default": '', "multiline": False}),
            }
        }

    RETURN_TYPES = ("ASCII",)
    FUNCTION = "text_combine"

    CATEGORY = "SC/Text"

    def text_combine(self, text1, text2):
        return (text1 + text2,)


class Multiple_Combine_Text_String:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text1": ("ASCII", {"default": '', "multiline": False}),
                "text2": ("ASCII", {"default": '', "multiline": False}),
            },
            "optional":{
                "text3": ("ASCII", {"default": '', "multiline": False}),
                "text4": ("ASCII", {"default": '', "multiline": False}),
                "text5": ("ASCII", {"default": '', "multiline": False}),
                "text6": ("ASCII", {"default": '', "multiline": False}),
                "text7": ("ASCII", {"default": '', "multiline": False}),
                "text8": ("ASCII", {"default": '', "multiline": False}),
            }
        }

    RETURN_TYPES = ("ASCII",)
    FUNCTION = "text_combine"
    # MULTIPLE_NODE = True
    CATEGORY = "SC/Text"

    def text_combine(self, text1, text2, text3="", text4="", text5="", text6="", text7="", text8=""):
        #检查所有输入变量，如果不为空，则添加字符串
        texts = []
        for i in [text1, text2, text3, text4, text5, text6, text7, text8]:
            if i:
                texts.append(i)
        #返回texts变成的字符串，不要有间隔
        return (''.join(str(t) for t in texts),)


class Builder_Text_String:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("ASCII",),
                "prefix": ("STRING", {"default": '', "multiline": False}),
                "suffix": ("STRING", {"default": '', "multiline": False}),
            }
        }

    RETURN_TYPES = ("ASCII",)
    FUNCTION = "text_to_console"

    CATEGORY = "SC/Text"

    def text_to_console(self, text, prefix, suffix):
        t = prefix + text + suffix
        return (t,)


# Text Search

class SC_Search_and_Replace:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("ASCII",),
                "find": ("STRING", {"default": '', "multiline": False}),
                "replace": ("STRING", {"default": '', "multiline": False}),
            }
        }

    RETURN_TYPES = ("ASCII",)
    FUNCTION = "text_search_and_replace"

    CATEGORY = "SC/Text"

    def text_search_and_replace(self, text, find, replace):
        return (self.replace_substring(text, find, replace),)

    def replace_substring(self, text, find, replace):
        import re
        text = re.sub(find, replace, text)
        return text


class SC_Text_to_Console:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("ASCII",),
                "label": ("STRING", {"default": f'Text Output', "multiline": False}),
            }
        }

    RETURN_TYPES = ("ASCII",)
    OUTPUT_NODE = True
    FUNCTION = "text_to_console"

    CATEGORY = "SC/Text"

    def text_to_console(self, text, label):
        print(f"\033[34m{label}\n\033[33m{text}")
        return (text,)


NODE_CLASS_MAPPINGS = {
    "One Post to GPT": OnePostGPT,
    "One GPT Builder": OneGPTBuilder,
    "Combine GPT Prompt": CombineGPTBuilder,
    "Multiple Combine GPT Prompt": MultipleCombineGPTBuilder,

    "Multiple Post to GPT": MultiplePostGPT,

    "Prompt Preview": PromptPreview,
    "String to ASCII": String_TO_ASCII,
    "SCSCCLIPTextEncode": SCCLIPTextEncode,
    "Multiple Text String": Multiple_Text_String,
    "Single Text String": Single_Text_String,
    "Combine Text String": Combine_Text_String,
    "8 Combine Text String": Multiple_Combine_Text_String,
    "Get Gradio": GradioGet,
    "Clean Gradio": GradioClean,
    "Out Gradio": GradioOutput,
    "Builder Text String": Builder_Text_String,
    "SCSearch and Replace": SC_Search_and_Replace,
    "SCText to Console": SC_Text_to_Console,
    "Verb One Post to GPT": VerbOnePostGPT,
}
