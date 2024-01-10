import json
import openai

class GLM3Prompt:
    """
    ChatGLM3接口调用
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "glm3_url": ("STRING",{"default": "http://localhost:8000/v1"}),
                "system_prompt": ("STRING", {"multiline": True,"default":"你是名绘图提示词专家，你会根据客户的描述内容，精确返回对应图片提示，注意提示词必须是英文"}),
                "text": ("STRING", {"multiline": True})
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("提示词",)

    FUNCTION = "translate"

    #OUTPUT_NODE = False

    CATEGORY = "lam"

    def translate(self, glm3_url,system_prompt,text):
        openai.api_base = glm3_url
        openai.api_key = "xxx"
        params = dict(model="chatglm3", messages=[{"role": "system", "content": system_prompt},{"role": "user", "content": text}], stream=False)
        response = openai.ChatCompletion.create(**params)
        reply = response.choices[0].message.content
        return (reply,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "GLM3Prompt": GLM3Prompt
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "GLM3Prompt": "ChatGLM3提示词生成工具"
}
