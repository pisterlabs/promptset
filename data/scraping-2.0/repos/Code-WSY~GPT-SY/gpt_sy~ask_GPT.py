import openai
from Box_Dialog import *

def askGPT(messages, MODEL, MODEL_use_mode, temperature, max_tokens):
    """
    :param messages: 历史记录
    :param MODEL: 应用的模型
    :param MODEL_use_mode: 模型使用的输入格式
    :param temperature: 温度
    :param max_tokens: 最大输出长度
    :return:
    """
    output = ""
    Dialog_box.config(state=tk.NORMAL)
    Dialog_box.insert(tk.END, "AI：\n")
    Dialog_box.config(state=tk.DISABLED)

    if MODEL_use_mode == "ChatCompletion":
        response = openai.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            n=1,
            max_tokens=max_tokens,
            stream=True,

        )
        for chunk in response:
            try:
                #正常输出
                answer = chunk.choices[0].delta.content
                output += answer
                Dialog_box.config(state=tk.NORMAL)
                Dialog_box.insert(tk.END, answer)
                Dialog_box.see(tk.END)
                Dialog_box.update()
                Dialog_box.config(state=tk.DISABLED)
            except:
                #流文本输出最后的换行符，不然会出现异常
                Dialog_box.config(state=tk.NORMAL)
                Dialog_box.insert(tk.END, "\n")
                Dialog_box.see(tk.END)
                Dialog_box.update()
                Dialog_box.config(state=tk.DISABLED)



    elif MODEL_use_mode == "Completion":

        response = openai.completions.create(
            model=MODEL,
            prompt=messages[-1]["prompt"],
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            stream=True,
        )
        #输出对话
        for chunk in response:
            Dialog_box.config(state=tk.NORMAL)
            answer = chunk.choices[0].text
            output += answer
            Dialog_box.insert(tk.END, answer)
            Dialog_box.see(tk.END)
            Dialog_box.update()
            Dialog_box.config(state=tk.DISABLED)

        #输出最后一个换行符
        Dialog_box.config(state=tk.NORMAL)
        Dialog_box.insert(tk.END, "\n")
        Dialog_box.see(tk.END)
        Dialog_box.config(state=tk.DISABLED)

    elif MODEL_use_mode== "Edit":
        #给定一个提示和一条指令，模型将返回提示的编辑版本。
        response = openai.edits.create(
            model=MODEL,
            input=messages[-1]["input"],
            instruction=messages[-1]["instruction"],
            temperature=temperature,
            n=1,
        )
        answer = response.choices[0].text
        output = answer
        # 输出对话
        for chunk in answer:
            Dialog_box.config(state=tk.NORMAL)
            Dialog_box.insert(tk.END, chunk)
            Dialog_box.see(tk.END)
            Dialog_box.update()
            Dialog_box.config(state=tk.DISABLED)

        # 输出最后一个换行符
        Dialog_box.config(state=tk.NORMAL)
        Dialog_box.insert(tk.END, "\n")
        Dialog_box.see(tk.END)
        Dialog_box.config(state=tk.DISABLED)

    elif MODEL_use_mode == "Embedding":
        response = openai.embeddings.create(
            model=MODEL,
            input=messages[-1]["input"],
        )
        answer = response["data"][0]["embedding"]
        output = answer
        # 输出对话
        for chunk in answer:
            Dialog_box.config(state=tk.NORMAL)
            Dialog_box.insert(tk.END, chunk+"\n")
            Dialog_box.see(tk.END)
            Dialog_box.update()
            Dialog_box.config(state=tk.DISABLED)

    elif MODEL_use_mode == "Image.create":
        response = openai.images.create_variation(
            image=messages[-1]["image"],
            n=1,
            size="1024x1024",
        )
        answer = response["data"][0]["url"]
        output = answer
        # 输出对话
        Dialog_box.config(state=tk.NORMAL)
        Dialog_box.insert(tk.END, answer)
        Dialog_box.see(tk.END)
        Dialog_box.update()
        Dialog_box.config(state=tk.DISABLED)

    elif MODEL_use_mode == "Image.create_edit":
        response = openai.images.edit(
            image=messages[-1]["image"],#要编辑的图像
            mask=messages[-1]["mask"],#一个额外的图像，其完全透明的区域（例如 alpha 值为零的区域）指示应该编辑图像的位置。
            prompt=messages[-1]["prompt"],#一个文本片段，用于指导编辑。
            n=1,
        )
        answer = response.data[0].url
        output = answer
        # 输出对话
        Dialog_box.config(state=tk.NORMAL)
        Dialog_box.insert(tk.END, answer)
        Dialog_box.see(tk.END)
        Dialog_box.update()
        Dialog_box.config(state=tk.DISABLED)
    return output



if __name__ == "main":
    pass
