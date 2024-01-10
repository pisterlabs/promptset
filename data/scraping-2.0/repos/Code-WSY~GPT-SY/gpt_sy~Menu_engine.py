import openai

from windows import *
def engine_list():
    models_list = tk.Tk()
    models_list.title("当前API可用模型")
    print("API:"+openai.api_key)
    print("API_Base:"+openai.api_base)
    #models_list.geometry("300x300")
    #models_list.resizable(False, False)
    #新建文本
    engine_list_text = tk.Text(models_list)
    #字体
    engine_list_text.config(font=(font_style, font_size))
    #添加滚动条
    engine_list_scroll = tk.Scrollbar(models_list)
    engine_list_scroll.config(command=engine_list_text.yview)
    engine_list_text.config(yscrollcommand=engine_list_scroll.set)
    engine_list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    engine_list_text.pack()
    #获取引擎列表
    models_list=openai.models.list()
    engine_list_text.config(state=tk.NORMAL)
    engine_list_text.delete(0.0, tk.END)
    engine_list_text.insert(tk.END, "API:"+openai.api_key+"\n")
    engine_list_text.insert(tk.END, "API_Base:"+openai.api_base+"\n")
    engine_list_text.insert(tk.END, "可用模型:\n")
    #显示引擎列表
    for i in models_list:
        #i是Model()对象
        #print(i)
        engine_list_text.insert(tk.END, i.object+": "+i.id+"\n")
    engine_list_text.config(state=tk.DISABLED)
# -----------------------------------------------------------------------------------#
filemenu_engine = tk.Menu(menubar, tearoff=0)
#加入菜单:可用模型
filemenu_engine.add_command(label="可用模型", command=lambda: engine_list())

