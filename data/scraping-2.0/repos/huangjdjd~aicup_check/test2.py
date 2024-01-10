<<<<<<< HEAD
import openai
import gradio as gr
import time
import numpy as np
=======
# import gradio as gr

# with gr.Blocks() as demo:

    # turn = gr.Textbox(1, interactive=False, label="Turn")
    # board = gr.Dataframe(value=[""] , interactive=False, type="array",headers=["name", "age", "gender"],row_count=5,)

    # def place(board, turn, evt: gr.SelectData):
    #     if evt.value:
    #         return board, turn
    #     board[evt.index[0]][evt.index[1]] = turn
    #     turn=1
    #     return board, turn

    # board.select(place, [board, turn], [board, turn])

# if __name__ == "__main__":
#     demo.launch()
import gradio as gr

def update_label():
    return "1"
>>>>>>> fc8ec496adbdcd990b7b7670612997e5df977023

# 创建一个按钮元素
button = gr.Button("Click me!")

<<<<<<< HEAD
openai.api_key = "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"

#紀錄該段落是否已經通過檢測的dict，key 為段落；value 為是否通過(1為通過，0為未通過)
Passed = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}

#檢查用的function，input_text為要檢核的段落內容；part為第幾段
#關鍵字: 檢核通過  (如果AI回覆為通過，則務必請AI在reply中加入繁體 "檢核通過" 以方便判定)
def check(input_text, part):
    
    
    reply = input_text   #隨便假設的
    
    
    #更新通過紀錄Passed
    if "檢核通過" in reply:
        Passed[part] = 1
    else:
        Passed[part] = 0
    reply += str(Passed[part])
    

    return reply   #回傳AI的回覆

#取得Passed dict裡面的值
def getPassed(part):
    return Passed[part]

#取得總進度
def getProgress():
    progress = 0
    for i in range(1,8):
        progress += Passed[i]
        return progress/7

#更新所有進度
def updateValue():
    
    dict = {"完成度": getProgress(),
                "壹、環境": getPassed(1), "貳、演算方法與模型架構": getPassed(2), "參、創新性": getPassed(3), 
                "肆、資料處理": getPassed(4), "伍、訓練方式": getPassed(5), "陸、分析與結論": getPassed(6), "捌、使用的外部資源與參考文獻": getPassed(8)}
    return dict
    
def value():
    arr = np.array([["完成度", getProgress()], 
                    ["壹", getPassed(1)], 
                    ["貳", getPassed(2)],
                    ["參", getPassed(3)],
                    ["肆", getPassed(4)],
                    ["伍", getPassed(5)],
                    ["陸", getPassed(6)],
                    ["捌", getPassed(8)],
                ])
    return arr

#主頁面
with gr.Blocks() as demo:
    gr.Markdown(
    """
    # 報告檢核系統
    請依序輸入要檢核的段落
    """)
    with gr.Row():
        
        # with gr.Column(scale=1):

            
        #     board = gr.DataFrame(
        #         headers=["段落", "檢核結果"],
        #         col_count=2, 
        #         row_count=(7,'fixed'), 
        #         interactive=False,
        #         type="array",
        #         value=value(),
        #         every=1
        #         )
        #     board.change(fn=value,inputs=None,outputs=board)
        with gr.Column(scale=3):
            
            with gr.Tab(label="壹、環境"):
                input = gr.Textbox(max_lines=7, lines=7, label="請說明使用的作業系統、語言、套件(函式庫)、預訓練模型、額外資料集等。如使用預訓練模型及額外資料集，請逐一列出來源。(200~600字)")
                check_btn = gr.Button("Check")
                part = gr.Slider(1, 8, value=1, visible = False)
                output = gr.Textbox(label="段落檢核結果")
                check_btn.click(fn=check, inputs=[input,part], outputs=output)
                
                
            with gr.Tab(label="貳、演算方法與模型架構"):
                input = gr.Textbox(max_lines=7, lines=7, label="說明演算法設計、模型架構與模型參數，包括可能使用的特殊處理方式。(400~1200字)")
                check_btn = gr.Button("Check")
                part = gr.Slider(1, 8, value=2, visible = False)
                output = gr.Textbox(label="段落檢核結果")
                check_btn.click(fn=check, inputs=[input,part], outputs=output)
                
            with gr.Tab(label="參、創新性"):
                input = gr.Textbox(max_lines=7, lines=7, label="說明演算法之創新性或者修改外部資源的哪一部分。(300~1200字)")
                check_btn = gr.Button("Check")
                part = gr.Slider(1, 8, value=3, visible = False)
                output = gr.Textbox(label="段落檢核結果")
                check_btn.click(fn=check, inputs=[input,part], outputs=output)
                
            with gr.Tab(label="肆、資料處理"):
                input = gr.Textbox(max_lines=7, lines=7, label="說明對資料的處理或擴增的方式，例如對資料可能的刪減、更正或增補。(300~1500字)")
                check_btn = gr.Button("Check")
                part = gr.Slider(1, 8, value=4, visible = False)
                output = gr.Textbox(label="段落檢核結果")
                check_btn.click(fn=check, inputs=[input,part], outputs=output)
                
            with gr.Tab(label="伍、訓練方式"):
                input = gr.Textbox(max_lines=7, lines=7, label="說明模型的訓練方法與過程。(400~1000字)")
                check_btn = gr.Button("Check")
                part = gr.Slider(1, 8, value=5, visible = False)
                output = gr.Textbox(label="段落檢核結果")
                check_btn.click(fn=check, inputs=[input,part], outputs=output)
                
            with gr.Tab(label="陸、分析與結論"):
                input = gr.Textbox(max_lines=7, lines=7, label="分析所使用的模型及其成效，簡述未來可能改進的方向。分析必須附圖，可將幾個成功的和失敗的例子附上並說明之。(400~2500字)")
                check_btn = gr.Button("Check")
                part = gr.Slider(1, 8, value=6, visible = False)
                output = gr.Textbox(label="段落檢核結果")
                check_btn.click(fn=check, inputs=[input,part], outputs=output)
                
            with gr.Tab(label="捌、使用的外部資源與參考文獻"):
                input = gr.Textbox(max_lines=7, lines=7, label="參考文獻請以APA格式為主。")
                check_btn = gr.Button("Check")
                part = gr.Slider(1, 8, value=8, visible = False)
                output = gr.Textbox(label="段落檢核結果")
                check_btn.click(fn=check, inputs=[input,part], outputs=output)
        
    
        
demo.queue().launch(share=True)
=======
# 创建一个标签元素，初始值为空字符串
label = gr.Label("")

# 当按钮被点击时，调用update_label函数，将返回的值显示在标签上
button.click(update_label, outputs=label)

# 将按钮和标签放入Gradio界面中
interface = gr.Interface(fn=None, inputs=button, outputs=label, live=True)

if __name__ == "__main__":
    interface.launch()
>>>>>>> fc8ec496adbdcd990b7b7670612997e5df977023
