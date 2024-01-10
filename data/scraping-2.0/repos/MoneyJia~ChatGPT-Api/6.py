import openai
import os

os.environ["all_proxy"] = 'http://127.0.0.1:10792' # 如果需要，请替换为你的代理地址

openai.api_key = "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"

completion = openai.Image.create(
    prompt= "一只白色小猫\在野地上\奔跑",
    n= 1,
    size= '512x512'
)

# 打印助手的回复
print(completion)


# # 修改图片尺寸
# from PIL import Image

# def transfer(infile, outfile) :
#     im = Image.open(infile)
#     reim=im.resize((512， 512))#宽x高
    
#     reim.save(outfile,dpi=(200.0,200.0)) ##200.0,200.0分别为想要设定的dpi值
# if1l main_':name_
#     infil=r"mask.png"
#     outfile=r"mask 512.png"
#     transfer(infil, outfile)
# 请逐行翻一下以上代码