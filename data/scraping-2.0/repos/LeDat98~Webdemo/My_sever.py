from flask import Flask, render_template, request
import os
import random
from deep_translator import GoogleTranslator
import openai
import urllib.request

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static"

# @app.route('/')
# def home():
#     # return '<button onclick="window.location.href=\'/text_to_image\'">Go to image</button>'
#     return render_template('index.html')

@app.route("/",methods = ['GET','POST'])
def text_To_Image():

  #lấy text từ client 
  text = request.args.get('text')
  Style_list = [' ','A digital illustration of a  with clockwork machines, 4k, detailed, trending in artstation, fantasy vivid colors',
                'masterpiece, masterpiece, anime, sadboi, aesthetic, transparent color vinyl, highly detailed, reflections, transparent iridescent opaque rgb, chromatic aberration, +4k UHD',
                'mid century modern, indoor garden with fountain, retro,m vintage, designer furniture made of wood and plastic, concrete table, wood walls, indoor potted tree, large window, outdoor forest landscape, beautiful sunset, cinematic, concept art, sunstainable architecture, octane render, utopia, ethereal, cinematic light, -ar 16:9 -stylize 45000',
                'futuristic nighttime cyberpunk New York City skyline landscape vista photography by Carr Clifton & Galen Rowell, 16K resolution, Landscape veduta photo by Dustin Lefevre & tdraw, 8k resolution, detailed landscape painting by Ivan Shishkin, DeviantArt, Flickr, rendered in Enscape, Miyazaki, Nausicaa Ghibli, Breath of The Wild, 4k detailed post processing, atmospheric, hyper realistic, 8k, epic composition, cinematic, artstation —ar 16:9',
                'The Legend of Zelda landscape atmospheric, hyper realistic, 8k, epic composition, cinematic, octane render, artstation landscape vista photography by Carr Clifton & Galen Rowell, 16K resolution, Landscape veduta photo by Dustin Lefevre & tdraw, 8k resolution, detailed landscape painting by Ivan Shishkin, DeviantArt, Flickr, rendered in Enscape, Miyazaki, Nausicaa Ghibli, Breath of The Wild, 4k detailed post processing, artstation, rendering by octane, unreal engine —ar 16:9']
  if text:

      if 'Style1' in text:
        text = text.replace('Style1', "")#xoá chuỗi Style trong chuỗi văn bản 
        styles = Style_list[1] #thay đổi chuỗi Style ở đây
      elif 'Style2' in text:
        text = text.replace('Style2', "")
        styles = Style_list[2] #thay đổi chuỗi Style ở đây
      elif 'Style3' in text:
        text = text.replace('Style3', "")
        styles = Style_list[3] #thay đổi chuỗi Style ở đây
      elif 'Style4' in text:
        text = text.replace('Style4', "")
        styles = Style_list[4] #thay đổi chuỗi Style ở đây
      elif 'Style5' in text:
        text = text.replace('Style5', "")
        styles = Style_list[5] #thay đổi chuỗi Style ở đây
      elif text in text:
        styles = ' ' 
      
      #Dịch văn bản 
      print("start translate text to english")
      translated_text = GoogleTranslator(source='auto', target='en').translate(text)
      
      #gửi translated_text vào model và chạy
      text_t_img = f"{translated_text},{styles}"
      print(text_t_img)
      # Load your API key from an environment variable or secret management service
      openai.api_key = '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'
      openai.Model.list()
      response = openai.Image.create(prompt=text_t_img, n= 3, size= "1024x1024")
      image_url = response['data'][0]['url'] 
      image_url1 = response['data'][1]['url']
      image_url2 = response['data'][2]['url']
      
      #lưu ảnh#  
      ## tạo một tên ảnh ngẫu nhiên 
      sample_string = 'qwertyuiopasdfghj'
      random_name = ''.join((random.choice(sample_string)) for x in range(len(sample_string)))
      random_name1 = ''.join((random.choice(sample_string)) for x in range(len(sample_string)))
      random_name2 = ''.join((random.choice(sample_string)) for x in range(len(sample_string)))
      img_name = f"img_{random_name}.png"
      img_name1 = f"img_{random_name1}.png"
      img_name2 = f"img_{random_name2}.png"
      #tạo đường dẫn để lưu ảnh 
      path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
      path_to_save1 = os.path.join(app.config['UPLOAD_FOLDER'], img_name1)
      path_to_save2 = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
      print(path_to_save) 
      print(path_to_save1)
      print(path_to_save2) 
      #lưu ảnh 
      urllib.request.urlretrieve(image_url,path_to_save)
      urllib.request.urlretrieve(image_url1,path_to_save1)
      urllib.request.urlretrieve(image_url2,path_to_save)
      #lấy tên ảnh đã lưu
      # image_path = "img01.png"
      print("have text")
      return render_template("item3.html", user_image = img_name, user_image1 = img_name1,
                          user_image2 = img_name2, msg="できあがりました。いかがでしょうか？")
  else:
      return render_template("item3.html")
    

