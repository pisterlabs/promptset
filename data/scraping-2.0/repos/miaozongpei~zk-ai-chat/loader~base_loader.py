from langchain.document_loaders import\
    UnstructuredWordDocumentLoader,\
    PyPDFium2Loader,\
    DirectoryLoader,\
    PyPDFLoader,\
    TextLoader,\
    UnstructuredPowerPointLoader,\
    BiliBiliLoader,\
    WebBaseLoader,\
    UnstructuredEmailLoader,\
    OutlookMessageLoader,\
    EverNoteLoader,\
    UnstructuredImageLoader


import pytesseract
from PIL import Image, ImageFilter, ImageEnhance

import pypandoc

from pdfminer.high_level import extract_pages

import pdf2image

import os

import pptx

def load_pdf(file_url):
    print(file_url)
    if file_url.endswith(".pdf"):
        # print the file name
        loader = PyPDFium2Loader(f'{file_url}')
        print(loader)
        return loader.load()

def load_word(file_url):
    print(file_url)
    # check if the file is a doc or docx file
    # 检查所有doc以及docx后缀的文件
    if file_url.endswith(".doc") or file_url.endswith(".docx"):
         # langchain自带功能，加载word文档
        loader = UnstructuredWordDocumentLoader(f'{file_url}')
        return loader.load()

def load_ppt(file_url):
    #import pptx
    #pip3 install python-pptx

    print(file_url)
    if file_url.endswith(".ppt") or file_url.endswith(".pptx"):
        loader = UnstructuredPowerPointLoader(f'{file_url}')
    return loader.load()


def load_txt(file_url):
    print(file_url)
    if file_url.endswith(".txt"):
        loader = TextLoader(f'{file_url}')
        return loader.load()

def load_email(file_url):
    print(file_url)
    if file_url.endswith(".eml"):
        loader = UnstructuredEmailLoader(f'{file_url}')
    if file_url.endswith(".msg"):
        loader = OutlookMessageLoader(f'{file_url}')
    return loader.load()

def load_pypandoc(file_url):
    # !pip install pypandoc
    # import pypandoc
    # pypandoc.download_pandoc()
    print(file_url)
    if file_url.endswith(".enex"):
        loader = EverNoteLoader(f'{file_url}')
    return loader.load()

def load_image(file_url):
    #import pdf2image
    #pip3 install pdfminer.six
    #pip3 install unstructured_inference
    print(file_url)
    if file_url.endswith(".jpg") or file_url.endswith(".png"):
        new_file_url = preprocess_image(f'{file_url}')
        loader = UnstructuredImageLoader(new_file_url, strategy="ocr_only", ocr_languages="chi_sim")

    return loader.load()



def pre_image(file_url):
    img = Image.open(f'{file_url}')

    # 使用Pillow库的Filter模块进行锐化操作。可以使用模糊、边缘增强、锐度增强等滤镜。这里我们使用UnsharpMask滤镜来增强图片的锐度：
    Img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=160, threshold=3))

    # 模式L”为灰色图像，它的每个像素用8个bit表示，0表示黑，255表示白，其他数字表示不同的灰度。
    Img = img.convert('L')

    # 自定义灰度界限，大于这个值为黑色，小于这个值为白色
    threshold = 80
    table = []
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)
    # 图片二值化
    Img = Img.point(table, '1')


    new_file_url = str.replace(file_url, ".jpg", '_pre.jpg')
    if file_url.endswith(".png"):
        new_file_url = str.replace(file_url, ".png", '_pre.png')


    Img.save(new_file_url)
    return new_file_url

def preprocess_image(file_url):
    image = Image.open(f'{file_url}')
    # 图像增强
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)

    # 降噪
    #image = image.filter(ImageFilter.SMOOTH)
    # 使用Pillow库的Filter模块进行锐化操作。可以使用模糊、边缘增强、锐度增强等滤镜。这里我们使用UnsharpMask滤镜来增强图片的锐度：
    image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=4))


    # 二值化
    image = image.convert('L')
    threshold = 125
    #image = image.point(lambda x: 255 if x > threshold else 0)


    new_file_url = str.replace(file_url, ".jpg", '_pre.jpg')
    if file_url.endswith(".png"):
        new_file_url = str.replace(file_url, ".png", '_pre.png')

    image.save(new_file_url)
    return new_file_url




def load_image_pytesseract(file_url):
    print(file_url)
    if file_url.endswith(".jpg") or file_url.endswith(".png"):
        # 打开图像文件
        img = Image.open(f'{file_url}')
        # 读取图像
        result = pytesseract.image_to_string(img, lang='chi_sim')
    return result



def load_file(file_url):
    if file_url.endswith(".txt"):
        return load_txt(file_url)
    if file_url.endswith(".doc") or file_url.endswith(".docx"):
        return load_word(file_url)
    if file_url.endswith(".pdf"):
        return load_pdf(file_url)
    if file_url.endswith(".ppt") or file_url.endswith(".pptx"):
        return load_ppt(file_url)
    if file_url.endswith(".eml") or file_url.endswith(".msg"):
        return load_email(file_url)
    if file_url.endswith(".enex"):
        return load_pypandoc(file_url)
    if file_url.endswith(".jpg") or file_url.endswith(".png"):
        return load_image(file_url)



def load_bilibi(url):
    print(url)
    # pip install bilibili-api-python
    loader = BiliBiliLoader(
        [url]
    )
    return loader.load()


def load_url(url):
    print(url)
    loader = WebBaseLoader([url])
    docs = loader.load()
    data = docs[0].page_content
    data = data.replace("\n",'')
    return data

#data=load_word("/Users/miao/mydocs/个人/公司/政府服务/政府服务数字人脚本（西乌旗）.docx")
#data=load_image("/Users/miao/mydocs/个人/公司/1.jpg")

#data=load_image("/Users/miao/mydocs/个人/公司/111.png")
#data=load_image("/Users/miao/mydocs/个人/z.jpg")
#data=load_image("/Users/miao/mydocs/个人/公司/6年级数学知识点/2.jpg")

#data=load_image("/Users/miao/mydocs/个人/公司/WechatIMG196.jpg")

#data=load_ppt("/Users/miao/mydocs/个人/公司/创影数字人产品介绍0726.pptx")
#data=load_pdf("/Users/miao/mydocs/个人/公司/新产品策划书-获客推广.pdf")
#data=load_txt("/Users/miao/mydocs/个人/公司/demo.txt")
#data=load_url("https://blog.csdn.net/WinkingJay/article/details/80497460")

#loader = BiliBiliLoader(["https://www.bilibili.com/video/BV1xt411o7Xu/"])
#print(loader.load())
#data=load_bilibi("https://www.bilibili.com/video/BV1ek4y1c718/?spm_id_from=333.880.my_history.page.click&vd_source=0cfb224441a3101611c842d6355a0ff3")
#print(data)





