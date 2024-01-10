
import os
import openai
import re  # 导入正则表达式模块
import sys
import time
import json
import docx2txt
import sys
import codecs

import shutil




import os

def convert_resume_to_txt(input_resume_path):
    """
    将简历文件转换为文本格式并保存到 'Text_Conversions' 目录中。
    
    参数:
    - input_resume_path: 简历文件的路径。
    """
    # 基础目录是简历目录的上一级
    base_directory = os.path.dirname(os.path.dirname(input_resume_path))

    # 定义文本转换的输出目录
    text_conversions_directory = os.path.join(base_directory, "Text_Conversions")

    # 确保文本转换目录存在
    os.makedirs(text_conversions_directory, exist_ok=True)

    # 准备输出文本文件路径
    file_name = os.path.basename(input_resume_path)
    output_txt_path = os.path.join(text_conversions_directory, os.path.splitext(file_name)[0] + ".txt")

    # 确定输入文件的类型
    file_extension = os.path.splitext(input_resume_path)[-1].lower()

    # 根据文件类型调用相应的转换函数
    if file_extension == '.docx':
        docx_to_txt(input_resume_path, output_txt_path)
        print("DOCX 文件已成功转换为文本。")

    elif file_extension == '.pdf':
        pdf_to_txt(input_resume_path, output_txt_path)
        print("PDF 文件已成功转换为文本。")

    elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        img_to_txt(input_resume_path, output_txt_path)
        print("图像文件已成功转换为文本。")

    elif file_extension == '.doc':
        doc_to_txt(input_resume_path, output_txt_path)
        print("DOC 文件已成功转换为文本。")

    else:
        raise ValueError("不支持的文件类型进行转换。")




def convert_resume_to_txt_for_gpt_analysis(resume_path, text_conversions_dir="Text_Conversions"):
    # 获取文件扩展名
    _, file_extension = os.path.splitext(resume_path)
    file_extension = file_extension.lower()

    # 构造文本文件的输出路径
    resume_filename_without_extension = os.path.basename(resume_path).split('.')[0]
    
    # 更新这行代码，使其构造在上级目录的 Text_Conversions 下的路径
    resume_txt_path = os.path.join(os.path.dirname(os.path.dirname(resume_path)), text_conversions_dir, resume_filename_without_extension + ".txt")

    # 确保文本转换目录存在
    # os.makedirs(os.path.join(os.path.dirname(resume_path), text_conversions_dir), exist_ok=True)

    # # 根据文件扩展名，调用相应的转换函数
    # if file_extension == '.docx':
    #     docx_to_txt(resume_path, resume_txt_path)
    # elif file_extension == '.pdf':
    #     pdf_to_txt(resume_path, resume_txt_path)
    # elif file_extension == '.doc':
    #     doc_to_txt(resume_path, resume_txt_path)
    # # 如果是图片格式
    # elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
    #     img_to_txt(resume_path, resume_txt_path)
    # else:
    #     raise ValueError(f"Unsupported file type: {file_extension}")

    return resume_txt_path







def docx_to_txt(input_docx_path, output_txt_path):
    # 使用docx2txt库读取docx文件内容
    text = docx2txt.process(input_docx_path)

    # 删除重复行
    lines = text.split('\n')
    unique_lines = []
    for line in lines:
        if line not in unique_lines:
            unique_lines.append(line)
    text = '\n'.join(unique_lines)

    # 将读取到的问题内容写入txt文件
    with open(output_txt_path, 'w', encoding='utf-8') as file:
        file.write(text)

    convert_docx_to_image(input_docx_path)





'''
pip install pdfminer.six
'''

# 导入所需的库

from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage


# from pdfminer.pdfparser import PDFParser, PDFDocument
from pdfminer.pdfdevice import PDFDevice
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LTTextBoxHorizontal, LAParams
# from pdfminer.pdfinterp import PDFTextExtractionNotAllowed
from pdfminer.pdfpage import PDFTextExtractionNotAllowed



# 定义函数，接受输入的PDF文件路径和输出的txt文件路径
def pdf_to_txt(input_pdf_path, output_txt_path):

    open(output_txt_path, 'w').close()  # 清空txt文件

    # 定义内部函数parse，处理PDF文件
    def parse(input_pdf_file, output_txt_file):
        
        # 用文件对象创建一个PDF文档分析器
        parser = PDFParser(input_pdf_file)
        # 创建一个PDF文档
        doc = PDFDocument()
        # 分析器和文档相互连接
        parser.set_document(doc)
        doc.set_parser(parser)
        # 提供初始化密码，没有默认为空
        doc.initialize()

        # 检查文档是否可以转成TXT，如果不可以就忽略
        if not doc.is_extractable:
            raise PDFTextExtractionNotAllowed
        else:
            # 创建PDF资源管理器，来管理共享资源
            rsrcmgr = PDFResourceManager()
            # 创建一个PDF设备对象
            laparams = LAParams()
            # 将资源管理器和设备对象聚合
            device = PDFPageAggregator(rsrcmgr, laparams=laparams)
            # 创建一个PDF解释器对象
            interpreter = PDFPageInterpreter(rsrcmgr, device)

            # 循环遍历列表，每次处理一个page内容
            for page in doc.get_pages():
                interpreter.process_page(page)
                # 接收该页面的LTPage对象
                layout = device.get_result()
                
                for x in layout:
                    try:
                        if isinstance(x, LTTextBoxHorizontal):
                            with open(output_txt_file, 'a', encoding='utf-8-sig') as f:
                                result = x.get_text()
                                # 删除任何前导/尾随的空格
                                result = result.strip()
                                # 如果行不为空，则写入文件
                                if result != '':
                                    f.write(result + "\n")
                    except:
                        print("Failed")

    # 打开并处理PDF文件
    with open(input_pdf_path, 'rb') as pdf_file:
        parse(pdf_file, output_txt_path)

    convert_pdf_to_image(input_pdf_path)




from ecloud import CMSSEcloudOcrClient
import json

accesskey = '4863f884aef84ea4a4af9895285b75ec' 
secretkey = '249b66cddeaa453f8c3689761476b08a'
url = 'https://api-wuxi-1.cmecloud.cn:8443'


def img_to_txt(input_img_path, output_txt_path):
    print("正在从图片转化为txt")
    print(input_img_path)
    print(output_txt_path)
    requesturl = '/api/ocr/v1/webimage'
    try:
        ocr_client = CMSSEcloudOcrClient(accesskey, secretkey, url)
        response = ocr_client.request_ocr_service_file(requestpath=requesturl, imagepath=input_img_path)

        response_json = json.loads(response.text)  # 解析JSON
        words_info = response_json['body']['content']['prism_wordsInfo']  # 取出所有识别出的文字的信息

        with open(output_txt_path, 'w', encoding='utf-8') as file:
            for word_info in words_info:
                file.write(word_info['word'] + '\n')  # 将识别出的文字写入到文件中
    except ValueError as e:
        print(e)
    return output_txt_path


import os
import win32com.client as win32
from docx import Document

import os
import tempfile

def doc_to_txt(input_doc_path, output_txt_path):
    # 生成临时的图片路径
    output_image_path = tempfile.mktemp(suffix='.png')

    # 将.doc文档转为图片
    convert_doc_to_image(input_doc_path, output_image_path)

    # 将图片转为.txt文件
    img_to_txt(output_image_path, output_txt_path)

    # 删除临时的图片文件
    os.remove(output_image_path)

    convert_doc_to_image(input_doc_path)










import os
import tempfile
from docx2pdf import convert
from PIL import Image
import fitz
import time

import shutil

def convert_docx_to_image(word_path):

    # 提取文件基础名和目录
    base_name = os.path.splitext(os.path.basename(word_path))[0]
    directory = os.path.dirname(word_path)
    image_name = os.path.join(directory, base_name + '.jpg')


    # 创建一个临时副本
    temp_word_path = tempfile.mktemp(suffix='.docx')
    shutil.copy(word_path, temp_word_path)

    # 将 Word 文档转换为 PDF
    pdf_path = tempfile.mktemp(suffix='.pdf')
    convert(temp_word_path, pdf_path)

    # 将 PDF 转换为图像
    pdfDoc = fitz.open(pdf_path)
    images = []
    for pg in range(pdfDoc.page_count):
        page = pdfDoc[pg]
        zoom_x = 1.33333333
        zoom_y = 1.33333333
        mat = fitz.Matrix(zoom_x, zoom_y)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        mode = "RGBA" if pix.alpha else "RGB"
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        images.append(img)

    # 关闭 PyMuPDF 对 PDF 文件的引用
    pdfDoc.close()

    # 清理生成的 PDF 文件和临时 Word 副本
    os.remove(pdf_path)
    os.remove(temp_word_path)

    # 合并所有的图像
    widths, heights = zip(*(i.size for i in images))
    total_width = max(widths)
    total_height = sum(heights)

    new_img = Image.new('RGB', (total_width, total_height))

    y_offset = 0
    for img in images:
        new_img.paste(img, (0, y_offset))
        y_offset += img.height

    # 保存图像
    new_img.save(image_name)





import os
from PIL import Image
import fitz

def convert_pdf_to_image(pdf_path):

   # 提取文件基础名和目录
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    directory = os.path.dirname(pdf_path)
    image_name = os.path.join(directory, base_name + '.jpg')

    # 打开 PDF 文件
    pdfDoc = fitz.open(pdf_path)

    # 将 PDF 转换为图像
    images = []
    for pg in range(pdfDoc.page_count):
        page = pdfDoc[pg]
        zoom_x = 1.33333333
        zoom_y = 1.33333333
        mat = fitz.Matrix(zoom_x, zoom_y)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        mode = "RGBA" if pix.alpha else "RGB"
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        images.append(img)

    # 关闭 PyMuPDF 对 PDF 文件的引用
    pdfDoc.close()

    # 合并所有的图像
    widths, heights = zip(*(i.size for i in images))
    total_width = max(widths)
    total_height = sum(heights)

    new_img = Image.new('RGB', (total_width, total_height))

    y_offset = 0
    for img in images:
        new_img.paste(img, (0, y_offset))
        y_offset += img.height

    # 保存图像
    new_img.save(image_name)






import os
import tempfile
import win32com.client as win32
from docx import Document
from docx2pdf import convert
from PIL import Image
import fitz

def convert_doc_to_image(input_doc_path):


   # 提取文件基础名和目录
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    directory = os.path.dirname(pdf_path)
    image_name = os.path.join(directory, base_name + '.jpg')

    # 初始化 Word 对象
    word = win32.gencache.EnsureDispatch('Word.Application')

    # 隐藏 Word 程序窗口
    word.Visible = False

    # 打开 .doc 文件
    doc = word.Documents.Open(input_doc_path)

    # 将 .doc 文件另存为 .docx 文件
    docx_path = os.path.splitext(input_doc_path)[0] + ".docx"
    doc.SaveAs(docx_path, FileFormat=16)  # 16 表示 .docx 文件格式

    # 关闭原始 .doc 文件
    doc.Close()

    # 将 .docx 文件转换为 PDF
    pdf_path = tempfile.mktemp(suffix='.pdf')
    convert(docx_path, pdf_path)

    # 将 PDF 转换为图像
    pdfDoc = fitz.open(pdf_path)
    images = []
    for pg in range(pdfDoc.page_count):
        page = pdfDoc[pg]
        zoom_x = 1.33333333
        zoom_y = 1.33333333
        mat = fitz.Matrix(zoom_x, zoom_y)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        mode = "RGBA" if pix.alpha else "RGB"
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        images.append(img)

    # 关闭 PyMuPDF 对 PDF 文件的引用
    pdfDoc.close()

    # 清理生成的 PDF 文件
    os.remove(pdf_path)

    # 合并所有的图像
    widths, heights = zip(*(i.size for i in images))
    total_width = max(widths)
    total_height = sum(heights)

    new_img = Image.new('RGB', (total_width, total_height))

    y_offset = 0
    for img in images:
        new_img.paste(img, (0, y_offset))
        y_offset += img.height

    # 保存图像
    new_img.save(image_name)









def convert_to_jpg(image_path):
    # 检查文件是否为 JPEG 或 PNG
    if not image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        print("文件不是 JPEG 或 PNG 格式。")
        return

    # 读取图片
    img = Image.open(image_path)

    # 构建新的 JPG 文件名
    jpg_path = os.path.splitext(image_path)[0] + '.jpg'

    # 保存图片为 JPG 格式
    img.convert('RGB').save(jpg_path, 'JPEG')
    print(f"已将 {image_path} 转换为 {jpg_path}")




