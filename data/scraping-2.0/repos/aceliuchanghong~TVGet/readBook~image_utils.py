import os
import shutil
from os.path import exists
from crawl.spiderDealer.checkPath import check
from readBook.PicResult import PicResult, PicInfo
import subprocess
import re
import json
from PIL import Image
import httpx
from openai import OpenAI


def get_image_size(picPath):
    image_path = picPath
    # 构建ffprobe命令
    command = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'json',
        image_path
    ]
    picInfo = PicInfo()
    match = r'([^/]+)\.(png|jpg|jpeg|gif|bmp)$'
    if re.search(match, image_path):
        picInfo.name = re.search(match, image_path).group(1)
        picInfo.ext = re.search(match, image_path).group(2)
    try:
        # 运行ffprobe命令
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # 解析输出结果
        if result.returncode == 0:
            output = result.stdout
            info = json.loads(output)
            width = info['streams'][0]['width']
            height = info['streams'][0]['height']
            picInfo.width = width
            picInfo.height = height
            return picInfo
        else:
            raise Exception('ffprobe error: ' + result.stderr.decode('utf-8'))
    except Exception as e:
        print(e)
        return picInfo


def blur_bg_image2(input_image_path, output_image_path, blur_strength=5, width=1280, height=1707, re_run=False):
    picInfo = get_image_size(input_image_path)
    try:
        # 判断是否需要调整图像尺寸
        if picInfo.width < width or picInfo.height < height:
            if picInfo.width >= picInfo.height:
                newHight = picInfo.height
                newWidth = newHight * width / height
            else:
                newWidth = picInfo.width
                newHight = newWidth * height / width
            vf_filter = f'boxblur={blur_strength},crop={newWidth}:{newHight}'
        else:
            vf_filter = f'boxblur={blur_strength},crop={width}:{height}'

        # 构建ffmpeg命令
        command = [
            'ffmpeg',
            '-i', input_image_path,  # 输入图片文件
            '-vf', vf_filter,
            '-y',  # 覆盖输出文件（如果已经存在）
            output_image_path  # 输出图片文件
        ]

        # 执行命令
        if not exists(output_image_path) or re_run:
            subprocess.run(command, check=True)
        return output_image_path
    except Exception as e:
        print(e)
        return "ERR:BLUR"


def blur_bg_image(input_image_path, output_image_path, blur_strength=5, width=1280, height=1707, re_run=False):
    picInfo = get_image_size(input_image_path)
    try:
        # 判断是否需要调整图像尺寸
        if picInfo.width < width or picInfo.height < height:
            # 计算新的宽高比例
            aspect_ratio = width / height
            pic_aspect_ratio = picInfo.width / picInfo.height

            if pic_aspect_ratio > aspect_ratio:
                newHeight = picInfo.height
                newWidth = int(newHeight * aspect_ratio)  # 确保是整数
            else:
                newWidth = picInfo.width
                newHeight = int(newWidth / aspect_ratio)  # 确保是整数
            vf_filter = f'boxblur={blur_strength},crop={newWidth}:{newHeight}'
        else:
            vf_filter = f'boxblur={blur_strength},crop={width}:{height}'

        # 构建ffmpeg命令
        command = [
            'ffmpeg',
            '-i', input_image_path,  # 输入图片文件
            '-vf', vf_filter,
            '-y',  # 覆盖输出文件（如果已经存在）
            output_image_path  # 输出图片文件
        ]

        # 执行命令
        if not exists(output_image_path) or re_run:
            subprocess.run(command, check=True)
        return output_image_path
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg command failed: {e}")
        return "ERR:BLUR"
    except Exception as e:
        print(f"Unexpected error: {e}")
        return "ERR:BLUR2"


def resize_image_proportionally(input_image_path, output_image_path, scale_factor=0.5, re_run=False):
    try:
        # 构建ffmpeg命令
        command = [
            'ffmpeg',
            '-i', input_image_path,  # 输入图片文件
            '-vf', f'scale=iw*{scale_factor}:ih*{scale_factor}',
            '-y',  # 覆盖输出文件（如果已经存在）
            output_image_path  # 输出图片文件
        ]
        # 执行命令
        if not exists(output_image_path) or re_run:
            subprocess.run(command, check=True)
        return output_image_path
    except Exception as e:
        print(e)
        return "ERR:resize"


def merge_images(input_image_path, output_image_path, background_image, smallPicCenterAxes=(0, 0), re_run=False):
    try:
        picinfo = get_image_size(input_image_path)
        # 构建ffmpeg命令
        command = [
            'ffmpeg',
            '-i', background_image,  # 输入图片文件
            '-i', input_image_path,  # 输入图片文件
            '-filter_complex',
            f'[1]scale={picinfo.width}:{picinfo.height}[small];[0][small]overlay={smallPicCenterAxes[0]}:{smallPicCenterAxes[1]}',
            '-y',  # 覆盖输出文件（如果已经存在）
            output_image_path  # 输出图片文件
        ]

        # 执行命令
        if not exists(output_image_path) or re_run:
            subprocess.run(command, check=True)
        return output_image_path
    except Exception as e:
        print(e)
        return "ERR:merge"


def calculate_position_and_scale2(input_image_info, background_image_info, scale_ratio, x_shift_ratio=0.0):
    # 确保scale_ratio在0到1之间
    scale_ratio = max(0, min(scale_ratio, 1))

    # 确保x_shift_ratio在-1到1之间
    x_shift_ratio = max(-1.0, min(x_shift_ratio, 1.0))

    # 计算基础的缩放因子
    width_scale = background_image_info.width / input_image_info.width
    height_scale = background_image_info.height / input_image_info.height
    base_scale_factor = min(width_scale, height_scale)

    # 应用scale_ratio来调整缩放因子
    scale_factor = base_scale_factor * scale_ratio

    # 计算新的图片尺寸
    new_input_image_width = input_image_info.width * scale_factor
    new_input_image_height = input_image_info.height * scale_factor

    # 计算新的图片应该放置的位置
    # xAxis现在包括x_shift_ratio的影响
    xAxis = ((background_image_info.width - new_input_image_width) / 2.0) + (
            x_shift_ratio * (background_image_info.width - new_input_image_width) / 2.0)
    yAxis = (background_image_info.height - new_input_image_height) / 2.0

    return xAxis, yAxis, scale_factor, new_input_image_width, new_input_image_height


def calculate_position_and_scale3(input_image_info, background_image_info, scale_ratio=1.0, x_shift_ratio=0.0,
                                  y_shift_ratio=0.0, vertical_position='top'):
    # 确保scale_ratio在0到1之间,大小放缩
    scale_ratio = max(0.0, min(scale_ratio, 1.0))

    # 确保x_shift_ratio在-1到1之间 x轴移动
    x_shift_ratio = max(-1.0, min(x_shift_ratio, 1.0))

    # 确保y_shift_ratio在-1到1之间 y轴移动
    y_shift_ratio = max(-1.0, min(y_shift_ratio, 1.0))

    # 计算基础的缩放因子
    width_scale = background_image_info.width / input_image_info.width
    height_scale = (background_image_info.height / 2) / input_image_info.height  # 由于图片只占据上半部分或下半部分，因此高度比例要除以2
    base_scale_factor = min(width_scale, height_scale)

    # 应用scale_ratio来调整缩放因子
    scale_factor = base_scale_factor * scale_ratio

    # 计算新的图片尺寸
    new_input_image_width = input_image_info.width * scale_factor
    new_input_image_height = input_image_info.height * scale_factor

    # 计算新的图片应该放置的位置，包括x轴和y轴的偏移
    xAxis = ((background_image_info.width - new_input_image_width) / 2.0) + (
            x_shift_ratio * (background_image_info.width - new_input_image_width) / 2.0)

    # 根据vertical_position参数来决定yAxis的基本位置
    if vertical_position == 'top':
        base_yAxis = 0  # 图片放在上半部分，因此y轴的基本坐标为0
    elif vertical_position == 'bottom':
        base_yAxis = background_image_info.height / 2  # 图片放在下半部分，因此y轴的基本坐标为背景图片高度的一半
    else:
        raise ValueError("vertical_position must be 'top' or 'bottom'")

    # 应用y_shift_ratio来调整y轴的偏移
    yAxis = base_yAxis + (y_shift_ratio * (background_image_info.height / 2 - new_input_image_height) / 2.0)

    return xAxis, yAxis, scale_factor, new_input_image_width, new_input_image_height


def calculate_position_and_scale(input_image_info, background_image_info, debug=False):
    width_scale = background_image_info.width / input_image_info.width
    height_scale = background_image_info.height / input_image_info.height

    scale_factor = min(width_scale, height_scale)

    new_input_image_width = input_image_info.width * scale_factor
    new_input_image_height = input_image_info.height * scale_factor

    xAxis = (background_image_info.width - new_input_image_width) / 2.0
    yAxis = (background_image_info.height - new_input_image_height) / 2.0

    if debug:
        print(f"Scale factor: {scale_factor}")
        print(f"New image size: {new_input_image_width}x{new_input_image_height}")
        print(f"Position: {xAxis}, {yAxis}")

    return xAxis, yAxis, scale_factor, new_input_image_width, new_input_image_height


def calculate_center_coords(image_path, number_width, number_height):
    """ 计算中心坐标 """
    width, height = get_image_size(image_path).width, get_image_size(image_path).height
    center_coords = (width / number_width, height / number_height)
    return center_coords


def escape_ffmpeg_text(text):
    # 对ffmpeg特殊字符进行转义
    return text.replace(':', '\\:').replace("'", "\\'")


def put_words_on_image(words, input_image_path, output_image_path, center_coords=(0, 0), fontfile="my.ttf",
                       fontcolor='ffffff', fontsize=24, re_run=False, alpha=0.75):
    try:
        # 转义文本
        escaped_words = escape_ffmpeg_text(words)

        # 转换透明度为16进制
        alpha_hex = format(int(alpha * 255), '02x')

        # 构建ffmpeg命令
        command = [
            'ffmpeg',
            '-i', input_image_path,
            '-vf',
            f"drawtext=fontfile={fontfile}:text='{escaped_words}':fontcolor={fontcolor}{alpha_hex}:fontsize={fontsize}:x={center_coords[0]}:y={center_coords[1]}",
            '-y',
            output_image_path
        ]

        # 执行命令
        if not exists(output_image_path) or re_run:
            subprocess.run(command, check=True)
        return output_image_path
    except Exception as e:
        print(e)
        return "ERR:WORDS"


def copy_file(source_path, destination_path, re_run=False):
    if os.path.exists(destination_path) and not re_run:
        return destination_path
    elif not os.path.exists(source_path):
        print("目标文件不存在:" + source_path)
        return "ERR:copy_file"
    else:
        shutil.copy(source_path, destination_path)
        return destination_path


def cut_image(input_image_path, output_image_path, width, height, center_coords=(0, 0), re_run=False):
    try:
        # 打开输入图片
        with Image.open(input_image_path) as img:
            img_width, img_height = img.size

            # 计算裁剪区域的起始点
            start_x = center_coords[0]
            start_y = center_coords[1]

            # 确保裁剪区域不会超出图片的边界
            end_x = min(start_x + width, img_width)
            end_y = min(start_y + height, img_height)

            # 裁剪图片
            crop_area = (start_x, start_y, end_x, end_y)
            cropped_img = img.crop(crop_area)

            # 如果输出路径不存在或者指定了重新运行，则保存裁剪后的图片
            if not os.path.exists(output_image_path) or re_run:
                cropped_img.save(output_image_path)

        return output_image_path
    except Exception as e:
        print(f"An error occurred: {e}")
        return "ERR:cut"


def fill_image(input_image_path, background_image_path, width=0, height=0, center_coords=(0, 0),
               re_run=False, debug=False):
    try:
        # 获取长宽
        input_image_info = get_image_size(input_image_path)
        background_image_info = get_image_size(background_image_path)
        """一共4种情况
        1.外面框架高宽均大于或等于下载图片
        2.外面框架高宽均小于下载图片
        3.外面框架宽大于或等于,高小于下载图片
        4.外面框架高大于或等于,宽小于下载图片
        """
        xAxis, yAxis, scale_factor, new_input_image_width, new_input_image_height = calculate_position_and_scale(
            input_image_info, background_image_info, debug)

        resize_image_path = "../crawl/files/redbook/resize_pic"
        check(resize_image_path)
        merge_pic_path = "../crawl/files/redbook/merge_pic"
        check(merge_pic_path)
        cut_pic_path = "../crawl/files/redbook/cut_pic"
        check(cut_pic_path)

        output_image_path = resize_image_proportionally(input_image_path,
                                                        resize_image_path + "/resize." + input_image_info.name + "." + background_image_info.name + "." + input_image_info.ext,
                                                        scale_factor,
                                                        re_run=re_run)

        output_image_path = merge_images(output_image_path,
                                         merge_pic_path + "/merge." + input_image_info.name + "." + background_image_info.name + "." + input_image_info.ext,
                                         background_image_path,
                                         smallPicCenterAxes=(xAxis, yAxis),
                                         re_run=re_run)

        output_image_path = cut_image(output_image_path,
                                      cut_pic_path + "/cut." + input_image_info.name + "." + background_image_info.name + "." + input_image_info.ext,
                                      new_input_image_width, new_input_image_height,
                                      center_coords=(xAxis, yAxis),
                                      re_run=re_run)

        return output_image_path
    except Exception as e:
        print(f"An error occurred: {e}")
        return "ERR:fill"


# 图片填充,左边input_image_path_left,右边input_image_path_right,调整2图合适长宽,满足各占一半,一左一右
def fill_image_model1(input_image_path_left, input_image_path_right, background_image_path, re_run):
    try:
        # 获取长宽
        input_image_info_left = get_image_size(input_image_path_left)
        input_image_info_right = get_image_size(input_image_path_right)
        background_image_info = get_image_size(background_image_path)

        xAxis_left, yAxis_left, scale_factor_left, new_input_image_width_left, new_input_image_height_left = calculate_position_and_scale2(
            input_image_info_left, background_image_info, 1)
        xAxis_right, yAxis_right, scale_factor_right, new_input_image_width_right, new_input_image_height_right = calculate_position_and_scale2(
            input_image_info_right, background_image_info, 0.85)

        resize_image_path = "../crawl/files/redbook/resize_pic"
        check(resize_image_path)
        merge_pic_path = "../crawl/files/redbook/merge_pic"
        check(merge_pic_path)
        cut_pic_path = "../crawl/files/redbook/cut_pic"
        check(cut_pic_path)

        output_image_path_left = resize_image_proportionally(input_image_path_left,
                                                             resize_image_path + "/resize.left." + input_image_info_left.name + "." + input_image_info_left.ext,
                                                             scale_factor_left,
                                                             re_run=re_run)

        output_image_path_left = merge_images(output_image_path_left,
                                              merge_pic_path + "/merge.left." + input_image_info_left.name + "." + input_image_info_left.ext,
                                              background_image_path,
                                              smallPicCenterAxes=(xAxis_left, yAxis_left),
                                              re_run=re_run)
        output_image_path_right = resize_image_proportionally(input_image_path_right,
                                                              resize_image_path + "/resize.right." + input_image_info_right.name + "." + input_image_info_right.ext,
                                                              scale_factor_right,
                                                              re_run=re_run)
        output_image_path_right = merge_images(output_image_path_right,
                                               merge_pic_path + "/merge.right." + input_image_info_right.name + "." + input_image_info_right.ext,
                                               output_image_path_left,
                                               smallPicCenterAxes=(xAxis_right, yAxis_right),
                                               re_run=re_run)

        return output_image_path_right
    except Exception as e:
        print(f"An error occurred: {e}")
        return "ERR:fill_image_model1"


def get_gpt_response(prompt, picFile, re_run):
    # 配置代理服务器
    proxyHost = "127.0.0.1"
    proxyPort = 10809
    if re_run or not exists(picFile):
        # 创建 OpenAI 客户端并配置代理
        client = OpenAI(http_client=httpx.Client(proxies=f"http://{proxyHost}:{proxyPort}"))
        client.api_key = os.getenv("OPENAI_API_KEY")
        try:
            # 使用 OpenAI GPT-4 API 获取回复
            completion = client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[
                    {"role": "user",
                     "content": prompt
                     }
                ]
            )
            # 返回回复的内容
            return completion.choices[0].message.content
        except Exception as e:
            # 如果有错误发生，打印错误信息
            print(f"An error occurred: {e}")
            return "Her grace is like a serene dawn, casting a gentle glow that captivates and enchants the soul."
    else:
        return "She possesses an ethereal beauty, a timeless elegance that whispers softly to the heart, yet echoes profoundly."


def get_gpt_response2(prompt, picFile, re_run):
    # 配置代理服务器
    proxyHost = "127.0.0.1"
    proxyPort = 10809
    if re_run or not exists(picFile):
        # 创建 OpenAI 客户端并配置代理
        client = OpenAI(http_client=httpx.Client(proxies=f"http://{proxyHost}:{proxyPort}"))
        client.api_key = os.getenv("OPENAI_API_KEY")
        try:
            # 使用 OpenAI GPT-4 API 获取回复
            completion = client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[
                    {"role": "user",
                     "content": prompt
                     }
                ]
            )
            # 返回回复的内容
            return completion.choices[0].message.content
        except Exception as e:
            # 如果有错误发生，打印错误信息
            print(f"An error occurred: {e}")
            return "云发皓齿月初圆"
    else:
        return "玉立花容月下生"


def fill_image_model2(input_image_path, background_image_path, x_shift_ratio=0, re_run=False):
    try:
        # 获取长宽
        input_image_info_left = get_image_size(input_image_path)
        background_image_info = get_image_size(background_image_path)

        xAxis_left, yAxis_left, scale_factor_left, new_input_image_width_left, new_input_image_height_left = calculate_position_and_scale2(
            input_image_info_left, background_image_info, 1, x_shift_ratio=x_shift_ratio)

        resize_image_path = "../crawl/files/redbook/resize_pic"
        check(resize_image_path)
        merge_pic_path = "../crawl/files/redbook/merge_pic"
        check(merge_pic_path)
        cut_pic_path = "../crawl/files/redbook/cut_pic"
        check(cut_pic_path)
        words_image_path = "../crawl/files/redbook/words_pic"
        check(words_image_path)
        tmp_files = merge_pic_path + "/merge.model2." + input_image_info_left.name + "." + input_image_info_left.ext
        ans_files = words_image_path + "/words.model2.1" + input_image_info_left.name + "." + input_image_info_left.ext
        ans_files2 = words_image_path + "/words.model2.2." + input_image_info_left.name + "." + input_image_info_left.ext
        ans_files3 = words_image_path + "/words.model2.3." + input_image_info_left.name + "." + input_image_info_left.ext
        ans_files4 = words_image_path + "/words.model2.4." + input_image_info_left.name + "." + input_image_info_left.ext
        ans_files5 = words_image_path + "/words.model2.5." + input_image_info_left.name + "." + input_image_info_left.ext
        # words = get_gpt_response("给我一段形容女子美丽的英语句子,要求文雅,字数在15-25个单词内", ans_files, re_run)
        from readBook import the_list
        import random
        words = random.choice(the_list.sentence_lists)
        output_image_path_left = resize_image_proportionally(input_image_path,
                                                             resize_image_path + "/resize.model3." + input_image_info_left.name + "." + input_image_info_left.ext,
                                                             scale_factor_left,
                                                             re_run=re_run)

        output_image_path = merge_images(output_image_path_left,
                                         tmp_files,
                                         background_image_path,
                                         smallPicCenterAxes=(xAxis_left, yAxis_left),
                                         re_run=re_run)

        output_image_path = put_words_on_image(words=words[0:36],
                                               input_image_path=output_image_path,
                                               output_image_path=ans_files,
                                               fontcolor="ffffff",
                                               fontfile="BiLuoSiJianHeLuoQingSong-2.ttf",
                                               fontsize=50,
                                               center_coords=(20, 700), re_run=re_run, alpha=0.9)
        output_image_path = put_words_on_image(words=words[36:72],
                                               input_image_path=output_image_path,
                                               output_image_path=ans_files2,
                                               fontcolor="ffffff",
                                               fontfile="BiLuoSiJianHeLuoQingSong-2.ttf",
                                               fontsize=50,
                                               center_coords=(20, 760), re_run=re_run, alpha=0.9)
        output_image_path = put_words_on_image(words=words[72:108],
                                               input_image_path=output_image_path,
                                               output_image_path=ans_files3,
                                               fontcolor="ffffff",
                                               fontfile="BiLuoSiJianHeLuoQingSong-2.ttf",
                                               fontsize=50,
                                               center_coords=(20, 820), re_run=re_run, alpha=0.9)
        output_image_path = put_words_on_image(words=words[108:1000],
                                               input_image_path=output_image_path,
                                               output_image_path=ans_files4,
                                               fontcolor="ffffff",
                                               fontfile="BiLuoSiJianHeLuoQingSong-2.ttf",
                                               fontsize=50,
                                               center_coords=(20, 880), re_run=re_run, alpha=0.9)

        with open('uploaded.log', 'r') as file:
            # 读取所有行到一个列表中
            lines = len(file.readlines())
            file.close()

        output_image_path = put_words_on_image(words="#" + str(lines),
                                               input_image_path=output_image_path,
                                               output_image_path=ans_files5,
                                               fontcolor="ffffff",
                                               fontfile="Bo Le Locust Tree Handwriting Pen Chinese Font-Simplified Chinese Fonts.ttf",
                                               fontsize=60,
                                               center_coords=(640, 40), re_run=re_run, alpha=0.55)
        return output_image_path, words, str(lines)
    except Exception as e:
        print(f"An error occurred: {e}")
        return "ERR:fill_image_model3"


def fill_image_model3(input_image_path_up, input_image_path_down, background_image_path, words="", re_run=False):
    try:
        # 获取长宽
        input_image_info_up = get_image_size(input_image_path_up)
        input_image_info_down = get_image_size(input_image_path_down)
        background_image_info = get_image_size(background_image_path)

        xAxis_left, yAxis_left, scale_factor_left, new_input_image_width_left, new_input_image_height_left = calculate_position_and_scale3(
            input_image_info_up, background_image_info, scale_ratio=0.75, vertical_position='top')
        xAxis_right, yAxis_right, scale_factor_right, new_input_image_width_right, new_input_image_height_right = calculate_position_and_scale3(
            input_image_info_down, background_image_info, y_shift_ratio=1.0, vertical_position='bottom')

        resize_image_path = "../crawl/files/redbook/resize_pic"
        check(resize_image_path)
        merge_pic_path = "../crawl/files/redbook/merge_pic"
        check(merge_pic_path)
        cut_pic_path = "../crawl/files/redbook/cut_pic"
        check(cut_pic_path)

        output_image_path_up = resize_image_proportionally(input_image_path_up,
                                                           resize_image_path + "/resize.up." + input_image_info_up.name + "." + input_image_info_up.ext,
                                                           scale_factor_left,
                                                           re_run=re_run)

        output_image_path_up = merge_images(output_image_path_up,
                                            merge_pic_path + "/merge.up." + input_image_info_up.name + "." + input_image_info_up.ext,
                                            background_image_path,
                                            smallPicCenterAxes=(xAxis_left, yAxis_left),
                                            re_run=re_run)
        output_image_path_down = resize_image_proportionally(input_image_path_down,
                                                             resize_image_path + "/resize.bottom." + input_image_info_down.name + "." + input_image_info_down.ext,
                                                             scale_factor_right,
                                                             re_run=re_run)
        output_image_path_down = merge_images(output_image_path_down,
                                              merge_pic_path + "/merge.bottom." + input_image_info_down.name + "." + input_image_info_down.ext,
                                              output_image_path_up,
                                              smallPicCenterAxes=(xAxis_right, yAxis_right),
                                              re_run=re_run)

        words_image_path = "../crawl/files/redbook/words_pic"
        check(words_image_path)
        words = words
        ans_files = words_image_path + "/words.model3.1." + input_image_info_down.name + "." + input_image_info_down.ext
        ans_files2 = words_image_path + "/words.model3.2." + input_image_info_down.name + "." + input_image_info_down.ext
        ans_files3 = words_image_path + "/words.model3.3." + input_image_info_down.name + "." + input_image_info_down.ext
        ans_files4 = words_image_path + "/words.model3.4." + input_image_info_down.name + "." + input_image_info_down.ext
        ans_files5 = words_image_path + "/words.model3.5." + input_image_info_down.name + "." + input_image_info_down.ext

        output_image_path = put_words_on_image(words=words[0:36],
                                               input_image_path=output_image_path_down,
                                               output_image_path=ans_files,
                                               fontcolor="000000",
                                               fontfile="my.ttf",
                                               fontsize=40,
                                               center_coords=(95, 400), re_run=re_run, alpha=1.0)
        output_image_path = put_words_on_image(words=words[36:72],
                                               input_image_path=output_image_path,
                                               output_image_path=ans_files2,
                                               fontcolor="000000",
                                               fontfile="my.ttf",
                                               fontsize=40,
                                               center_coords=(95, 440), re_run=re_run, alpha=1.0)
        output_image_path = put_words_on_image(words=words[72:108],
                                               input_image_path=output_image_path,
                                               output_image_path=ans_files3,
                                               fontcolor="000000",
                                               fontfile="my.ttf",
                                               fontsize=40,
                                               center_coords=(95, 480), re_run=re_run, alpha=1.0)
        output_image_path = put_words_on_image(words=words[108:1000],
                                               input_image_path=output_image_path,
                                               output_image_path=ans_files4,
                                               fontcolor="000000",
                                               fontfile="my.ttf",
                                               fontsize=40,
                                               center_coords=(95, 520), re_run=re_run, alpha=1.0)
        with open('uploaded.log', 'r') as file:
            # 读取所有行到一个列表中
            lines = len(file.readlines())
            file.close()

        output_image_path = put_words_on_image(words="#" + str(lines),
                                               input_image_path=output_image_path,
                                               output_image_path=ans_files5,
                                               fontcolor="ffffff",
                                               fontfile="Bo Le Locust Tree Handwriting Pen Chinese Font-Simplified Chinese Fonts.ttf",
                                               fontsize=60,
                                               center_coords=(670, 40), re_run=re_run, alpha=0.55)
        return output_image_path
    except Exception as e:
        print(f"An error occurred: {e}")
        return "ERR:fill_image_model1"


if __name__ == '__main__':
    picresult = PicResult()
    picresult.name = "stevenbills_silky._flowing._Smokey._Gloomy._sharp._cenobite._h_a2b8dcc2-9016-4093-a25c-3fb62ce17cd8.png"
    picresult.ext = "png"
    picresult.date = "20231213"
    picresult.keyword = "stevenbills_silky._flowing._Smokey._Gloomy._sharp._cenobite._h_a2b8dcc2-9016-4093-a25c-3fb62ce17cd8"
    picresult.url = "https://cdn.discordapp.com/attachments/1054958023698825266/1181353473258827908/stevenbills_silky._flowing._Smokey._Gloomy._sharp._cenobite._h_a2b8dcc2-9016-4093-a25c-3fb62ce17cd8.png"
    picresult.downpath = "../crawl/files/redbook/original_pic/stevenbills_silky._flowing._Smokey._Gloomy._sharp._cenobite._h_a2b8dcc2-9016-4093-a25c-3fb62ce17cd8.png"
    picresult.bakpath = "../crawl/files/redbook/original_bak_pic/stevenbills_silky._flowing._Smokey._Gloomy._sharp._cenobite._h_a2b8dcc2-9016-4093-a25c-3fb62ce17cd8.png"
    picresult.fix1path = "../crawl/files/redbook/blur_pic/stevenbills_silky._flowing._Smokey._Gloomy._sharp._cenobite._h_a2b8dcc2-9016-4093-a25c-3fb62ce17cd8.png"
    picresult.fix2path = "../crawl/files/redbook/fix_pic/fix_merge_android.stevenbills_silky._flowing._Smokey._Gloomy._sharp._cenobite._h_a2b8dcc2-9016-4093-a25c-3fb62ce17cd8.png"
    picresult.fix3path = "../crawl/files/redbook/fix_pic/fix_cut_android_cut.stevenbills_silky._flowing._Smokey._Gloomy._sharp._cenobite._h_a2b8dcc2-9016-4093-a25c-3fb62ce17cd8.png"
    picresult.anspath = "None"
    picresult.describe = "SUC"

    print(get_gpt_response("给我一段形容女子美丽的英语句子,要求文雅,字数在15-25个单词内",
                           "../crawl/files/redbook/cut_pic/cut.iphone_ok.stevenbills_silky._flowing._Smokey._Gloomy._sharp._cenobite._h_a2b8dcc2-9016-4093-a25c-3fb62ce17cd8.png",
                           False))
