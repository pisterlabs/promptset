import ast
import os

import openai
import xml.etree.ElementTree as ET
import subprocess
import shutil
import time

openai.api_key = ''
def ask_gpt(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    # 解析回答
    answer = response.choices[0].message.content
    return answer

def decompilation(apk_name):
    command = "java -jar apktool_2.9.0.jar d " + apk_name + " -o app"
    # 运行命令行指令
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # 获取命令行输出结果
    output, error = process.communicate()
    # 解码输出结果和错误信息
    # output = output.decode()
    # error = error.decode()
    # 返回输出结果和错误信息
    return output, error

def analyzeOneApk(apk_src,description):
    try:
        # 反编译apk，输出为app文件夹
        decompilation(apk_src)
        # 读取xml
        tree = ET.parse('app/AndroidManifest.xml')
        root = tree.getroot()
        xml_string = ET.tostring(root, encoding='utf-8').decode('utf-8')


        # # 读取游戏介绍（用于测试，现改为直接作为形式参数输入）
        # with open("description.txt", 'r', encoding='utf-8') as file:
        #     description = file.read()

        # 初始化messages
        messages = [
            {"role": "system", "content": "You are an assistant with clear and concise assistant"},

            {"role": "user",
             "content": "Based on the Permissions And Features and game description,  which sensitive permissions and features should not be requested (Simply answer the permissions and features mentioned in the question). \n" \
                        "** All you have to do is analyze what's below: **\n" \
                        "\n```\n" + "Permissions used:- android.permission.INTERNET- android.permission.ACCESS_NETWORK_STATE- android.permission.VIBRATE- android.permission.SYSTEM_ALERT_WINDOW- android.permission.READ_EXTERNAL_STORAGE- android.permission.WRITE_EXTERNAL_STORAGE- android.permission.WRITE_MEDIA_STORAGE- android.permission.WRITE_SETTINGS- android.permission.WRITE_SECURE_SETTINGS- android.permission.CHANGE_CONFIGURATION- android.permission.BLUETOOTH- android.permission.BLUETOOTH_ADMIN- android.permission.INJECT_EVENTS- android.permission.DEVICE_POWER- android.permission.RECORD_AUDIO- android.permission.MODIFY_AUDIO_SETTINGS- android.permission.REORDER_TASKS- android.permission.CHANGE_WIFI_STATE- android.permission.ACCESS_WIFI_STATEFeatures used:- android.hardware.microphone (required: false)- android.hardware.touchscreen (required: false)- android.hardware.touchscreen.multitouch (required: false)- android.hardware.touchscreen.multitouch.distinct (required: false)" + "\n```\n" \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 "\ndescription:\n```\n" + "一场这是一个多人联机游戏。比赛10项运动！■ 丰富的运动项目体验-棒球、射箭、乒乓球、篮球、保龄球、羽毛球、高尔夫、飞镖、台球、拳击■ 多人实时PvP-拳击、棒球和乒乓球还不受支持■ 通过高级物理实现逼真的虚拟运动体验■ 播放器定制■ 简单的用户界面和简单的控制可以帮助任何人玩■ 从初学者到专业人士有5种不同的困难*比赛形式与奥运会官方规则相同，因此你可以学习运动训练和规则。*适用于由于体育设施和天气条件而必须在室内进行的虚拟现实体育教室。" + "\n```\n"},
            {"role": "assistant", "content": "no permissions and features should not be requested."},

            {"role": "user",
             "content": "Based on the Permissions And Features and game description,  which sensitive permissions and features should not be requested (Simply answer the permissions and features mentioned in the question). \n" \
                        "** All you have to do is analyze what's below: **\n" \
                        "\n```\n" + "The permissions used in the decompiled XML file are:- android.permission.INTERNET- android.permission.WRITE_EXTERNAL_STORAGE- android.permission.ACCESS_NETWORK_STATE- android.permission.WAKE_LOCK- com.android.vending.CHECK_LICENSE- android.permission.ACCESS_WIFI_STATE- android.permission.MODIFY_AUDIO_SETTINGS- android.permission.VIBRATE- android.permission.READ_EXTERNAL_STORAGE- android.permission.WRITE_SETTINGS- android.permission.CHANGE_CONFIGURATION- android.permission.BLUETOOTH- android.permission.BLUETOOTH_ADMIN- android.permission.INJECT_EVENTSThe features used in the decompiled XML file are:- android.hardware.touchscreen- android.hardware.touchscreen.multitouch- android.hardware.touchscreen.multitouch.distinct" + "\n```\n" \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               "\ndescription:\n```\n" + "启示骑士 是一个 VR 街机摩托车游戏，您必须在高速荒地公路上占优势，避免敌对交通，并继续生存，飞速行驶和骑行！特点 启示录骑士 - VR自行车赛车游戏 Android的：•20级纯VR肾上腺素•5辆摩托车可提供数十种升级•完整的虚拟现实环境•游戏手柄和运动控制器支持。" + "\n```\n"},
            {"role": "assistant", "content": "no permissions and features should not be requested."},

            {"role": "user",
             "content": "Based on the Permissions And Features and game description,  which sensitive permissions and features should not be requested (Simply answer the permissions and features mentioned in the question). \n" \
                        "** All you have to do is analyze what's below: **\n" \
                        "\n```\n" + "- android.permission.ACCESS_NETWORK_STATE- android.permission.INTERNET- android.permission.VIBRATE- android.permission.WRITE_EXTERNAL_STORAGE- android.permission.READ_EXTERNAL_STORAGE- android.permission.WRITE_SETTINGS- android.permission.CHANGE_CONFIGURATION- android.permission.BLUETOOTH- android.permission.BLUETOOTH_ADMIN- android.permission.INJECT_EVENTS- android.permission.CHANGE_WIFI_MULTICAST_STATE- android.permission.ACCESS_FINE_LOCATION android.permission.CAMERA- android.permission.RECORD_AUDIO- android.permission.MODIFY_AUDIO_SETTINGS- android.permission.WAKE_LOCK- android.permission.REORDER_TASKS android.permission.CHANGE_WIFI_STATE-android.permission.ACCESS_WIFI_STATE- android.permission.CHANGE_NETWORK_STATEThe features used in the decompiled XML file are:- android.hardware.location.gps (required: false)- android.hardware.location (required: false)- android.hardware.camera (required: false)- android.hardware.camera.autofocus (required: false)- android.hardware.camera.front (required: false)- android.hardware.microphone (required: false)- android.hardware.sensor.accelerometer (required: false)- android.hardware.touchscreen (required: false)- android.hardware.touchscreen.multitouch (required: false)- android.hardware.touchscreen.multitouch.distinct (required: false)" + "\n```\n" \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     "\ndescription:\n```\n" + "【游戏名称】：Space Fitness【游戏类型】：运动【游戏平台】：pico【游戏模式】：原生VR游戏（定位控制器）【游戏语言】：多国语言游戏容量】：143MB【游戏介绍】：关于这款游戏在太空中，飞船突发意外，你需要几岁各式各样的陨石来通过不同挑战。《太空健身计划》集健身与游戏于一体，在这个广阔的太空中，你可以享受无限挑战乐趣。" + "\n```\n"},
            {"role": "assistant",
             "content": "android.permission.CAMERA\nandroid.permission.ACCESS_FINE_LOCATION \n- android.hardware.location.gps (required: false)- android.hardware.location (required: false)- android.hardware.camera (required: false)- android.hardware.camera.autofocus (required: false)- android.hardware.camera.front (required: false)- android.hardware.microphone \nandroid.permission.RECORD_AUDIO\n"},

            {"role": "user",
             "content": "Based on the Permissions And Features and game description,  which sensitive permissions and features should not be requested (Simply answer the permissions and features mentioned in the question). \n" \
                        "** All you have to do is analyze what's below: **\n" \
                        "\n```\n" + "Permissions used:- android.permission.ACCESS_NETWORK_STATE- android.permission.INTERNET- android.permission.CHANGE_WIFI_MULTICAST_STATE- android.permission.WRITE_EXTERNAL_STORAGE- android.permission.WRITE_SETTINGS- android.permission.READ_EXTERNAL_STORAGE- android.permission.REORDER_TASKS- android.permission.CHANGE_WIFI_STATE- android.permission.ACCESS_WIFI_STATEFeatures used:- android.hardware.touchscreen (required: false)- android.hardware.touchscreen.multitouch (required: false)- android.hardware.touchscreen.multitouch.distinct (required: false)" + "\n```\n" \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   "\ndescription:\n```\n" + "这个游戏是一款模拟经营一家店的游戏，不过我不是店长。而是打杂的实习生啊！来了顾客要了解他想吃什么，然后自己做出来给顾客吃，手忙脚乱的感觉有木有，终于知道饭店实习生是有多么的辛苦了。还要定期打扫卫生，还有大细菌boss挑战，体验忙碌的一天。这个游戏让我想到了童年时光，记得那时候没有什么玩具，和小伙伴们用泥巴石头还有草树叶什么的当食物，一起玩过家家的游戏。哈哈，这是一款不错的游戏价格也便宜。" + "\n```\n"},
            {"role": "assistant", "content": "no permissions and features should not be requested."},

            # {"role": "user",
            #  "content": "Based on the Permissions And Features and game description,  which sensitive permissions and features should not be requested (Simply answer the permissions and features mentioned in the question). \n" \
            #             "** All you have to do is analyze what's below: **\n" \
            #             "\n```\n" + "The permissions used in the decompiled XML file are:- android.permission.INTERNET- android.permission.CHANGE_WIFI_MULTICAST_STATE- android.permission.RECORD_AUDIO- android.permission.MODIFY_AUDIO_SETTINGS- android.permission.BLUETOOTH- android.permission.WRITE_EXTERNAL_STORAGE- android.permission.WRITE_SETTINGS- android.permission.REORDER_TASKS- android.permission.CHANGE_WIFI_STATE android.permission.ACCESS_NETWORK_STATE- android.permission.ACCESS_WIFI_STATE- android.permission.READ_EXTERNAL_STORAGEThe features used in the decompiled XML file are:- android.hardware.microphone (required: false)- android.hardware.touchscreen (required: false)- android.hardware.touchscreen.multitouch (required: false)- android.hardware.touchscreen.multitouch.distinct (required: false)" + "\n```\n" \
            #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     "\ndescription:\n```\n" + "Cave Digger」游戏背景设定在一个充斥着黑色幽默元素的另类西方世界，玩家将扮演一名掘金者深入地下矿井挖掘宝藏，并不断升级或解锁挖掘工具以提高挖掘效率；续作「Cave Digger 2：Dig Harder」沿用了「Cave Digger」背景，并添加了包括4人合作模式和更豪华柴油朋克世界在内的更多内容，玩家将继续在游戏中挖掘宝藏以及探究隐藏秘密。" + "\n```\n"},
            # {"role": "assistant", "content": "no permissions and features should not be requested."},

        ]

        # 提问xml文件
        question = "Analyze which permissions and features are used by the decompiled xml file below (only need permissions and features):\n"+xml_string
        messages.append({"role": "user", "content": question})
        answer = ask_gpt(messages)
        xmlPermissionAndFeature = answer
        print(answer)
        # 附加回答信息
        # messages.append({"role": "assistant", "content": answer})
        # print("--------------------------------------------------\n")

        # # 提问游戏介绍
        # question = "Take a look at permissions and features that might be used for this game, as described below:\n"+description
        # messages.append({"role": "user", "content": question})
        # answer = ask_gpt(messages)
        # print(answer)
        # # 附加回答信息
        # messages.append({"role": "assistant", "content": answer})
        # print("--------------------------------------------------\n")

        for i in range(21):
            time.sleep(1)
            print("sleep:" + str(i + 1) + "s")

        # 重新开一个提问进行二阶提问，针对问题描述和已经提取好的xml文件
        messages = [
            {"role": "system", "content": "You are an assistant with clear and concise assistant"},
        ]

        # 提问哪些不该用
        question = "Based on the Permissions And Features and game description,  which sensitive permissions and features should not be requested (Simply answer the permissions and features mentioned in the question). \n" \
                   "** All you have to do is analyze what's below: **\n" \
                   "\n```\n"+xmlPermissionAndFeature+"\n```\n" \
                   "\ndescription:\n```\n" + description + "\n```\n"
        print(question)
        messages.append({"role": "user", "content": question})
        answer = ask_gpt(messages)
        print(answer)

        # 删除反编译文件夹
        shutil.rmtree("app")

        return answer,xml_string

    except Exception as e:
        if os.path.exists("app") and os.path.isdir("app"):
            shutil.rmtree("app")
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":

    # apk_src = "D:\\ict\\My_Project\\Security\\reverse\\reverse_app\\gpt_app\\9月的晚餐_SepsDiner_8592_19.apk"
    apk_src = "D:\\ict\\My_Project\\Security\\reverse\\reverse_app\\gpt_app\\9月的晚餐_SepsDiner_8592_19.apk"
    description = "欢迎来到 Sep’s Diner，这是一家由您担任主厨的新汉堡餐厅！它真的会成为镇上最好的汉堡吗？现在就看你了！一层又一层，您的体验将增加，美食客户的数量也会增加。他们很匆忙，所以在他们离开之前尽快满足他们！饥饿和不耐烦，他们不会让您犯错误…… 细心和精确，以获得最大的利润！包括全新的甜点餐厅Sep’s Donut！游戏特色：· 2间餐厅· 包括 3 种游戏模式：定时、轻松、多人· 每家餐厅包括 27 个级别（定时/休闲 12 个，多人游戏 3 个）· 最多 4 名玩家的多人合作游戏· 紧张刺激的关卡！· 身临其境的音频氛围· 不耐烦的客户用有趣的声音· 美丽的风景和彩灯· 逐级增加难度· 超过 30 种不同的汉堡食谱组合！· 煎饼、甜甜圈、华夫饼、纸杯蛋糕、冰淇淋和奶昔！"
    analyzeOneApk(apk_src,description)



