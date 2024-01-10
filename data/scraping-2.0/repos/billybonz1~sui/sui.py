import pyautogui
from datetime import datetime
import json
import requests
import asyncio
import os
import openai
import json
from types import SimpleNamespace
import random

data = [
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0]
]

def move_left(board):
    new_board = []
    for row in board:
        merged_row = []
        last_merged = False
        for value in row:
            if value != 0:
                if merged_row and not last_merged and merged_row[-1] == value:
                    merged_row[-1] *= 2
                    last_merged = True
                else:
                    merged_row.append(value)
                    last_merged = False
        merged_row += [0] * (len(row) - len(merged_row))
        new_board.append(merged_row)
    return new_board

def move_right(board):
    new_board = []
    for row in board:
        merged_row = []
        last_merged = False
        for value in reversed(row):
            if value != 0:
                if merged_row and not last_merged and merged_row[-1] == value:
                    merged_row[-1] *= 2
                    last_merged = True
                else:
                    merged_row.append(value)
                    last_merged = False
        merged_row = [0] * (len(row) - len(merged_row)) + merged_row[::-1]
        new_board.append(merged_row)
    return new_board

def move_up(board):
    size = len(board)
    new_board = [[0] * size for _ in range(size)]

    for col in range(size):
        merged_column = []
        last_merged = False
        for row in range(size):
            value = board[row][col]
            if value != 0:
                if merged_column and not last_merged and merged_column[-1] == value:
                    merged_column[-1] *= 2
                    last_merged = True
                else:
                    merged_column.append(value)
                    last_merged = False

        for row, merged_value in enumerate(merged_column):
            new_board[row][col] = merged_value

    return new_board

def move_down(board):
    size = len(board)
    new_board = [[0] * size for _ in range(size)]

    for col in range(size):
        merged_column = []
        last_merged = False
        for row in reversed(range(size)):
            value = board[row][col]
            if value != 0:
                if merged_column and not last_merged and merged_column[-1] == value:
                    merged_column[-1] *= 2
                    last_merged = True
                else:
                    merged_column.append(value)
                    last_merged = False

        for row, merged_value in enumerate(reversed(merged_column)):
            new_board[size - row - 1][col] = merged_value

    return new_board



def calculate_metric(board):
    empty_count = 0
    for row in board:
        for value in row:
            if value == 0:
                empty_count += 1

    return empty_count

def determine_direction(data):
    board = json.loads(data)
    metrics = []

    # Влево
    moved_left = move_left(board)
    left_metric = calculate_metric(moved_left)
    metrics.append(left_metric)

    # Вправо
    moved_right = move_right(board)
    right_metric = calculate_metric(moved_right)
    metrics.append(right_metric)

    # Вверх
    moved_up = move_up(board)
    up_metric = calculate_metric(moved_up)
    metrics.append(up_metric)

    # Вниз
    moved_down = move_down(board)
    down_metric = calculate_metric(moved_down)
    metrics.append(down_metric)
    print(metrics)

    # Находим индекс направления с наибольшей метрикой
    max_index = metrics.index(max(metrics))

    # Возвращаем соответствующее направление
    if max_index == 0:
        return "left"
    elif max_index == 1:
        return "right"
    elif max_index == 2:
        return "up"
    elif max_index == 3:
        return "down"


    

async def read_digits():
    # data = [
    #     [0,0,0,0],
    #     [0,0,0,0],
    #     [0,0,0,0],
    #     [0,0,0,0]
    # ]

    # pyautogui.screenshot('screenshot.png')
    # # find image on another image
    # r2 = pyautogui.locateAllOnScreen('2.png')
    # r21 = pyautogui.locateAllOnScreen('21.png')
    # r4 = pyautogui.locateAllOnScreen('4.png')
    # r8 = pyautogui.locateAllOnScreen('8.png')
    # r16 = pyautogui.locateAllOnScreen('16.png')
    # r161 = pyautogui.locateAllOnScreen('161.png')
    # r32 = pyautogui.locateAllOnScreen('32.png')
    # r321 = pyautogui.locateAllOnScreen('321.png')
    # r64 = pyautogui.locateAllOnScreen('64.png')
    # r641 = pyautogui.locateAllOnScreen('641.png')
    # r128 = pyautogui.locateAllOnScreen('128.png')
    # r1281 = pyautogui.locateAllOnScreen('1281.png')
    # # python console log
    # for i in r2:
    #     data = append_item(i, 2, data)
    
    # for i in r21:
    #     data = append_item(i, 2, data)

    # for i in r4:
    #     data = append_item(i, 4, data)    
    
    # for i in r8:
    #     data = append_item(i, 8, data)

    # for i in r16:
    #     data = append_item(i, 16, data)

    # for i in r161:
    #     data = append_item(i, 16, data)

    # for i in r32:
    #     data = append_item(i, 32, data)
    
    # for i in r321:
    #     data = append_item(i, 32, data)
    
    # for i in r64:
    #     data = append_item(i, 64, data)
    
    # for i in r641:
    #     data = append_item(i, 64, data)
    
    # for i in r128:
    #     print(i)
    #     data = append_item(i, 128, data)
    
    # for i in r1281:
    #     print(i)
    #     data = append_item(i, 128, data)

    # # array to json
    # print(data)

    # json1 = json.dumps(data)
    # dir = determine_direction(json1)
    dirs = ['left', 'right', 'up', 'down']
    dir = random.choice(dirs)
    print(dir)
    # exit()
    # openai.api_key = "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"


    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages = [
    #             {"role": "system", "content": "Ты будешт играть в игру 2048. Ты будешь получать json массив данных игры и тебе нужно будет вернуть json массив с указанием стороны движения."},
    #             {"role": "user", "content": "Я даю json массив данных игры: " + json1},
    #             {"role": "user", "content": "Определи пожалуйста правильную сторону движения и верни в формате JSON и верни только его: {\"direction\": \"right or left or up or down\"}"}
    #     ]
    # )

    # result = ''
    # for choice in response.choices:
    #     result += choice.message.content

    # # detect json in string result
    # result = result[result.find('{'):result.rfind('}')+1]
    # # parse json to object
    # x = json.loads(result, object_hook=lambda d: SimpleNamespace(**d))
    pyautogui.press(dir)
    pyautogui.sleep(4)
    pyautogui.click(1648, 698)
    pyautogui.sleep(4)
    pyautogui.click(1648, 698)
    await read_digits()

def append_item(i, number, data):
    x = 0
    y = 0
    if i.top > 400 and i.top < 493:
        y = 0
    
    if i.top > 532 and i.top < 625:
        y = 1
    
    if i.top > 663 and i.top < 746:
        y = 2
    
    if i.top > 788 and i.top < 867:
        y = 3
    
    if i.left > 713 and i.left < 809:
        x = 0
    
    if i.left > 844 and i.left < 934:
        x = 1

    if i.left > 971 and i.left < 1067:
        x = 2
    
    if i.left > 1097 and i.left < 1195:
        x = 3
    
    data[y][x] = number

    return data


async def click_upscale(clickCounter = 0, needScroll = True, element = None):
    if clickCounter == 0:
        sleep = 10
    else:
        sleep = 2
    

    if element is not None:
        pyautogui.click(element.left + (100*(clickCounter-1)), element.top + 20)
        pyautogui.sleep(sleep)
        needScroll = False
        clickCounter += 1
    else:    
        result = pyautogui.locateAllOnScreen('btn.png')
        count = 0
        resultArr = []
        for i in result:
            count += 1
            resultArr.append(i)
        
        offset = 0
        if count > (9 - clickCounter):
            offset = count - (9 - clickCounter)

        print(offset, clickCounter)

        # exit()
        print(resultArr)

        # if isset resultArr[offset] then click

        if len(resultArr) > 0:  
            pyautogui.click(resultArr[offset])
            pyautogui.click(resultArr[offset].left + 500, resultArr[offset].top)
            pyautogui.sleep(sleep)
            if needScroll:
                pyautogui.scroll(300)
            needScroll = False
            pyautogui.sleep(sleep)
            clickCounter += 1
            if clickCounter >= 2:
                element = resultArr[offset]

        
    if clickCounter <= 3:
        await click_upscale(clickCounter, needScroll, element)



async def main():
    await read_digits()


asyncio.run(main())

# click_upscale()

# type_message('male archangel for 2d platformer game digital art')

# get_channel_history()

# python wait 3 seconds
#pyautogui.sleep(3)

#console.log(pyautogui.position())
#pyautogui.click(100, 100)

