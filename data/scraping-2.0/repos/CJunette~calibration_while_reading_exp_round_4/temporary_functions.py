import json

import cv2
import math
import os
from multiprocessing import Pool
import re

import numpy as np
import openai
import pandas as pd

import analyse_calibration_data
import analyse_reading_data
import configs
import read_files
import talk_with_GPT


def combine_error_data_in_round_4():
    '''
    用于将出错的实验中的有效阅读数据与后续补救实验中的数据合到一起。
    :return:
    '''

    dir_target = "D:\\Work\\2023.7.25_visualization_of_gaze_pilot_experiment_3_ver_0\\calibrationdataanalyzer\\packed_data\\round_4\\tobii\\20230810_111446_20230809_1.5_fxz\\reading\\"
    dir_source = "D:\\Work\\2023.7.25_visualization_of_gaze_pilot_experiment_3_ver_0\\calibrationdataanalyzer\\packed_data\\round_4\\tobii\\20230810_102118_20230810_1_fxz\\reading\\"

    target_file_list = os.listdir(dir_target)
    target_df_list = []
    for file_name in target_file_list:
        df = pd.read_csv(dir_target + file_name, encoding="utf-8")
        target_df_list.append(df)

    source_file_list = os.listdir(dir_source)
    source_df_list = []
    for file_name in source_file_list:
        df = pd.read_csv(dir_source + file_name, encoding="utf-8")
        source_df_list.append(df)

    source_para_id_list_1 = []
    source_df_of_para_id_list_1 = []
    for source_index in range(len(source_df_list)):
        source_para_id_list_2 = source_df_list[source_index]["matrix_x"].unique()
        source_para_id_list_1.append(source_para_id_list_2)
        source_df_of_para_id_list_2 = []
        for para_id in source_para_id_list_2:
            df = source_df_list[source_index]
            df_para_id = df[df["matrix_x"] == para_id]
            source_df_of_para_id_list_2.append(df_para_id)
        source_df_of_para_id_list_1.append(source_df_of_para_id_list_2)

    target_para_id_list_1 = []
    target_df_of_para_id_list_1 = []
    for target_index in range(len(target_df_list)):
        target_para_id_list_2 = target_df_list[target_index]["matrix_x"].unique()
        target_para_id_list_1.append(target_para_id_list_2)
        target_df_of_para_id_list_2 = []
        for para_id in target_para_id_list_2:
            df = target_df_list[target_index]
            df_para_id = df[df["matrix_x"] == para_id]
            target_df_of_para_id_list_2.append(df_para_id)
        target_df_of_para_id_list_1.append(target_df_of_para_id_list_2)

    for source_index in range(len(source_df_of_para_id_list_1)):
        for sub_source_index in range(len(source_para_id_list_1[source_index])):
            actual_source_index = source_para_id_list_1[source_index][sub_source_index]
            source_df = source_df_of_para_id_list_1[source_index][sub_source_index]
            for target_index in range(len(target_df_of_para_id_list_1)):
                if actual_source_index in target_para_id_list_1[target_index]:
                    target_df_index = np.where(target_para_id_list_1[target_index] == actual_source_index)[0][0]
                    target_df = target_df_of_para_id_list_1[target_index][target_df_index]

                    phase = source_df["phase"].tolist()
                    source_length = len(phase)
                    username = [target_df["username"].tolist()[0] for _ in range(source_length)]
                    iteration = [target_df["iteration"].tolist()[0] for _ in range(source_length)]
                    time = source_df["time"].tolist()
                    gaze_x = source_df["gaze_x"].tolist()
                    gaze_y = source_df["gaze_y"].tolist()
                    speed = source_df["speed"].tolist()
                    matrix_x = [target_df["matrix_x"].tolist()[0] for _ in range(source_length)]
                    matrix_y = [target_df["matrix_y"].tolist()[0] for _ in range(source_length)]

                    new_df = pd.DataFrame({
                        "username": username,
                        "iteration": iteration,
                        "phase": phase,
                        "time": time,
                        "gaze_x": gaze_x,
                        "gaze_y": gaze_y,
                        "speed": speed,
                        "matrix_x": matrix_x,
                        "matrix_y": matrix_y
                    })

                    target_df_of_para_id_list_1[target_index][target_df_index] = new_df
                    break

    for target_index in range(len(target_df_of_para_id_list_1)):
        con_df = pd.concat(target_df_of_para_id_list_1[target_index])
        con_df.reset_index()

        con_df_group_by_matrix_x = con_df.groupby("matrix_x")
        start_time_list = []
        matrix_x_list = []
        for matrix_x, df in con_df_group_by_matrix_x:
            start_time_list.append(df["time"].tolist())
            matrix_x_list.append(matrix_x)
        start_time_difference = abs(start_time_list[0][-1] - start_time_list[1][0])
        if start_time_difference > 100: # 这里的100是个经验值，指的是同一个iteration下，前一个df的最后一个时间和后一个df的第一个时间之间的差。
            second_df_start_time = start_time_list[1][0]
            new_start_time = start_time_list[0][-1] + 20 # 这里的20也是经验值，是同一个iteration下，给前一个df的最后一个时间加上20后作为后一个df的第一个时间。
            for gaze_index in range(con_df.shape[0]):
                if con_df.iloc[gaze_index]["matrix_x"] == matrix_x_list[1]:
                    second_df_original_time = con_df.iloc[gaze_index]["time"]
                    con_df.iloc[gaze_index, con_df.columns.get_loc("time")] = second_df_original_time - second_df_start_time + new_start_time

        save_prefix = dir_target
        if not os.path.exists(os.path.dirname(save_prefix)):
            os.makedirs(os.path.dirname(save_prefix))

        save_name = target_file_list[target_index][:-4]
        save_path = save_prefix + save_name + ".csv"

        con_df.to_csv(save_path, encoding="utf-8_sig")


def split_data_in_round_1():
    exp_text_file = "data/back_up_gaze_data/round_1/exp_text.txt"
    with open(exp_text_file, "r", encoding="utf-8_sig") as f:
        exp_text = f.read()
    exp_text = exp_text.replace("\n\n\n", "\n-----\n")

    round_1_original_data_dir = "data/back_up_gaze_data/round_1/original_data/"
    file_list = os.listdir(round_1_original_data_dir)
    file_list = [file for file in file_list if file.endswith(".csv")]
    df_list = []
    for file_index in range(len(file_list)):
        file_name = round_1_original_data_dir + file_list[file_index]
        df = pd.read_csv(file_name, encoding="utf-8_sig")
        df_list.append(df)

    file_prefix = "data/back_up_gaze_data/round_1/modified_data/"
    for file_index in range(len(df_list)):
        file_path = file_prefix + file_list[file_index][:-4] + "/"
        reading_file_path = file_path + "reading/"
        if not os.path.exists(reading_file_path):
            os.makedirs(reading_file_path)

        # save reading of each iteration as different files.
        df = df_list[file_index]
        df_reading = df[df["phase"] == "word"]
        df_reading_group_by_iteration = df_reading.groupby("iteration")
        for iteration, df_iteration in df_reading_group_by_iteration:
            reading_save_path = f"{reading_file_path}{int(float(iteration))}.csv"
            df_iteration.to_csv(reading_save_path, encoding="utf-8_sig")

        # save calibration
        df_calibration = df[df["phase"] == "cali"]
        cali_save_path = f"{file_path}calibration.csv"
        df_calibration.to_csv(cali_save_path, encoding="utf-8_sig")

        # save exp_text
        exp_text_save_path = f"{file_path}exp_text.txt"
        with open(exp_text_save_path, "w", encoding="utf-8_sig") as f:
            f.write(exp_text)


def split_seeso_data():
    file_path = "data/modified_gaze_data/round_4/seeso/"
    file_list = os.listdir(file_path)
    file_list = [file for file in file_list if not file.startswith("seeso")]

    for file_index in range(len(file_list)):
        file_name = f"{file_path}{file_list[file_index]}"
        df = pd.read_csv(file_name, encoding="utf-8_sig", index_col=False)
        # TODO 这里交换列的地方可能可以去掉，之后的数据应该会被处理好。
        df_wrong_time = df["time"].tolist()
        # 将df的第6列到第10列数据左移一列。第6列数据覆盖第5列数据。
        df.iloc[:, 5:10] = df.iloc[:, 5:10].shift(-1, axis=1)
        # 将df的第10列数据替换为df_wrong_time
        df["TrackingState"] = df_wrong_time

        df_reading = df[df["phase"] == "word"]
        df_cali = df[df["phase"] == "cali"]

        save_path = f"{file_path}/{file_list[file_index][:-4]}/"
        df_reading_grouped_by_iteration = df_reading.groupby("iteration")
        for iteration, df_iteration in df_reading_grouped_by_iteration:
            df_reading_grouped_by_matrix_x = df_iteration.groupby("matrix_x")
            speed_list = []
            for matrix_x, df_matrix_x in df_reading_grouped_by_matrix_x:
                speed = [0]
                for i in range(1, df_matrix_x.shape[0]):
                    gaze_x = df_matrix_x.iloc[i]["gaze_x"]
                    gaze_y = df_matrix_x.iloc[i]["gaze_y"]
                    last_x = df_matrix_x.iloc[i - 1]["gaze_x"]
                    last_y = df_matrix_x.iloc[i - 1]["gaze_y"]
                    distance = math.sqrt((gaze_x - last_x) ** 2 + (gaze_y - last_y) ** 2)
                    speed.append(distance)
                speed_list.extend(speed)
            df_iteration["speed"] = speed_list
            df_iteration.dropna(inplace=True)
            save_name = f"{save_path}/seeso_reading/{int(float(iteration))}.csv"
            if not os.path.exists(os.path.dirname(save_name)):
                os.makedirs(os.path.dirname(save_name))
            df_iteration.to_csv(save_name, encoding="utf-8_sig")

        df_cali.to_csv(f"{save_path}/seeso_calibration.csv", encoding="utf-8_sig")


def modify_round_1_reading_data_using_calibration():
    file_path_prefix = f"data/back_up_gaze_data/round_1/reformat_data/"
    file_list = os.listdir(file_path_prefix)

    for file_index in range(len(file_list)):
        calibration_file_name = f"{file_path_prefix}/{file_list[file_index]}/calibration.csv"
        df_calibration = pd.read_csv(calibration_file_name, encoding="utf-8_sig")
        homography_matrix = analyse_calibration_data.get_homography_matrix_for_calibration(df_calibration)

        reading_file_path = f"{file_path_prefix}/{file_list[file_index]}/reading"
        reading_file_list = os.listdir(reading_file_path)
        reading_file_list = [int(i[:-4]) for i in reading_file_list]
        reading_file_list.sort()

        for reading_file_index in range(len(reading_file_list)):
            df = pd.read_csv(f"{reading_file_path}/{reading_file_list[reading_file_index]}.csv", encoding="utf-8_sig", index_col=False)
            df.drop(df.columns[0], axis=1, inplace=True)
            transformed_df = analyse_reading_data.apply_homography_to_reading(df, homography_matrix)

            save_name = f"data/modified_gaze_data/round_1/tobii/{file_list[file_index]}/reading/{reading_file_index}.csv"
            transformed_df.to_csv(save_name, encoding="utf-8_sig", index=False)


def generate_reading_data_90_to_94():
    '''
        单独生成90-94的reading数据。
    '''
    file_path_prefix = f"data/back_up_gaze_data/round_1/reformat_data/"
    file_list = os.listdir(file_path_prefix)

    reading_df_list_1 = []
    calibration_df_list_1 = []
    for file_index in range(len(file_list)):
        calibration_file_name = f"{file_path_prefix}/{file_list[file_index]}/calibration.csv"
        df_calibration = pd.read_csv(calibration_file_name, encoding="utf-8_sig")
        calibration_df_list_1.append(df_calibration)

        reading_file_path = f"{file_path_prefix}/{file_list[file_index]}/reading"
        reading_file_list = os.listdir(reading_file_path)
        reading_file_list = [int(i[:-4]) for i in reading_file_list]
        reading_file_list.sort()
        df_90_94_list = []
        index_list = [90, 91, 92, 93, 94]
        for reading_file_index in range(len(reading_file_list)):
            df = pd.read_csv(f"{reading_file_path}/{reading_file_list[reading_file_index]}.csv", encoding="utf-8_sig")
            para_id_list = df["matrix_x"].unique()
            para_id_list = [int(para_id_list[i]) for i in range(len(para_id_list))]
            for index in index_list:
                if index in para_id_list:
                    df_90_94_list.append(df[df["matrix_x"] == index])
        df_new = pd.concat(df_90_94_list)
        save_path = f"data/back_up_gaze_data/round_1/temp/90-94/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df_new.to_csv(save_path + f"{file_list[file_index]}_90-94.csv", encoding="utf-8_sig", index=False)


def get_tokens_of_certain_para(token_type, para_list):
    for para_index in para_list:
        file_name = f"data/text/{configs.round}/tokens/{token_type}_tokens/{para_index}.csv"
        df = pd.read_csv(file_name, encoding="utf-8_sig", index_col=False)
        token_list = []
        for token_index in range(df.shape[0]):
            token_list.append((token_index, df.iloc[token_index]["tokens"]))
        print(token_list)


def ask_gpt_about_density():
    talk_with_GPT.set_openai()
    talk_with_GPT.start_using_IDE()

    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": f"你是一个分析文本与眼动行为的专家。"},
            {"role": "user", "content": "我会给你一系列文本文字和该文字对应的眼动停留时间（用相对密度来表示），将这些文字组合起来就是完整的文章。"
                                        "我会给你传递一个字典构成的列表，字典中的'token'代表文字，'token_density'代表相对的眼动停留时间。"
                                        "请告诉我这些眼动停留时间较长的文字的特点，为什么人在阅读它时会发生停留；以及我应该设计什么样的prompt才能让你能够预测出这样的眼动停留行为。\n"
                                        "文字及停留时间如下：\n"
                                        "[{'token_index': 0, 'token': 很, 'relative_density': 0.000204}, {'token_index': 1, 'token': 努, 'relative_density': 0.000126}, {'token_index': 2, 'token': 力, 'relative_density': 0.001339}, {'token_index': 3, 'token': 的, 'relative_density': 0.001728}, {'token_index': 4, 'token': 班, 'relative_density': 0.000794}, {'token_index': 5, 'token': 主, 'relative_density': 0.000749}, {'token_index': 6, 'token': 任, 'relative_density': 0.000094}, {'token_index': 7, 'token': :, 'relative_density': 0.000095}, {'token_index': 8, 'token':   , 'relative_density': 0.000000}, {'token_index': 9, 'token':   , 'relative_density': 0.000135}, {'token_index': 10, 'token': 有, 'relative_density': 0.000314}, {'token_index': 11, 'token': 没, 'relative_density': 0.001710}, {'token_index': 12, 'token': 有, 'relative_density': 0.001799}, {'token_index': 13, 'token': 发, 'relative_density': 0.000764}, {'token_index': 14, 'token': 现, 'relative_density': 0.000625}, {'token_index': 15, 'token': 现, 'relative_density': 0.000772}, {'token_index': 16, 'token': 在, 'relative_density': 0.000853}, {'token_index': 17, 'token': 小, 'relative_density': 0.000912}, {'token_index': 18, 'token': 孩, 'relative_density': 0.000866}, {'token_index': 19, 'token': 子, 'relative_density': 0.000777}, {'token_index': 20, 'token': 的, 'relative_density': 0.000750}, {'token_index': 21, 'token': 表, 'relative_density': 0.000780}, {'token_index': 22, 'token': 达, 'relative_density': 0.000645}, {'token_index': 23, 'token': 能, 'relative_density': 0.000791}, {'token_index': 24, 'token': 力, 'relative_density': 0.000805}, {'token_index': 25, 'token': 下, 'relative_density': 0.000497}, {'token_index': 26, 'token': 降, 'relative_density': 0.000338}, {'token_index': 27, 'token': 了, 'relative_density': 0.000198}, {'token_index': 28, 'token': ？, 'relative_density': 0.000115}, {'token_index': 29, 'token': 比, 'relative_density': 0.000145}, {'token_index': 30, 'token': 如, 'relative_density': 0.000285}, {'token_index': 31, 'token': 遇, 'relative_density': 0.000456}, {'token_index': 32, 'token': 到, 'relative_density': 0.000214}, {'token_index': 33, 'token': 任, 'relative_density': 0.000401}, {'token_index': 34, 'token': 何, 'relative_density': 0.000394}, {'token_index': 35, 'token': 事, 'relative_density': 0.000160}, {'token_index': 36, 'token': 情, 'relative_density': 0.000205}, {'token_index': 37, 'token': 都, 'relative_density': 0.000076}, {'token_index': 38, 'token': 可, 'relative_density': 0.000023}, {'token_index': 39, 'token': 以, 'relative_density': 0.000038}, {'token_index': 40, 'token': 用, 'relative_density': 0.000422}, {'token_index': 41, 'token': “, 'relative_density': 0.000690}, {'token_index': 42, 'token': 6, 'relative_density': 0.000541}, {'token_index': 43, 'token': ”, 'relative_density': 0.000561}, {'token_index': 44, 'token': 字, 'relative_density': 0.000614}, {'token_index': 45, 'token': 回, 'relative_density': 0.000873}, {'token_index': 46, 'token': 答, 'relative_density': 0.000456}, {'token_index': 47, 'token': 。, 'relative_density': 0.000306}, {'token_index': 48, 'token': 网, 'relative_density': 0.000142}, {'token_index': 49, 'token': 络, 'relative_density': 0.000634}, {'token_index': 50, 'token': 用, 'relative_density': 0.000779}, {'token_index': 51, 'token': 语, 'relative_density': 0.000430}, {'token_index': 52, 'token': 替, 'relative_density': 0.000763}, {'token_index': 53, 'token': 代, 'relative_density': 0.000337}, {'token_index': 54, 'token': 了, 'relative_density': 0.000275}, {'token_index': 55, 'token': 常, 'relative_density': 0.000334}, {'token_index': 56, 'token': 规, 'relative_density': 0.000401}, {'token_index': 57, 'token': 表, 'relative_density': 0.000244}, {'token_index': 58, 'token': 达, 'relative_density': 0.000132}, {'token_index': 59, 'token': ，, 'relative_density': 0.000165}, {'token_index': 60, 'token': 很, 'relative_density': 0.000272}, {'token_index': 61, 'token': 多, 'relative_density': 0.000704}, {'token_index': 62, 'token': 孩, 'relative_density': 0.000818}, {'token_index': 63, 'token': 子, 'relative_density': 0.000345}, {'token_index': 64, 'token': 作, 'relative_density': 0.000039}, {'token_index': 65, 'token': 业, 'relative_density': 0.000019}, {'token_index': 66, 'token': 、, 'relative_density': 0.000000}, {'token_index': 67, 'token': 日, 'relative_density': 0.000000}, {'token_index': 68, 'token': 记, 'relative_density': 0.000000}, {'token_index': 69, 'token': 、, 'relative_density': 0.000057}, {'token_index': 70, 'token': 作, 'relative_density': 0.000102}, {'token_index': 71, 'token': 文, 'relative_density': 0.000125}, {'token_index': 72, 'token': 都, 'relative_density': 0.000392}, {'token_index': 73, 'token': 有, 'relative_density': 0.000256}, {'token_index': 74, 'token': 网, 'relative_density': 0.000321}, {'token_index': 75, 'token': 络, 'relative_density': 0.000463}, {'token_index': 76, 'token': 用, 'relative_density': 0.000388}, {'token_index': 77, 'token': 语, 'relative_density': 0.000267}, {'token_index': 78, 'token': 的, 'relative_density': 0.000239}, {'token_index': 79, 'token': 影, 'relative_density': 0.000084}, {'token_index': 80, 'token': 子, 'relative_density': 0.000038}, {'token_index': 81, 'token': 。, 'relative_density': 0.000145}, {'token_index': 82, 'token':   , 'relative_density': 0.000000}, {'token_index': 83, 'token':   , 'relative_density': 0.000000}, {'token_index': 84, 'token': 记, 'relative_density': 0.000039}, {'token_index': 85, 'token': 得, 'relative_density': 0.000211}, {'token_index': 86, 'token': 有, 'relative_density': 0.000394}, {'token_index': 87, 'token': 一, 'relative_density': 0.000774}, {'token_index': 88, 'token': 阵, 'relative_density': 0.000523}, {'token_index': 89, 'token': 子, 'relative_density': 0.000810}, {'token_index': 90, 'token': 我, 'relative_density': 0.000488}, {'token_index': 91, 'token': 自, 'relative_density': 0.000638}, {'token_index': 92, 'token': 己, 'relative_density': 0.000894}, {'token_index': 93, 'token': 也, 'relative_density': 0.000658}, {'token_index': 94, 'token': 玩, 'relative_density': 0.000186}, {'token_index': 95, 'token': 梗, 'relative_density': 0.000362}, {'token_index': 96, 'token': ，, 'relative_density': 0.000294}, {'token_index': 97, 'token': “, 'relative_density': 0.000519}, {'token_index': 98, 'token': 信, 'relative_density': 0.000822}, {'token_index': 99, 'token': XX, 'relative_density': 0.001051}, {'token_index': 100, 'token': ，, 'relative_density': 0.001002}, {'token_index': 101, 'token': 得, 'relative_density': 0.000969}, {'token_index': 102, 'token': 永, 'relative_density': 0.000746}, {'token_index': 103, 'token': 生, 'relative_density': 0.000158}, {'token_index': 104, 'token': ”, 'relative_density': 0.000151}, {'token_index': 105, 'token': ，, 'relative_density': 0.000336}, {'token_index': 106, 'token': 时, 'relative_density': 0.000499}, {'token_index': 107, 'token': 间, 'relative_density': 0.001198}, {'token_index': 108, 'token': 长, 'relative_density': 0.000542}, {'token_index': 109, 'token': 了, 'relative_density': 0.000026}, {'token_index': 110, 'token': 导, 'relative_density': 0.000000}, {'token_index': 111, 'token': 致, 'relative_density': 0.000000}, {'token_index': 112, 'token': 一, 'relative_density': 0.000000}, {'token_index': 113, 'token': 些, 'relative_density': 0.000000}, {'token_index': 114, 'token': 本, 'relative_density': 0.000019}, {'token_index': 115, 'token': 会, 'relative_density': 0.000355}, {'token_index': 116, 'token': 组, 'relative_density': 0.000337}, {'token_index': 117, 'token': 织, 'relative_density': 0.000237}, {'token_index': 118, 'token': 的, 'relative_density': 0.000406}, {'token_index': 119, 'token': 语, 'relative_density': 0.000243}, {'token_index': 120, 'token': 言, 'relative_density': 0.000157}, {'token_index': 121, 'token': 都, 'relative_density': 0.000218}, {'token_index': 122, 'token': 不, 'relative_density': 0.000537}, {'token_index': 123, 'token': 知, 'relative_density': 0.000636}, {'token_index': 124, 'token': 怎, 'relative_density': 0.000093}, {'token_index': 125, 'token': 么, 'relative_density': 0.000084}, {'token_index': 126, 'token': 去, 'relative_density': 0.000081}, {'token_index': 127, 'token': 表, 'relative_density': 0.000221}, {'token_index': 128, 'token': 达, 'relative_density': 0.000065}, {'token_index': 129, 'token': 。, 'relative_density': 0.000180}, {'token_index': 130, 'token': 我, 'relative_density': 0.000405}, {'token_index': 131, 'token': 是, 'relative_density': 0.000412}, {'token_index': 132, 'token': 成, 'relative_density': 0.000124}, {'token_index': 133, 'token': 年, 'relative_density': 0.000069}, {'token_index': 134, 'token': 人, 'relative_density': 0.000054}, {'token_index': 135, 'token': ，, 'relative_density': 0.000073}, {'token_index': 136, 'token': 我, 'relative_density': 0.000152}, {'token_index': 137, 'token': 能, 'relative_density': 0.000451}, {'token_index': 138, 'token': 意, 'relative_density': 0.000149}, {'token_index': 139, 'token': 识, 'relative_density': 0.000077}, {'token_index': 140, 'token': 到, 'relative_density': 0.000000}, {'token_index': 141, 'token': 问, 'relative_density': 0.000000}, {'token_index': 142, 'token': 题, 'relative_density': 0.000000}, {'token_index': 143, 'token': 的, 'relative_density': 0.000091}, {'token_index': 144, 'token': 严, 'relative_density': 0.000181}, {'token_index': 145, 'token': 重, 'relative_density': 0.000120}, {'token_index': 146, 'token': 性, 'relative_density': 0.000105}, {'token_index': 147, 'token': ，, 'relative_density': 0.000111}, {'token_index': 148, 'token': 所, 'relative_density': 0.000147}, {'token_index': 149, 'token': 以, 'relative_density': 0.000243}, {'token_index': 150, 'token': 立, 'relative_density': 0.000144}, {'token_index': 151, 'token': 刻, 'relative_density': 0.000095}, {'token_index': 152, 'token': 克, 'relative_density': 0.000375}, {'token_index': 153, 'token': 制, 'relative_density': 0.000567}, {'token_index': 154, 'token': 网, 'relative_density': 0.000430}, {'token_index': 155, 'token': 络, 'relative_density': 0.000462}, {'token_index': 156, 'token': 表, 'relative_density': 0.000166}, {'token_index': 157, 'token': 达, 'relative_density': 0.000213}, {'token_index': 158, 'token': ，, 'relative_density': 0.000164}, {'token_index': 159, 'token': 多, 'relative_density': 0.000332}, {'token_index': 160, 'token': 看, 'relative_density': 0.000237}, {'token_index': 161, 'token': 名, 'relative_density': 0.000608}, {'token_index': 162, 'token': 著, 'relative_density': 0.000596}, {'token_index': 163, 'token': 和, 'relative_density': 0.000222}, {'token_index': 164, 'token': 经, 'relative_density': 0.000316}, {'token_index': 165, 'token': 典, 'relative_density': 0.000358}, {'token_index': 166, 'token': ，, 'relative_density': 0.000405}, {'token_index': 167, 'token': 才, 'relative_density': 0.000354}, {'token_index': 168, 'token': 把, 'relative_density': 0.000586}, {'token_index': 169, 'token': 思, 'relative_density': 0.000241}, {'token_index': 170, 'token': 维, 'relative_density': 0.000000}, {'token_index': 171, 'token': 带, 'relative_density': 0.000000}, {'token_index': 172, 'token': 回, 'relative_density': 0.000000}, {'token_index': 173, 'token': 来, 'relative_density': 0.000072}, {'token_index': 174, 'token': 。, 'relative_density': 0.000123}, {'token_index': 175, 'token': 成, 'relative_density': 0.000151}, {'token_index': 176, 'token': 年, 'relative_density': 0.001365}, {'token_index': 177, 'token': 人, 'relative_density': 0.000345}, {'token_index': 178, 'token': 都, 'relative_density': 0.000340}, {'token_index': 179, 'token': 是, 'relative_density': 0.000331}, {'token_index': 180, 'token': 如, 'relative_density': 0.000101}, {'token_index': 181, 'token': 此, 'relative_density': 0.000200}, {'token_index': 182, 'token': ，, 'relative_density': 0.000118}, {'token_index': 183, 'token': 小, 'relative_density': 0.000163}, {'token_index': 184, 'token': 孩, 'relative_density': 0.000235}, {'token_index': 185, 'token': 子, 'relative_density': 0.000707}, {'token_index': 186, 'token': 三, 'relative_density': 0.000888}, {'token_index': 187, 'token': 观, 'relative_density': 0.000626}, {'token_index': 188, 'token': 没, 'relative_density': 0.000407}, {'token_index': 189, 'token': 有, 'relative_density': 0.000630}, {'token_index': 190, 'token': 形, 'relative_density': 0.000206}, {'token_index': 191, 'token': 成, 'relative_density': 0.000159}, {'token_index': 192, 'token': ，, 'relative_density': 0.000104}, {'token_index': 193, 'token': 更, 'relative_density': 0.000222}, {'token_index': 194, 'token': 容, 'relative_density': 0.000425}, {'token_index': 195, 'token': 易, 'relative_density': 0.000372}, {'token_index': 196, 'token': 被, 'relative_density': 0.000250}, {'token_index': 197, 'token': 影, 'relative_density': 0.000086}, {'token_index': 198, 'token': 响, 'relative_density': 0.000096}, {'token_index': 199, 'token': 。, 'relative_density': 0.000000}, {'token_index': 200, 'token': 在, 'relative_density': 0.000000}, {'token_index': 201, 'token': 班, 'relative_density': 0.000000}, {'token_index': 202, 'token': 级, 'relative_density': 0.000019}, {'token_index': 203, 'token': ，, 'relative_density': 0.000019}, {'token_index': 204, 'token': 对, 'relative_density': 0.000000}, {'token_index': 205, 'token': 于, 'relative_density': 0.000140}, {'token_index': 206, 'token': 学, 'relative_density': 0.000151}, {'token_index': 207, 'token': 生, 'relative_density': 0.000229}, {'token_index': 208, 'token': 玩, 'relative_density': 0.000328}, {'token_index': 209, 'token': 梗, 'relative_density': 0.000093}, {'token_index': 210, 'token': 一, 'relative_density': 0.000073}, {'token_index': 211, 'token': 直, 'relative_density': 0.000146}, {'token_index': 212, 'token': 以, 'relative_density': 0.000163}, {'token_index': 213, 'token': 来, 'relative_density': 0.000316}, {'token_index': 214, 'token': 我, 'relative_density': 0.000306}, {'token_index': 215, 'token': 都, 'relative_density': 0.000195}, {'token_index': 216, 'token': 没, 'relative_density': 0.000166}, {'token_index': 217, 'token': 有, 'relative_density': 0.000107}, {'token_index': 218, 'token': 一, 'relative_density': 0.000165}, {'token_index': 219, 'token': 棒, 'relative_density': 0.000132}, {'token_index': 220, 'token': 子, 'relative_density': 0.000296}, {'token_index': 221, 'token': 打, 'relative_density': 0.000019}, {'token_index': 222, 'token': 死, 'relative_density': 0.000019}, {'token_index': 223, 'token': ，, 'relative_density': 0.000077}, {'token_index': 224, 'token': 但, 'relative_density': 0.000090}, {'token_index': 225, 'token': 是, 'relative_density': 0.000052}, {'token_index': 226, 'token': 希, 'relative_density': 0.000058}, {'token_index': 227, 'token': 望, 'relative_density': 0.000163}, {'token_index': 228, 'token': 未, 'relative_density': 0.000039}, {'token_index': 229, 'token': 来, 'relative_density': 0.000000}, {'token_index': 230, 'token': 孩, 'relative_density': 0.000000}, {'token_index': 231, 'token': 子, 'relative_density': 0.000000}, {'token_index': 232, 'token': 们, 'relative_density': 0.000019}, {'token_index': 233, 'token': 可, 'relative_density': 0.000000}, {'token_index': 234, 'token': 以, 'relative_density': 0.000000}, {'token_index': 235, 'token': 早, 'relative_density': 0.000140}, {'token_index': 236, 'token': 日, 'relative_density': 0.000019}, {'token_index': 237, 'token': 有, 'relative_density': 0.000000}, {'token_index': 238, 'token': 自, 'relative_density': 0.000000}, {'token_index': 239, 'token': 己, 'relative_density': 0.000073}, {'token_index': 240, 'token': 的, 'relative_density': 0.000097}, {'token_index': 241, 'token': 判, 'relative_density': 0.000148}, {'token_index': 242, 'token': 断, 'relative_density': 0.000229}, {'token_index': 243, 'token': ，, 'relative_density': 0.000297}, {'token_index': 244, 'token': 家, 'relative_density': 0.000286}, {'token_index': 245, 'token': 长, 'relative_density': 0.000174}, {'token_index': 246, 'token': 、, 'relative_density': 0.000210}, {'token_index': 247, 'token': 学, 'relative_density': 0.000277}, {'token_index': 248, 'token': 校, 'relative_density': 0.000329}, {'token_index': 249, 'token': 、, 'relative_density': 0.000222}, {'token_index': 250, 'token': 社, 'relative_density': 0.000185}, {'token_index': 251, 'token': 会, 'relative_density': 0.000362}, {'token_index': 252, 'token': 共, 'relative_density': 0.000196}, {'token_index': 253, 'token': 同, 'relative_density': 0.000438}, {'token_index': 254, 'token': 引, 'relative_density': 0.000258}, {'token_index': 255, 'token': 导, 'relative_density': 0.000237}, {'token_index': 256, 'token': ，, 'relative_density': 0.000200}, {'token_index': 257, 'token': 千, 'relative_density': 0.000318}, {'token_index': 258, 'token': 万, 'relative_density': 0.000441}, {'token_index': 259, 'token': 不, 'relative_density': 0.000116}, {'token_index': 260, 'token': 可, 'relative_density': 0.000023}, {'token_index': 261, 'token': 让, 'relative_density': 0.000000}, {'token_index': 262, 'token': 孩, 'relative_density': 0.000058}, {'token_index': 263, 'token': 子, 'relative_density': 0.000000}, {'token_index': 264, 'token': 们, 'relative_density': 0.000019}, {'token_index': 265, 'token': 沉, 'relative_density': 0.000085}, {'token_index': 266, 'token': 迷, 'relative_density': 0.000252}, {'token_index': 267, 'token': 网, 'relative_density': 0.000826}, {'token_index': 268, 'token': 络, 'relative_density': 0.000591}, {'token_index': 269, 'token': 梗, 'relative_density': 0.000246}, {'token_index': 270, 'token': ！, 'relative_density': 0.000267}]"}]
    )

    response_str = response["choices"][0]["message"]["content"].strip()
    print(response_str)


def ask_gpt_about_article():
    talk_with_GPT.set_openai()
    talk_with_GPT.start_using_IDE()

    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": f"你是一个分析文本阅读行为的专家。你需要找出用户在快速阅读时，最可能关注的那些文字。注意，由于用户是在做快速阅读，所以对于首尾内容关注会更多，对于中间的内容则会更有选择性。"},
            {"role": "user", "content": "我会给你一段文本，你需要告诉我这段文本的类型是什么（新闻、推荐、微博、评论还是其他类型），可能是在哪个平台发布的；这篇文章的5个关键词；"
                                        "以及对于这个平台的用户而言，这些类型的文本应该着重关注哪些内容（what_to_focus，用自然语言描述，尽量全面）；基于关键词与着重关注内容，用户在阅读这篇文章中主要关注的文字（分词）（focus_words）；"
                                        "这类文本非关注的内容是哪些（what_to_ignore，用自然语言描述，尽量全面）；基于与关键词差别及着非重关注内容，用户在阅读这篇文章时没有关注的文字（分词）是哪些（ignored_words）。"
                                        "请注意，值得关注的内容与不值得关注的内容间不应该存在冲突（即对于2个相似的概念，不应该一个出现在值得关注，另一个出现在不值得关注）。你可以尽可能多地找出需要关注和不需要关注的内容。\n"
                                        "对于关注和不关注的划分，一个相对统一的标准是文本中“吸引眼球”的、极具话题性的词语一般都会被关注。"
                                        "但此外，不同类型的文本会有不同的标准。"
                                        "比如对于新闻，用户可能更关注新闻的主题、背景、趋势，而不关注具体的历史背景、原因探讨、详细的阐述部分；"
                                        "对于观点阐述类的文本，用户会更关注观点的核心内容、造成该观点的原因、作者对观点的举例、作者观点与主流想法的冲突点，而不关注那些大家都心知肚明的内容；"
                                        "对于评论类文本，用户会关注对被评论对象的细节描写，而不关注那些主观的印象。"
                                        "假设具体文本为："
                                        """“饭圈”产业价值千亿却走向癫狂，谁该负责
                                        “饭圈”文化愈演愈烈，粉丝与网络水军混杂，因各种立场、观点、利益冲突，而引发各类网上互撕互黑等风波。6月15日，中央网信办宣布开展为期两个月的“饭圈”乱象整治专项行动，聚焦明星榜单、热门话题、粉丝社群、互动评论等环节。突如其来的强监管，显示近年狂飙突进的“粉丝经济”走到了十字路口。
                                        官方文件直接点名“饭圈”，前所未见。“饭圈”是一个近年走红的网络用语，主要指娱乐明星粉丝（Fans）组成的圈子。不同于过去所谓“追星族”，“饭圈”更多是基于社群网络的半职业化组织，一些娱乐明星的粉丝业已形成职业分工运作模式，包括“粉头”“数据女工”等新型角色，深度参与明星日常活动，为明星造热度，维持形象和商业价值。\n"""
                                        "你需要用字典返回以下六个内容，如"
                                        "{'type': '这段文本类型是新闻，可能在新闻网站或新闻APP上发布。', "
                                        "'keywords': ['饭圈', '整治', '乱象', '娱乐明星', '产业价值千亿']"
                                        "'what_to_focus': '读者在此类文本中应关注的内容包括文中谈到的主要问题和情况，即“饭圈”的现状以及由这种现状产生的问题。同时，他们还需要关注由官方提出的各种应对策略，以及这些策略预计会对所关注的问题产生的影响。', "
                                        "'what_to_ignore': '在这段文本中，需要关注的分词包括“饭圈”、“癫狂”、“中央网信办”、“整治专项行动”、“粉丝经济”、“十字路口”、“粉丝”、“半职业化组织”、“粉头”、“数据女工”。这些词语各自代表了此文的主题，背景，趋势，而它们之间的关系则构成了文章的主要论述。'"
                                        "'focus_words': '在快速阅读时，较具体的历史背景、原因探讨、详细的阐述部分可以忽略，比如“饭圈”的起源、粉丝分工运作模式等内容。这些内容对于理解文章的主旨和大意没有直接帮助，特别是在时间紧张的情况下。'"
                                        "'ignored_words: '可以忽略的分词包括“追星族”、“网络用语”、“走红”、“商业价值”。这些词对于文章的核心观点（饭圈乱象的现状以及应对策略）的理解并没有决定性的影响。'}\n"
                                        "文本如下：\n"
                                        """
    很努力的班主任:
    有没有发现现在小孩子的表达能力下降了？比如遇到任何事情都可以用“6”字回答。网络用语替代了常规表达，很多孩子作业、日记、作文都有网络用语的影子。
    记得有一阵子我自己也玩梗，“信XX，得永生”，时间长了导致一些本会组织的语言都不知怎么去表达。我是成年人，我能意识到问题的严重性，所以立刻克制网络表达，多看名著和经典，才把思维带回来。成年人都是如此，小孩子三观没有形成，更容易被影响。在班级，对于学生玩梗一直以来我都没有一棒子打死，但是希望未来孩子们可以早日有自己的判断，家长、学校、社会共同引导，千万不可让孩子们沉迷网络梗！
    """}]
    )

    # 之后先将文本给GPT让它给出5段分析，然后把这5段分析给GPT，让它对每个分词进行打分。
    response_str = response["choices"][0]["message"]["content"].strip()
    print(response_str)


def ask_gpt_to_subsume_article_single_pool(full_text_dict, finished_list):
    talk_with_GPT.start_using_IDE()
    talk_with_GPT.set_openai()

    text = full_text_dict["text"]
    text_index = int(full_text_dict["text_index"])
    if text_index in finished_list:
        return
    while True:
        response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=[
                {"role": "system", "content": "你是一个文本分类和归纳总结的专家。"},
                {"role": "user", "content": "我会给你一段文本，你需要详细阅读这篇文章，并告诉我这段文本的类型是什么（type，新闻、推荐、微博、评论还是其他类型，列表长度不限，尽可能全面），可能是在哪个平台发布的（platform，列表长度不限），这篇文章的7个关键词（keywords）。"
                                            "你的回答需要以字典的方式返回。接下来我会给你2个案例。\n"
                                            "文本：\n"
                                            """“饭圈”产业价值千亿却走向癫狂，谁该负责
                                            “饭圈”文化愈演愈烈，粉丝与网络水军混杂，因各种立场、观点、利益冲突，而引发各类网上互撕互黑等风波。6月15日，中央网信办宣布开展为期两个月的“饭圈”乱象整治专项行动，聚焦明星榜单、热门话题、粉丝社群、互动评论等环节。突如其来的强监管，显示近年狂飙突进的“粉丝经济”走到了十字路口。
                                            官方文件直接点名“饭圈”，前所未见。“饭圈”是一个近年走红的网络用语，主要指娱乐明星粉丝（Fans）组成的圈子。不同于过去所谓“追星族”，“饭圈”更多是基于社群网络的半职业化组织，一些娱乐明星的粉丝业已形成职业分工运作模式，包括“粉头”“数据女工”等新型角色，深度参与明星日常活动，为明星造热度，维持形象和商业价值。\n"""
                                            "你需要返回的内容如下：\n"
                                            "{'type': ['新闻', '报道'],"
                                            "'platform': ['新闻类app', '新浪微博'],"
                                            "'keywords': ['饭圈', '整治', '乱象', '娱乐明星', '产业价值千亿', '圈子', '数据女工']}\n"
                                            "又如，文本：\n"
                                            """大家好，我是蔡徐坤。近日网络上出现关于我的诸多话题，很抱歉占用了大量社会资源。两年前我处于单身状态，与C女士有过交往，双方之间的私事已经在2021年妥善解决，彼此没有进一步的纠葛。
                                            需要在此向大家和媒体澄清的是，我和C女士的交往均属双方自愿，既不存在“女方为未成年”的情况，也不存在所谓的“强制堕胎”，不涉及违法行为。恳请相关自媒体不传谣、不信谣。
                                            这个教训对我来说是惨痛的，两年的时间里，我也在自责和懊悔中。再次向一直支持信任我的歌迷们道歉，向一直关注我发展与成长的媒体朋友们道歉。今后，我会严格约束自己的言行，接受大众和社会监督。也请大家尊重和保护当事人特别是C女士的个人隐私。\n"""
                                            "你需要返回的内容如下：\n"
                                            "{'type': ['微博', '绯闻声明', '明星八卦'],"
                                            "'platform': ['新浪微博', '官方公众号'],"
                                            "'keywords': ['蔡徐坤', '社会资源', '媒体', '澄清', 'C女士', '强制堕胎', '两年前']}\n"

                 },
                {"role": "assistant", "content": "好的，我理解了你的要求，接下来请给我需要分析的具体文本。"},
                {"role": "user", "content": f"{text}"}
            ])

        response_str = response["choices"][0]["message"]["content"].strip()
        print(response_str)

        try:
            # 找出response_str中被{}包围的内容。
            filtered_response_str = re.findall(r"\{.*?}", response_str)[0]
            response_value = json.loads(filtered_response_str.replace("\'", "\""))
            bool_check = False
            key_list = ["type", "platform", "keywords"]
            for key_index in range(len(key_list)):
                if key_list[key_index] not in response_value.keys():
                    print("key not in response_value", filtered_response_str)
                    bool_check = True
                    raise Exception
            if not bool_check:
                save_path = f"data/text/{configs.round}/description/{text_index}.json"
                with open(save_path, "w", encoding="utf-8_sig") as f:
                    json.dump(response_value, f, ensure_ascii=False, indent=4)
                return
        except Exception as e:
            print("ERROR: ", e, response_str)
            continue


def ask_gpt_to_subsume_article():
    finished_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                     10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                     20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                     30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                     40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                     50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                     60, 61, 63, 64, 65, 66, 67, 68, 69,
                     70, 71, 72, 77, 78, 79,
                     80, 81, 82, 83,
                     98, 99,
                     100, 101, 102, 103, 105, 106, 107]

    raw_full_text_list = read_files.read_sorted_text()
    full_text_list = []
    for file_index in range(len(raw_full_text_list)):
        full_text_list.append([raw_full_text_list[file_index], finished_list])

    with Pool(3) as p:
        p.starmap(ask_gpt_to_subsume_article_single_pool, full_text_list)


def read_article_category():
    talk_with_GPT.start_using_IDE()
    talk_with_GPT.set_openai()

    file_path = f"data/text/{configs.round}/description/"
    file_list = os.listdir(file_path)
    category_list = []
    for file_index in range(len(file_list)):
        json_data = json.load(open(f'{file_path}{file_index}.json', 'r', encoding='utf-8_sig'))
        category_list.append(json_data)
    raw_type_list = [category_list[i]["type"] for i in range(len(category_list))]

    type_list = []
    for i in raw_type_list:
        type_list.extend(i)
    type_set = set(type_list)

    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": "你是一个对文本进行分类的专家"},
            {"role": "user", "content": "我会给你一些词汇，你需要根据这些词汇的语义，将其分为若干个类别。你需要返回这些类别，并告诉我那些词汇属于这些类别。"
                                        "词汇如下：\n"
                                        f"{', '.join(type_set)}"},
        ])
    response_str = response["choices"][0]["message"]["content"].strip()
    print(response_str)


def compute_edge_point_distance():
    def compute_distance(point, std_point_list):
        distance = 0
        for std_point in std_point_list:
            distance += math.sqrt((point[0] - std_point[0]) ** 2 + (point[1] - std_point[1]) ** 2)
        return distance

    calibration_points = np.array(analyse_calibration_data.create_standard_calibration_points()).reshape(-1, 2).tolist()

    calibration_points.sort(key=lambda x: x[0])
    left = calibration_points[0][0]
    right = calibration_points[-1][0]
    calibration_points.sort(key=lambda x: x[1])
    up = calibration_points[0][1]
    down = calibration_points[-1][1]

    horizontal_middle = (left + right) / 2
    vertical_middle = (up + down) / 2
    up_left = [left, up]
    up_middle = [horizontal_middle, up]
    up_right = [right, up]
    middle_left = [left, vertical_middle]
    middle_middle = [horizontal_middle, vertical_middle]
    middle_right = [left, vertical_middle]
    down_left = [left, down]
    down_middle = [horizontal_middle, down]
    down_right = [right, down]

    points = [up_left, up_middle, up_right, middle_left, middle_middle, middle_right, down_left, down_middle, down_right]
    points_names = ['up_left', 'up_middle', 'up_right', 'middle_left', 'middle_middle', 'middle_right', 'down_left', 'down_middle', 'down_right']
    distance_list = []
    for point_index in range(len(points)):
        distance = compute_distance(points[point_index], calibration_points)
        distance_list.append(distance / 360)

    for i in range(len(points)):
        print(points_names[i], distance_list[i])


def change_punctuation_weight(weight_file_name):
    weight_file_prefix = f"data/text/{configs.round}/weight/"
    weight_df = pd.read_csv(f"{weight_file_prefix}{weight_file_name}", encoding="utf-8_sig")

    weight_list = []
    for text_unit_index in range(weight_df.shape[0]):
        text_unit = weight_df["text_unit"].iloc[text_unit_index]
        weight = weight_df["weight"].iloc[text_unit_index]
        if text_unit in configs.punctuation_list:
            weight = 1
        weight_list.append(weight)
    weight_df["weight"] = weight_list

    weight_df.to_csv(f"{weight_file_prefix}{weight_file_name}", encoding="utf-8_sig", index=False)


def output_bias_log(test_str):
    file_path = f"output/alignment/{configs.round}/{configs.device}/"
    file_list = os.listdir(file_path)
    for file_index in range(len(file_list)):
        target_file_path = f"{file_path}{file_list[file_index]}/{test_str}.txt"
        # print(file_list[file_index])
        with open(f"{target_file_path}", "r", encoding="utf-8_sig") as f:
            lines = f.readlines()
            bias_lines = []
            for line_index in range(len(lines)):
                if "BIAS" in lines[line_index]:
                    bias_lines.append(float(lines[line_index].split("BIAS: ")[1]))
            for bias_index in range(len(bias_lines)):
                print(bias_lines[bias_index], end="\t")
            f.close()
            print()


def retrieve_best_H_rectangle_features():
    def compute_radius_between_vectors(vector_1, vector_2):
        dot_product = np.dot(vector_1, vector_2)
        norm_1 = np.linalg.norm(vector_1)
        norm_2 = np.linalg.norm(vector_2)
        cos_theta = dot_product / (norm_1 * norm_2)
        theta_radians = np.arccos(cos_theta)
        theta_angle = np.degrees(theta_radians)
        return theta_angle

    file_path = f"data/modified_gaze_data/{configs.round}/{configs.device}/"
    file_list = os.listdir(file_path)

    feature_list = []
    H_list = []
    for file_index in range(len(file_list)):
        file_name = f"{file_path}{file_list[file_index]}/calibration.csv"

        df_cali = pd.read_csv(file_name)
        centroid_list = analyse_calibration_data.compute_centroids(df_cali)
        list_std_cali = analyse_calibration_data.create_standard_calibration_points()
        H, centroid_list_after_homography = analyse_calibration_data.apply_homography_transform(centroid_list, list_std_cali)

        points1 = np.array([[0, 0], [1920, 0], [1920, 1200], [0, 1200]], dtype=np.float32)
        points2 = cv2.perspectiveTransform(points1.reshape(-1, 1, 2), H).reshape(-1, 2)

        space_1 = cv2.contourArea(points1)
        space_2 = cv2.contourArea(points2)
        space_ratio = space_2 / space_1

        scale_up = np.linalg.norm(points2[0] - points2[1]) / np.linalg.norm(points1[0]-points1[1])
        scale_down = np.linalg.norm(points2[3] - points2[2]) / np.linalg.norm(points1[3] - points1[2])
        scale_left = np.linalg.norm(points2[3] - points2[0]) / np.linalg.norm(points1[3] - points2[0])
        scale_right = np.linalg.norm(points2[2] - points2[1]) / np.linalg.norm(points1[2] - points1[1])

        transform_distance = np.linalg.norm(points1[0] - points2[0]) + np.linalg.norm(points1[1] - points2[1]) + np.linalg.norm(points1[2] - points2[2]) + np.linalg.norm(points1[3] - points2[3])

        angle_left_up_2 = compute_radius_between_vectors(points2[3] - points2[0], points2[1] - points2[0])
        angle_left_down_2 = compute_radius_between_vectors(points2[0] - points2[3], points2[2] - points2[3])
        angle_right_up_2 = compute_radius_between_vectors(points2[2] - points2[1], points2[0] - points2[1])
        angle_right_down_2 = compute_radius_between_vectors(points2[1] - points2[2], points2[3] - points2[2])

        angle_left_up_1 = compute_radius_between_vectors(points1[3] - points1[0], points1[1] - points1[0])
        angle_left_down_1 = compute_radius_between_vectors(points1[0] - points1[3], points1[2] - points1[3])
        angle_right_up_1 = compute_radius_between_vectors(points1[2] - points1[1], points1[0] - points1[1])
        angle_right_down_1 = compute_radius_between_vectors(points1[1] - points1[2], points1[3] - points1[2])

        angle_left_up_ratio = angle_left_up_2 / angle_left_up_1
        angle_left_down_ratio = angle_left_down_2 / angle_left_down_1
        angle_right_up_ratio = angle_right_up_2 / angle_right_up_1
        angle_right_down_ratio = angle_right_down_2 / angle_right_down_1

        cv2ms = cv2.matchShapes(points1.reshape(-1, 1, 2), points2.reshape(-1, 1, 2), 1, 0.0)

        feature_list.append(
            f"name: {file_list[file_index]}, space_ratio: {space_ratio:.5f}, transform_distance: {transform_distance:.5f}, cv2ms: {cv2ms:.5f}, "
            f"scale_up: {scale_up: .5f}, scale_down: {scale_down: .5f}, scale_left: {scale_left: .5f}, scale_right: {scale_right: .5f}, "
            f"angle_left_up_ratio: {angle_left_up_ratio:.5f}, angle_left_down_ratio: {angle_left_down_ratio:.5f}, angle_right_up_ratio: {angle_right_up_ratio:.5f}, angle_right_down_ratio: {angle_right_down_ratio:.5f}")
        H_list.append(H)

    for H in H_list:
        print(H)
    for feature_str in feature_list:
        print(feature_str)
