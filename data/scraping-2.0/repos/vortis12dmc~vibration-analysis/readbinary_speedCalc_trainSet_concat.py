import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import coherence, csd
from scipy import signal
import struct

args = sys.argv
num1 = []
num2 = []

readFile1 = args[1] #振動データ(.bin)
readFile2 = args[2] #振動データ(.bin)

threshold_val = int(args[3])  #信号部分の切り出し閾値

fs = 6400 #サンプリング周波数
#print("Sampling frequency: " + str(fs))
split_readFile1 = readFile1.split('/')

if readFile2 != "nothing":
    split_readFile2 = readFile2.split('/')

#print("Filename1 to read: " + readFile1)
#print("Threshold_val: " + str(threshold_val))

with open(readFile1,'rb') as f:
    data1 = f.read() #読み出し

if readFile2 != "nothing":
    with open(readFile2,'rb') as f:
        data2 = f.read() #読み出し

#data1_d_N = int(len(data1)/2) #data1のデータ数
#data2_d_N = int(len(data2)/2) #data2のデータ数
#print("The number of data1: " + str(data1_d_N))

for i in range(0, len(data1), 2):
    # 2バイト取り出し
    two_bytes = data1[i:i+2]

    #print(i)
    (value_1,) = struct.unpack('<h', two_bytes)
    num1.append(value_1)

if readFile2 != "nothing":
    for i in range(0, len(data2), 2):
        # 2バイト取り出し
        two_bytes = data2[i:i+2]
   
        #print(i)
        (value_2,) = struct.unpack('<h', two_bytes)
        num2.append(value_2)
    num1.extend(num2) #num1にnum2を結合

# 処理：強い振動が起きている範囲の切り出し
# signal1
#print("signal1 処理前: " + str(len(num1)))
start_index = next((i for i, x in enumerate(num1) if x >= threshold_val), None)  # 規定値以上の値が最初に出現するインデックスを検索

if start_index is None:
    # 閾値以上の値が一つも見つからなかった場合、信号が通過していないと判断
    print(split_readFile1[2] +","+split_readFile1[3][0]+ ", notPassed, ")
else:
    # 閾値以上の値が最後に出現するインデックスを検索
    end_index = next((i for i, x in enumerate(reversed(num1)) if x >= threshold_val), None)

    # 閾値以上の値が最後に見つかった場合、リストの末尾からの位置を計算
    if end_index is not None:
        end_index = len(num1) - end_index

    # `end_index` が None または計算された場合に signal1 を抽出
    signal1 = np.array(num1[start_index:end_index])  # 条件に合致する部分リストを抽出
    #print("signal1 処理後: " + str(len(signal1)))

    # 時間の計算と出力
    # 通過にかかった時間（秒）
    passing_time_seconds = len(signal1) / fs
    # split_readFile1[2] の最後から3番目の文字を取得
    third_last_char = split_readFile1[2][-3]

    # third_last_char が U, S, R のどれであるかに応じて処理を変える
    if third_last_char in ['R', 'S']:
        # 車両の長さ（メートル）
        vehicle_length_meters = 201.2
        # 秒速（メートル/秒）
        speed_meters_per_second = vehicle_length_meters / passing_time_seconds
        # 時速（キロメートル/時）
        speed_kilometers_per_hour = speed_meters_per_second * 3.6
        print("{}, {:.2f} km/h, {:.2f} s".format(split_readFile1[2], speed_kilometers_per_hour, passing_time_seconds))

    elif third_last_char == 'U':
        # 車両の長さ（メートル）
        vehicle_length_meters = 154.7
        # 秒速（メートル/秒）
        speed_meters_per_second = vehicle_length_meters / passing_time_seconds
        # 時速（キロメートル/時）
        speed_kilometers_per_hour = speed_meters_per_second * 3.6
        print("{}, {:.2f} km/h, {:.2f} s".format(split_readFile1[2], speed_kilometers_per_hour, passing_time_seconds))

    else:
        # それ以外の場合の処理
        print("{},undefindTrainSetCode, {:.2f} s".format(split_readFile1[2], passing_time_seconds))
