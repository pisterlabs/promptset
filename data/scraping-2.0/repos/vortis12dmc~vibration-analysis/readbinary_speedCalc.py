import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import coherence, csd
from scipy import signal
import struct

args = sys.argv
num1 = []

readFile1 = args[1] #振動データ(.bin)
threshold_val = int(args[2])  #信号部分の切り出し閾値

fs = 6400 #サンプリング周波数
print("Sampling frequency: " + str(fs))
sprit_readFile1 = readFile1.split('/')

print("Filename1 to read: " + readFile1)
print("Threshold_val: " + str(threshold_val))

with open(readFile1,'rb') as f:
    data1 = f.read() #読み出し
f.close()

data1_d_N = int(len(data1)/2) #data1のデータ数
print("The number of data1: " + str(data1_d_N))

for i in range(0, len(data1), 2):
    # 2バイト取り出し
    two_bytes = data1[i:i+2]

    #print(i)
    (value_1,) = struct.unpack('<h', two_bytes)
    num1.append(value_1)

# 処理：強い振動が起きている範囲の切り出し
# signal1
print("signal1 処理前: " + str(len(num1)))
start_index = next((i for i, x in enumerate(num1) if x >= threshold_val), None)  # 規定値以上の値が最初に出現するインデックスを検索

if start_index is None:
    # 閾値以上の値が一つも見つからなかった場合、信号が通過していないと判断
    print(sprit_readFile1[2] + ": not passed")
else:
    # 閾値以上の値が最後に出現するインデックスを検索
    end_index = next((i for i, x in enumerate(reversed(num1)) if x >= threshold_val), None)

    # 閾値以上の値が最後に見つかった場合、リストの末尾からの位置を計算
    if end_index is not None:
        end_index = len(num1) - end_index

    # `end_index` が None または計算された場合に signal1 を抽出
    signal1 = np.array(num1[start_index:end_index])  # 条件に合致する部分リストを抽出
    print("signal1 処理後: " + str(len(signal1)))

    # 時間の計算と出力
    print(sprit_readFile1[2] + ": " + str(len(signal1) / fs) + " s")

"""
# 処理：強い振動が起きている範囲の切り出し
#signal1
print("signal1 処理前: "+str(len(num1)))
start_index = next((i for i, x in enumerate(num1) if x >= threshold_val), None) # 規定値以上の値が最初に出現するインデックスを検索
if start_index is None:
    print(sprit_readFile1[2] + ": " + " not passed")
else:
    end_index = next((i for i, x in enumerate(reversed(num1)) if x >= threshold_val), None) # 規定値以上の値が最後に出現するインデックスを検索
    if end_index is None:
        end_index = len(num1)  # ここでリストの末尾を指定
    else:
        end_index = len(num1) - end_index  # リストの末尾からの位置をリスト全体の長さから引いて正しいインデックスを得る
    signal1 = np1.array(num1[start_index:end_index]) # 条件に合致する部分リストを抽出
    print("signal1 処理後: "+str(len(signal1)))
    # 時間の計算
    print(sprit_readFile1[2] + ": " + str(len(signal1)/fs) +" s")
"""
