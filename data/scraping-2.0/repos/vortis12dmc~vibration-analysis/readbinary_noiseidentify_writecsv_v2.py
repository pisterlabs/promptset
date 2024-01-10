import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import coherence, csd
from scipy import signal
from scipy.signal import find_peaks
import struct
import re
import csv
import os
import glob

args = sys.argv

# 引数が与えられなかった場合のエラーメッセージ
if len(args) != 2:
    print("Usage: python3 readbinary_noiseidentify_writecsv.py <file_path>")
    sys.exit(1)

# バイナリファイルが格納されているディレクトリへのパス
data_directory_path = args[1]

# データを書き込むディレクトリへのパス
#write_directory_path = "/2021/v2158218/analyTrainData/output/"
write_directory_path = "/gdsfs/gdsfs/mukai/jr/analyTrainData/output/peakVal/"

# 辞書を使用してファイル名をマッピング（ディレクトリパスを含む）
file_mapping = {
    "355.1_rail": os.path.join(write_directory_path, "355.1_rail_noiseidentify.csv"),
    "355.2_bond": os.path.join(write_directory_path, "355.2_bond_noiseidentify.csv"),
    "355.2_rail": os.path.join(write_directory_path, "355.2_rail_noiseidentify.csv"),
    "356.6_rail": os.path.join(write_directory_path, "356.6_rail_noiseidentify.csv"),
    "356.7_bond": os.path.join(write_directory_path, "356.7_bond_noiseidentify.csv"),
    "356.7_rail": os.path.join(write_directory_path, "356.7_rail_noiseidentify.csv")
}

# 対象のバイナリファイルパターン
file_pattern = os.path.join(data_directory_path, "*.bin")

# バイナリファイルのリストを取得
binary_files = glob.glob(file_pattern)

# 閾値
threshold = 0.1e+8

for file in binary_files:

    # それぞれのファイルをプログラムで使用する箇所に適用する
    readFile = file

    num = []
    fs = 6400 #サンプリング周波数
    axis = readFile[-5].upper()
    sprit_readFile = readFile.split('/')
    #sprit_readFile = readFile.split('\\')

    # 振動測定時刻（タイムスタンプ）
    time_stamp = re.sub('_x.bin|_y.bin|_z.bin', '', sprit_readFile[9])
    #time_stamp = re.sub('_x.bin|_y.bin|_z.bin', '', sprit_readFile[4])
    
    # センサー名 取得
    input_argument = sprit_readFile[7]
    #input_argument = sprit_readFile[3]

    with open(readFile,'rb') as f:
        data = f.read() #読み出し
        f.close()

    data_d_N = int(len(data)/2) #dataのデータ数

    for i in range(0, len(data), 2):
        # 2バイト取り出し
        two_bytes = data[i:i+2]

        #print(i)
        (value_1,) = struct.unpack('<h', two_bytes)
        num.append(value_1)

    signal=np.array(num)
    #fs_1 = (int)((data1_d_N)/(int(duration)*2))*2 #サンプリング周波数 (実際の値)
    #dt=1/fs_1

    # フーリエ変換
    fft_result = np.fft.fft(signal)

    # パワースペクトラム
    power_spectrum = np.abs(fft_result)**2

    # 周波数軸の計算
    freq = np.fft.fftfreq(len(signal), d=1/fs)

    # ピークを検出
    peaks, _ = find_peaks(power_spectrum, height=threshold)

    # ピークの値と周波数を取得
    peak_values = power_spectrum[peaks]
    peak_indices = freq[peaks]

    # CSVファイルに書き込むデータの組を格納する変数
    data_to_write = []

    # 入力された引数がマッピングにあるか確認し、該当するファイルに新規書き込みまたは追記
    if input_argument in file_mapping:
        filename = file_mapping[input_argument]

        # ファイルが存在しない場合は新規作成
        if not os.path.exists(filename):
            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file)
                # 必要に応じてヘッダーを書き込む
                writer.writerow(['Timestamp', 'Axis', 'Frequency', 'Value'])

        # CSVファイルに書き込むデータの組を用意
        for i in range(len(peaks)):
            if peak_indices[i] > 0:
                data_to_write.append([time_stamp, axis, peak_indices[i], peak_values[i]])

        # 既に書かれているデータを読み取る
        existing_data = set()
        with open(filename, 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 2:  # 必要に応じて条件を調整
                    existing_data.add((row[0], row[1]))  # 1列目と2列目の組み合わせをセットに追加

        # 新規データが既存のデータと重複しているかチェック
        for new_row in data_to_write:
            if (new_row[0], new_row[1]) in existing_data:
                print("指定したデータは既に書き込まれています。")
                sys.exit(0)

        # 重複がなければ追記
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data_to_write)
    else:
        print("無効な引数です。")

