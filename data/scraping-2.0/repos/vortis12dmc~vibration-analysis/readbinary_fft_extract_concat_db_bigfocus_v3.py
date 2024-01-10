import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import coherence, csd
from scipy import signal
import struct
from matplotlib.ticker import ScalarFormatter


#writeFilePass = "/gdsfs/gdsfs/mukai/jr/analyTrainData/output/coherence/"

reference_value = 0.000001  #dB変換の 基準値を1マイクロとする(ISO規格)

ylim_range_top = 2057
ylim_range_bottom = -2058

ylim_range_top_db = 200
ylim_range_bottom_db = 40

args = sys.argv
arguments_count = len(sys.argv)
if arguments_count < 4:
    print("Usage: "+args[0]+" <信号1の前半> <信号1の後半> <閾値> <始点> <終点>")
#    print("Usage: "+args[0]+" <信号1の前半> <信号1の後半>")
    sys.exit(1)

num1 = []

readFile1_1 = args[1]
readFile1_2 = args[2]
threshold_val = int(args[3])
# x軸の上限と下限についてはデータに応じて適宜変更
xlim_range_top = float(args[5])
xlim_range_bottom = float(args[4])

fs = 6400 #サンプリング周波数
axis = readFile1_1[-5].upper()
sprit_readFile = readFile1_1.split('/')

accAxis = "Raw " + axis + "-axis"
fftAxis = "FFT " + axis + "-axis"
powAxis = "POW " + axis + "-axis"
writeFile = readFile1_1[:-4] + "_db_bigfocus_"+ str(xlim_range_bottom) + "-" +str(xlim_range_top) +".png"

#信号1
with open(readFile1_1,'rb') as f:
    data1_1 = f.read() #読み出し

for i in range(0, len(data1_1), 2):
    # 2バイト取り出し
    two_bytes = data1_1[i:i+2]
    #print(i)
    (value_1,) = struct.unpack('<h', two_bytes)
    num1.append(value_1)

if readFile1_2 != "none": #信号1の結合作業
    with open(readFile1_2,'rb') as f:
        data1_2 = f.read() #読み出し
    for i in range(0, len(data1_2), 2):
        # 2バイト取り出し
        two_bytes = data1_2[i:i+2]
        #print(i)
        (value_1,) = struct.unpack('<h', two_bytes)
        num1.append(value_1)


#data1_d_N = int(len(data1)/2) #data1のデータ数
#data2_d_N = int(len(data2)/2) #data2のデータ数
data1_d_N = len(num1)  # num1のデータ数
print("The number of data1: " + str(data1_d_N))

signal1=np.array(num1)


print("signal1 処理前: "+str(len(num1)))
# 処理：強い振動が起きている範囲の切り出し
# 規定値以上の値が最初に出現するインデックスを検索
start_index = next((i for i, x in enumerate(num1) if x >= threshold_val), None)
# 規定値以上の値が最後に出現するインデックスを検索
end_index = len(num1) - next((i for i, x in enumerate(reversed(num1)) if x >= threshold_val), None)
# 条件に合致する部分リストを抽出
signal1 = np.array(num1[start_index:end_index])
data1_d_N = len(signal1)
print("signal1 処理後: "+str(len(signal1)))



#fs_1 = (int)((data1_d_N)/(int(duration)*2))*2 #サンプリング周波数 (実際の値)
#dt=1/fs_1

print("Sampling frequency: " + str(fs))

"""
#窓関数の準備
fw1 = signal.hann(data1_d_N)
#fw2 = signal.hamming(N)
#fw3 = signal.blackman(N)

raw_signal = signal1
signal1 = signal1 * fw1

# フーリエ変換
fft_result1 = np.fft.fft(signal1)

# パワースペクトラム
power_spectrum1 = np.abs(fft_result1)**2

# 周波数軸の計算
freq1 = np.fft.fftfreq(len(signal1), d=1/fs)
"""

#時間軸の計算
time1 = np.arange(0, len(signal1)/fs, 1/fs)
time1 = time1[:len(signal1)]
print("PassingTime 1: "+ str(len(signal1)/fs) +" s")

# xlim_range_topとxlim_range_bottomの範囲で加速度データを切り出し、窓関数をかけてFFTする
trimmed_signal = signal1[(time1 >= xlim_range_bottom) & (time1 <= xlim_range_top)]

# 窓関数の準備
fw = signal.hann(len(trimmed_signal))

# 窓関数をかける
trimmed_signal_windowed = trimmed_signal * fw

# フーリエ変換
fft_result_trimmed = np.fft.fft(trimmed_signal_windowed)

# パワースペクトラム
power_spectrum_trimmed = np.abs(fft_result_trimmed)**2


"""
# 時間軸の計算
time_trimmed = np.arange(0, len(trimmed_signal_windowed)/fs, 1/fs)
time_trimmed = time_trimmed[:len(trimmed_signal_windowed)]
"""
# 周波数軸の計算
freq_trimmed = np.fft.fftfreq(len(trimmed_signal_windowed), d=1/fs)

#図へのプロット
fig = plt.figure(figsize = (10,5))

fig.suptitle('[' + sprit_readFile[2] + '] [' + sprit_readFile[3] + '] ' + '[' + axis + '] ' + sprit_readFile[4][:-8])
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax1.plot(time1, signal1)
ax1.grid()
ax1.set_title(accAxis, loc='center')
ax1.set_xlim(xlim_range_bottom, xlim_range_top)
ax1.set_ylim(ylim_range_bottom, ylim_range_top)
ax1.set_xlabel("Time[s]")
ax1.set_ylabel("Acceleration")

# デシベル値に変換
power_in_dB_trimmed = 10 * np.log10(np.abs(power_spectrum_trimmed) / reference_value)

# 切り出した加速度データのFFT結果をプロット
ax2.plot(freq_trimmed[:len(freq_trimmed)//2], power_in_dB_trimmed[:len(power_in_dB_trimmed)//2])
ax2.grid()
ax2.set_title(powAxis, loc='center')
ax2.set_ylim(ylim_range_bottom_db, ylim_range_top_db)
ax2.set_xlabel("Frequency [Hz]")
ax2.set_ylabel("Power [dB]")

plt.tight_layout()
#plt.show()
#print(a)

print("Filename to write: " + writeFile)
plt.savefig(writeFile)
plt.close()

