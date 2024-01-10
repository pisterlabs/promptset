import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import coherence, csd
from scipy import signal
import struct
#from matplotlib.ticker import ScalarFormatter
from scipy.signal import find_peaks
import re


#writeFilePass = "/gdsfs/gdsfs/mukai/jr/analyTrainData/output/coherence/"

ylim_range_top = 2057
ylim_range_bottom = -2058

args = sys.argv
arguments_count = len(sys.argv)
if arguments_count < 5:
    print("Usage: "+args[0]+" <信号1の前半> <信号1の後半> <切り出し閾値> <ピーク閾値>")
#    print("Usage: "+args[0]+" <信号1の前半> <信号1の後半>")
    sys.exit(1)

num1 = []

readFile1_1 = args[1]
readFile1_2 = args[2]
threshold_val = int(args[3])
threshold_peak = int(args[4])

fs = 6400 #サンプリング周波数
axis = readFile1_1[-5].upper()
sprit_readFile = readFile1_1.split('/')

accAxis = "Raw " + axis + "-axis"
fftAxis = "FFT " + axis + "-axis"
writeFile = readFile1_1[:-3] + "png"

"""
accTitle1 = "Raw " +sprit_readFile1[2]+'/'+sprit_readFile1[3]+'/'+ axis1 + "-axis" 
accTitle2 = "Raw " +sprit_readFile2[2]+'/'+sprit_readFile2[3]+'/'+ axis2 + "-axis"
powTitle1 = "Pow " +sprit_readFile1[2]+'/'+sprit_readFile1[3]+'/'+ axis1 + "-axis"
powTitle2 = "Pow " +sprit_readFile2[2]+'/'+sprit_readFile2[3]+'/'+ axis2 + "-axis"
plotTitle = sprit_readFile1[2]+'/'+sprit_readFile1[3]+'/'+ axis1 + "-axis" +' & '+sprit_readFile2[2]+'/'+sprit_readFile2[3]+'/'+ axis2 + "-axis_npseg-"+str(npseg)
crosTitle = "Cros " +sprit_readFile1[2]+'/'+sprit_readFile1[3]+'/'+ axis1 + "-axis" +'&'+sprit_readFile2[2]+'/'+sprit_readFile2[3]+'/'+ axis2 + "-axis"
coheTitle = "Coherence " +sprit_readFile1[2]+'/'+sprit_readFile1[3]+'/'+ axis1 + "-axis" +'&'+sprit_readFile2[2]+'/'+sprit_readFile2[3]+'/'+ axis2 + "-axis"
#writeFile = writeFilePass+sprit_readFile1[2]+'_'+sprit_readFile1[3]+'_'+ axis1 + "axis" +'__'+sprit_readFile2[2]+'_'+sprit_readFile2[3]+'_'+ axis2 + "axis" + "_coherence_npseg-"+str(npseg)+".png"
"""
"""
print("Filename1 to read: " + readFile1)
print("Filename2 to read: " + readFile2)
"""

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
#print("Sampling frequency: " + str(fs_1))
#print("Sampling frequency: " + str(fs_2))

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

"""
print("nperseg: " + str(npseg))
nlap = int(int(npseg)/2)
print("noverlap: " + str(nlap))

# クロススペクトラムの計算
f, Pxy = csd(signal1, signal2, fs=fs, nperseg=npseg, noverlap=nlap)
Cxy, f_coh = coherence(signal1, signal2, fs=fs, nperseg=npseg, noverlap=nlap)
"""

#時間軸の計算
time1 = np.arange(0, len(signal1)/fs, 1/fs)
time1 = time1[:len(signal1)]
print("PassingTime 1: "+ str(len(signal1)/fs) +" s")

# 周波数軸の計算
freq1 = np.fft.fftfreq(len(signal1), d=1/fs)

"""
#図へのプロット
fig = plt.figure(figsize = (10,5))

fig.suptitle('[' + sprit_readFile[2] + '] [' + sprit_readFile[3] + '] ' + '[' + axis + '] ' + sprit_readFile[4][:-8])
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax1.plot(time1, raw_signal)
ax1.grid()
ax1.set_title(accAxis, loc='center')
ax1.set_ylim(ylim_range_bottom, ylim_range_top)
ax1.set_xlabel("Time[s]")
ax1.set_ylabel("Acceleration")

ax2.plot(freq1[:len(freq1)//2], np.abs(fft_result1)[:len(fft_result1)//2])
#ax2.plot(freq1[1:int(data1_d_N/2)], fft_result1[1:int(data1_d_N/2)])
ax2.grid()
ax2.set_title(fftAxis, loc='center')
ax2.set_xlabel("Freqency[Hz]")
ax2.set_ylabel("Amplitude")

plt.tight_layout()
#plt.show()
#print(a)

print("Filename to write: " + writeFile)
plt.savefig(writeFile)
plt.close()
"""

# ピークを検出
peaks1, _ = find_peaks(power_spectrum1, height=threshold_val)
# ピークの値とインデックスを取得
peak1_values = power_spectrum1[peaks1]
peak1_indices = freq1[peaks1]
#print("Sensor1:" + re.sub('_x.bin|_y.bin|_z.bin', '', sprit_readFile1[9]))
# ピークの値とインデックスを表示
for value, index in zip(peak1_values, peak1_indices):
    if(index > 0):
                print(f"ピーク値: {value}, 周波数: {index}[Hz]")
