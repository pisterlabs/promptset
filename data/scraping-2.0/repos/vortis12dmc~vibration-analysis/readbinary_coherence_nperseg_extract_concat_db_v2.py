import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import coherence, csd
from scipy import signal
import struct

#writeFilePass = "/2021/v2158218/analyTrainData/output/coherence_db/"
writeFilePass = "/gdsfs/gdsfs/mukai/jr/analyTrainData/output/coherence_db/"

reference_value = 0.000001  #dB変換の 基準値を1マイクロメートルとする(ISO規格)

ylim_range_top = 2057
ylim_range_bottom = -2058

ylim_range_top_db = 200
ylim_range_bottom_db = 40

args = sys.argv
arguments_count = len(sys.argv)
if arguments_count < 8:
    print("Usage: "+args[0]+" <信号1の前半> <信号1の後半> <信号2の前半> <信号2の後半> <nperseg> <閾値1> <閾値2>")
    sys.exit(1)

num1 = []
num2 = []

readFile1_1 = args[1]
readFile1_2 = args[2]
readFile2_1 = args[3]
readFile2_2 = args[4]
npseg = int(args[5])
threshold_val1 = int(args[6])  #30  #信号部分の切り出し閾値
threshold_val2 = int(args[7])  #30  #信号部分の切り出し閾値

fs = 6400 #サンプリング周波数
axis1 = readFile1_1[-5].upper()
axis2 = readFile2_1[-5].upper()
sprit_readFile1 = readFile1_1.split('/')
sprit_readFile2 = readFile2_1.split('/')
#sprit_readFile1 = readFile1.split('\\')
#sprit_readFile2 = readFile2.split('\\')

accTitle1 = "Raw " +sprit_readFile1[7]+'/'+sprit_readFile1[8]+'/'+ axis1 + "-axis" 
accTitle2 = "Raw " +sprit_readFile2[7]+'/'+sprit_readFile2[8]+'/'+ axis2 + "-axis"
powTitle1 = "Pow " +sprit_readFile1[7]+'/'+sprit_readFile1[8]+'/'+ axis1 + "-axis"
powTitle2 = "Pow " +sprit_readFile2[7]+'/'+sprit_readFile2[8]+'/'+ axis2 + "-axis"
plotTitle = sprit_readFile1[7]+'/'+sprit_readFile1[8]+'/'+ axis1 + "-axis" +' & '+sprit_readFile2[7]+'/'+sprit_readFile2[8]+'/'+ axis2 + "-axis_npseg-"+str(npseg)
#crosTitle = "Cros " +sprit_readFile1[7]+'/'+sprit_readFile1[8]+'/'+ axis1 + "-axis" +'&'+sprit_readFile2[7]+'/'+sprit_readFile2[8]+'/'+ axis2 + "-axis"
#coheTitle = "Coherence " +sprit_readFile1[7]+'/'+sprit_readFile1[8]+'/'+ axis1 + "-axis" +'&'+sprit_readFile2[7]+'/'+sprit_readFile2[8]+'/'+ axis2 + "-axis"
writeFile = writeFilePass+sprit_readFile1[7]+'_'+sprit_readFile1[8]+'_'+ axis1 + "axis" +'__'+sprit_readFile2[7]+'_'+sprit_readFile2[8]+'_'+ axis2 + "axis" + "_coherence_npseg-"+str(npseg)+".png"

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

#信号2
with open(readFile2_1,'rb') as f:
    data2_1 = f.read() #読み出し
for i in range(0, len(data2_1), 2):
    # 2バイト取り出し
    two_bytes = data2_1[i:i+2]
    #print(i)
    (value_2,) = struct.unpack('<h', two_bytes)
    num2.append(value_2)

if readFile2_2 != "none":
    with open(readFile2_2,'rb') as f:
        data2_2 = f.read() #読み出し
    for i in range(0, len(data2_2), 2):
        # 2バイト取り出し
        two_bytes = data2_2[i:i+2]
        #print(i)
        (value_2,) = struct.unpack('<h', two_bytes)
        num2.append(value_2)

#data1_d_N = int(len(data1)/2) #data1のデータ数
#data2_d_N = int(len(data2)/2) #data2のデータ数
data1_d_N = len(num1)  # num1のデータ数
data2_d_N = len(num2)  # num2のデータ数
print("The number of data1: " + str(data1_d_N))
print("The number of data2: " + str(data2_d_N))

print("signal1 処理前: "+str(len(num1)))
# 処理：強い振動が起きている範囲の切り出し
# 規定値以上の値が最初に出現するインデックスを検索
start_index = next((i for i, x in enumerate(num1) if x >= threshold_val1), None)
# 規定値以上の値が最後に出現するインデックスを検索
end_index = len(num1) - next((i for i, x in enumerate(reversed(num1)) if x >= threshold_val1), None)
# 条件に合致する部分リストを抽出
signal1 = np.array(num1[start_index:end_index])

print("signal1 処理後: "+str(len(signal1)))
print("signal2 処理前: "+str(len(num2)))

# 同様の処理
start_index = next((i for i, x in enumerate(num2) if x >= threshold_val2), None)
end_index = len(num2) - next((i for i, x in enumerate(reversed(num2)) if x >= threshold_val2), None)
signal2 = np.array(num2[start_index:end_index])

print("signal2 処理後: "+str(len(signal2)))

#fs_1 = (int)((data1_d_N)/(int(duration)*2))*2 #サンプリング周波数 (実際の値)
#dt=1/fs_1

print("Sampling frequency: " + str(fs))
#print("Sampling frequency: " + str(fs_1))
#print("Sampling frequency: " + str(fs_2))

# フーリエ変換
fft_result1 = np.fft.fft(signal1)
fft_result2 = np.fft.fft(signal2)

# パワースペクトラム
power_spectrum1 = np.abs(fft_result1)**2
power_spectrum2 = np.abs(fft_result2)**2

print("nperseg: " + str(npseg))
nlap = int(int(npseg)/2)
print("noverlap: " + str(nlap))

# クロススペクトラムの計算
f, Pxy = csd(signal1, signal2, fs=fs, nperseg=npseg, noverlap=nlap)
Cxy, f_coh = coherence(signal1, signal2, fs=fs, nperseg=npseg, noverlap=nlap)

#時間軸の計算
#time1 = np.arange(0, duration, 1/(len(signal1)/duration))
#time2 = np.arange(0, duration, 1/(len(signal2)/duration))
#time1 = np.arange(0, len(signal1)/fs, 1/(len(signal1)/int(len(signal1)/fs)))
time1 = np.arange(0, len(signal1)/fs, 1/fs)
#time1 = np.arange(0, len(signal1)/fs, 1/(len(signal1)/len(signal1)/fs))
#time2 = np.arange(0, int(len(signal2)/fs), 1/(len(signal2)/int(len(signal2)/fs)))
time2 = np.arange(0, len(signal2)/fs, 1/fs)
#time2 = np.arange(0, len(signal2)/fs, 1/(len(signal2)/len(signal2)/fs))
time1 = time1[:len(signal1)]
time2 = time2[:len(signal2)]
print("PassingTime 1: "+ str(len(signal1)/fs) +" s")
print("PassingTime 2: "+ str(len(signal2)/fs) +" s")

# 周波数軸の計算
freq1 = np.fft.fftfreq(len(signal1), d=1/fs)
freq2 = np.fft.fftfreq(len(signal2), d=1/fs)

# グラフのプロット
fig, ax = plt.subplots(2, 3, figsize=(15, 10))

plt.suptitle(plotTitle, fontsize=16)

# 1つ目の信号
ax[0,0].plot(time1, signal1)
ax[0,0].set_title(accTitle1)
ax[0,0].set_xlabel('Time (s)')
ax[0,0].set_ylabel('Amplitude')
ax[0,0].set_ylim(ylim_range_bottom, ylim_range_top)
ax[0,0].grid(True)

# 2つ目の信号
ax[1,0].plot(time2, signal2)
ax[1,0].set_title(accTitle2)
ax[1,0].set_xlabel('Time (s)')
ax[1,0].set_ylabel('Amplitude')
ax[1,0].set_ylim(ylim_range_bottom, ylim_range_top)
ax[1,0].grid(True)

# デシベル値に変換
power_in_dB1 = 10 * np.log10(np.abs(power_spectrum1) / reference_value)
power_in_dB2 = 10 * np.log10(np.abs(power_spectrum2) / reference_value)

# パワースペクトラムのプロット
ax[0,1].set_title(powTitle1)
ax[0,1].plot(freq1[:len(freq1)//2], power_in_dB1[:len(power_spectrum1)//2])
ax[0,1].set_xlabel('Frequency [Hz]')
ax[0,1].set_ylabel('Power [dB]')
ax[0,1].set_ylim(ylim_range_bottom_db, ylim_range_top_db)
ax[0,1].grid(True)

ax[1,1].set_title(powTitle2)
ax[1,1].plot(freq2[:len(freq2)//2], power_in_dB2[:len(power_spectrum2)//2])
ax[1,1].set_xlabel('Frequency [Hz]')
ax[1,1].set_ylabel('Power [dB]')
ax[1,1].set_ylim(ylim_range_bottom_db, ylim_range_top_db)
ax[1,1].grid(True)

# クロススペクトラムの振幅
ax[0,2].plot(f, np.abs(Pxy))
#ax[0,2].set_title(crosTitle)
ax[0,2].set_title('Cross-Spectrum Amplitude')
ax[0,2].set_xlabel('Frequency (Hz)')
ax[0,2].set_ylabel('Amplitude')
ax[0,2].grid(True)

#コヒーレンス
ax[1,2].plot(Cxy,f_coh)
#ax[1,2].set_title(coheTitle)
ax[1,2].set_title('Coherence')
ax[1,2].set_xlabel('Frequency (Hz)')
ax[1,2].set_ylabel('Coherence')
ax[1,2].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
print("Filename to write: " + writeFile)
plt.savefig(writeFile)
plt.show()
plt.close()

