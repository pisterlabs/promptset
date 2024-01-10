import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import coherence, csd
from scipy import signal
import struct


args = sys.argv
num1 = []
num2 = []

readFile1 = args[1]
readFile2 = args[2]
duration = args[3]
writeFilePass = "/gdsfs/gdsfs/mukai/jr/analyTrainData/output/coherence/"

fs = 6400 #サンプリング周波数
axis1 = readFile1[-5].upper()
axis2 = readFile2[-5].upper()
sprit_readFile1 = readFile1.split('/')
sprit_readFile2 = readFile2.split('/')
#sprit_readFile1 = readFile1.split('\\')
#sprit_readFile2 = readFile2.split('\\')

accTitle1 = "Raw " +sprit_readFile1[2]+'/'+sprit_readFile1[3]+'/'+ axis1 + "-axis" 
accTitle2 = "Raw " +sprit_readFile2[2]+'/'+sprit_readFile2[3]+'/'+ axis2 + "-axis"
powTitle1 = "Pow " +sprit_readFile1[2]+'/'+sprit_readFile1[3]+'/'+ axis1 + "-axis"
powTitle2 = "Pow " +sprit_readFile2[2]+'/'+sprit_readFile2[3]+'/'+ axis2 + "-axis"
plotTitle = sprit_readFile1[2]+'/'+sprit_readFile1[3]+'/'+ axis1 + "-axis" +' & '+sprit_readFile2[2]+'/'+sprit_readFile2[3]+'/'+ axis2 + "-axis"
crosTitle = "Cros " +sprit_readFile1[2]+'/'+sprit_readFile1[3]+'/'+ axis1 + "-axis" +'&'+sprit_readFile2[2]+'/'+sprit_readFile2[3]+'/'+ axis2 + "-axis"
coheTitle = "Coherence " +sprit_readFile1[2]+'/'+sprit_readFile1[3]+'/'+ axis1 + "-axis" +'&'+sprit_readFile2[2]+'/'+sprit_readFile2[3]+'/'+ axis2 + "-axis"
writeFile = writeFilePass+sprit_readFile1[2]+'_'+sprit_readFile1[3]+'_'+ axis1 + "axis" +'__'+sprit_readFile2[2]+'_'+sprit_readFile2[3]+'_'+ axis2 + "axis" + "_coherence.png"

print("Filename1 to read: " + readFile1)
print("Filename2 to read: " + readFile2)
print("Duration for each: " + duration)
duration=int(duration)

with open(readFile1,'rb') as f:
    data1 = f.read() #読み出し
f.close()

with open(readFile2,'rb') as f:
    data2 = f.read() #読み出し
f.close()

data1_d_N = int(len(data1)/2) #data1のデータ数
data2_d_N = int(len(data2)/2) #data2のデータ数
print("The number of data1: " + str(data1_d_N))
print("The number of data2: " + str(data2_d_N))

for i in range(0, len(data1), 2):
    # 2バイト取り出し
    two_bytes = data1[i:i+2]

    #print(i)
    (value_1,) = struct.unpack('<h', two_bytes)
    num1.append(value_1)

for i in range(0, len(data2), 2):
    # 2バイト取り出し
    two_bytes = data2[i:i+2]

    #print(i)
    (value_2,) = struct.unpack('<h', two_bytes)
    num2.append(value_2)

signal1=np.array(num1)
signal2=np.array(num2)
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

# クロススペクトラムの計算
f, Pxy = csd(signal1, signal2, fs=fs, nperseg=256, noverlap=128)
Cxy, f_coh = coherence(signal1, signal2, fs=fs, nperseg=256, noverlap=128)

#時間軸の計算
time1 = np.arange(0, duration, 1/(len(signal1)/duration))
time2 = np.arange(0, duration, 1/(len(signal2)/duration))
time1 = time1[:len(signal1)]
time2 = time2[:len(signal2)]

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
ax[0,0].grid(True)

# 2つ目の信号
ax[1,0].plot(time2, signal2)
ax[1,0].set_title(accTitle2)
ax[1,0].set_xlabel('Time (s)')
ax[1,0].set_ylabel('Amplitude')
ax[1,0].grid(True)

# パワースペクトラムのプロット
ax[0,1].set_title(powTitle1)
ax[0,1].plot(freq1[:len(freq1)//2], np.abs(power_spectrum1)[:len(power_spectrum1)//2])
ax[0,1].set_xlabel('Frequency (Hz)')
ax[0,1].set_ylabel('Power')
#ax[2].set_xlim(0,3200)
ax[0,1].grid(True)

ax[1,1].set_title(powTitle2)
ax[1,1].plot(freq2[:len(freq2)//2], np.abs(power_spectrum2)[:len(power_spectrum2)//2])
ax[1,1].set_xlabel('Frequency (Hz)')
ax[1,1].set_ylabel('Power')
#ax[3].set_xlim(0,3200)
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
