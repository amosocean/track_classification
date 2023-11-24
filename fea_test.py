import numpy as np
from scipy import fftpack

# 定义一个函数来获取频率的前五个
def top5freqs(input_array):
    fftResult = np.fft.fft(input_array)
    powerSpectrum = np.abs(fftResult)
    idx = np.argpartition(powerSpectrum, -2)[-2:]
    top_freqs = input_array[idx]
    return top_freqs

# 输入数据
lat = [1,2,3]
lon = [1,2,3]
v = [1,2,3]
angle = [1,2,3]

# 计算距离
distances = np.sqrt(np.diff(lat)**2 + np.diff(lon)**2)
dis = np.sum(distances)
feature = [dis]

# 最大距离
points = np.column_stack((lat, lon))
distances = np.sqrt(np.sum((points[1:] - points[0])**2, axis=1))
max_distance = np.max(distances)
feature.append(max_distance)

# 主菜比
zhucai_fea = dis / max_distance
feature.append(zhucai_fea)

# 起始点
start_1 = lat[0]
start_2 = lon[0]
mid_1 = lat[len(lat)//2]
mid_2 = lon[len(lon)//2]
end_1 = lat[-1]
end_2 = lon[-1]
mid_11 = lat[len(lat)//4]
mid_22 = lon[len(lon)//4]
mid_111 = lat[len(lat)//4*3]
mid_222 = lon[len(lon)//4*3]
feature.extend([start_1, start_2, end_1, end_2, mid_1, mid_2, mid_11, mid_22, mid_111, mid_222]) #12

# 速度
mean_v = np.mean(v)
max_v = np.percentile(v, 95)
var_v = np.var(v)
min_v = np.min(v)
v_20 = np.percentile(v, 20)
v_50 = np.percentile(v, 50)#11, 12, 13, 14, 16, 18, 21, 38,
v_75 = np.percentile(v, 75)
v_rate = np.mean(np.diff(v))
feature.extend([mean_v, max_v, var_v, min_v, v_20, v_50, v_75, v_rate])#20

# 速度变化率
rate_v = np.diff(v) / np.diff(np.arange(len(v)))
mean_rate_v = np.mean(rate_v)
max_rate_v = np.max(rate_v) #21, 38, 40, 41, 42, 43, 44, 45
var_rate_v = np.var(rate_v)
min_rate_v = np.min(rate_v)
feature.extend([mean_rate_v, max_rate_v, var_rate_v, min_rate_v])#24

# 航向
mean_angle = np.mean(angle)
max_angle = np.percentile(angle, 95)
var_angle = np.var(angle)
min_angle = np.min(angle)
angle_20 = np.percentile(angle, 20)
angle_50 = np.percentile(angle, 50)
angle_75 = np.percentile(angle, 75)
feature.extend([mean_angle, max_angle, var_angle, min_angle, angle_20, angle_50, angle_75])#31

# 航向变化率
rate_angle = np.diff(angle) / np.diff(np.arange(len(angle)))
mean_rate_angle = np.mean(rate_angle)
max_rate_angle = np.max(rate_angle)
var_rate_angle = np.var(rate_angle)
min_rate_angle = np.min(rate_angle)
feature.extend([mean_rate_angle, max_rate_angle, var_rate_angle, min_rate_angle])#35

# fft
fft_lat = top5freqs(lat)
fft_lon = top5freqs(lon)
fft_v = top5freqs(v)
fft_angle = top5freqs(angle)
feature.extend([fft_lat, fft_lon, fft_v, fft_angle])

# 选择特征
feature = np.array(feature)[[0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 16, 18, 21, 38, 40, 41, 42, 43, 44, 45]]

