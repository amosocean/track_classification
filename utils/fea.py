import numpy as np
from multiprocessing.dummy import Pool as TreadPool
import heapq

def top5freqs(input_array):
    fftResult = np.fft.fft(input_array)

    # Get absolute value
    powerSpectrum = np.abs(fftResult)

    # Find the indices of the 5 highest power frequencies
    idx = heapq.nlargest(2, range(len(powerSpectrum)), powerSpectrum.take)

    top_freqs = [input_array[i] for i in idx]
    return top_freqs

def _kinetic_feature(single_sample):
    pt_num = 95
    lat,lon,v,angle,timestep = single_sample[0:5]
    median_T = np.median(np.diff(timestep))
    median_sample_rate = 1/median_T
    theta = angle*np.pi/180
    points = lat+1j*lon
    v_vec = v*np.exp(1j * theta)
    # lat,lon,v,angle 为输入的单个轨迹的
    # 长度
    diff = np.diff(points)
    distances = np.abs(diff)
    dis = np.sum(distances)
    # 最大距离
    distances = np.abs(points[1:] - points[0])
    max_distance = np.max(distances)
    pt_distances = np.percentile(distances,pt_num)
    #主菜比
    pt_zhucai_fea = dis / pt_distances
    zhucai_fea = dis / np.max(distances)
    #坐标点
    start_1 = lat[0]
    start_2 = lon[0]
    end =  len(lat)-1
    mid1 = int(end/2)
    mid2 = int(end/2)
    mid_1 = lat[mid1]
    mid_2 = lon[mid2]
    end_1 = lat[-1]
    end_2 = lon[-1]
    # 速度
    mean_v = np.mean(v)
    max_v = np.percentile(v, 95)
    var_v = np.var(v)
    min_v = np.min(v)
    v_20 = np.percentile(v, 20)
    v_50 = np.percentile(v, 50)
    v_75 = np.percentile(v, 75)
    v_rate = np.mean(np.diff(v))
    pt_v = np.percentile(v,pt_num)
    # 速度变化率(这个不一的好,可以先不用)
    rate_v = np.diff(v) / np.diff(np.arange(1, len(v)+1))
    mean_rate_v = np.mean(rate_v)
    max_rate_v = np.max(rate_v)
    pt_rate_v = np.percentile(rate_v,pt_num)
    var_rate_v = np.var(rate_v)
    min_rate_v = np.min(rate_v)
    
    mid_11 = lat[int(end/4)]
    mid_22 = lon[int(end/4)]
    mid_111 = lat[int(end/4)*3]
    mid_222 = lon[int(end/4)*3]
    
    # #复数速度
    # mean_v = np.mean(v_vec)
    # max_v = np.max(v_vec)
    # pt_v = np.percentile(v_vec,pt_num)
    # # 速度变化率(这个不一的好,可以先不用)
    # rate_v = np.diff(v_vec) / np.diff(np.arange(1, len(v_vec)+1))
    # mean_rate_v = np.mean(rate_v)
    # max_rate_v = np.max(rate_v)
    # pt_rate_v = np.percentile(rate_v,pt_num)
    # var_rate_v = np.var(rate_v)
    # min_rate_v = np.min(rate_v)

    
    #角度与位移角度偏差

    unit_complex = np.exp(1j * theta)
    angle_diff = np.angle(diff/unit_complex[1:])
    angle_diff_max = np.percentile(angle_diff,95)
    angle_diff_mean = np.mean(angle_diff)
    
    #频谱
    spectrum = np.fft.fftshift(np.fft.fft(points))
    pt_spectrum = np.percentile(spectrum,pt_num)
    pt_index = (np.abs(spectrum - pt_spectrum)).argmin()
    pt_freq = (pt_index-len(timestep)/2)/(len(timestep)/2)*median_sample_rate/2
    
    #频谱
    spectrum = np.fft.fftshift(np.fft.fft(v_vec))
    pt_spectrum = np.percentile(spectrum,pt_num)
    pt_index = (np.abs(spectrum - pt_spectrum)).argmin()
    pt_freq_v = (pt_index-len(timestep)/2)/(len(timestep)/2)*median_sample_rate/2
    
    #航向
    mean_angle = np.mean(angle)
    max_angle = np.percentile(angle, 95)
    var_angle = np.var(angle)
    min_angle = np.min(angle)
    angle_20 = np.percentile(angle, 20)
    angle_50 = np.percentile(angle, 50)
    angle_75 = np.percentile(angle, 75)

    #航向变化率
    rate_angle = np.diff(angle) / np.diff(np.arange(len(angle)))
    mean_rate_angle = np.mean(rate_angle)
    max_rate_angle = np.max(rate_angle)
    var_rate_angle = np.var(rate_angle)
    min_rate_angle = np.min(rate_angle)

    #fft
    fft_lat = top5freqs(lat)
    fft_lon = top5freqs(lon)
    fft_v = top5freqs(v)
    fft_angle = top5freqs(angle)

    feature_list = [dis,pt_distances,zhucai_fea,start_1,start_2,mid_1,mid_2,
                    end_1,end_2,
                    #max_v,max_rate_v,
                    mean_v,max_v,min_v,v_50,
                    mean_rate_v,var_rate_v,min_rate_v,
                    pt_v,pt_rate_v,pt_zhucai_fea,
                    # *np.abs([mean_v,max_v,pt_v,mean_rate_v,max_rate_v,pt_rate_v,var_rate_v,min_rate_v]),
                    # *np.angle([mean_v,max_v,pt_v,mean_rate_v,max_rate_v,pt_rate_v,var_rate_v,min_rate_v]),
                    mid_11,
                    mid_22,mid_111,
                    mid_222,
                    angle_diff_max,angle_diff_mean,pt_freq,pt_freq_v,
                    mean_angle,max_angle,var_angle,min_angle,angle_20,angle_50,angle_75,
                    mean_rate_angle, max_rate_angle, var_rate_angle, min_rate_angle,
                    fft_lat[0],fft_lat[1],fft_lon[0],fft_lon[1],fft_v[0],fft_v[1],fft_angle[0],fft_angle[1]
                    ]
    #航向变化率
    # rate_angle = np.diff(angle) / np.diff(np.arange(len(angle)))
    # mean_rate_angle = np.mean(rate_angle)
    # max_rate_angle = np.max(rate_angle)
    # var_rate_angle = np.var(rate_angle)
    # min_rate_angle = np.min(rate_angle)
    # feature_list.extend([mean_rate_angle, max_rate_angle, var_rate_angle, min_rate_angle])
    
    feature = np.array(feature_list)
    
    return feature

def kinetic_feature(sample_list,n_jobs:int = 1):
    if n_jobs == 1: 
        features = map(_kinetic_feature,sample_list)
        return list(features)
    else:
        with TreadPool(processes=n_jobs) as pool:
            features = pool.map(_kinetic_feature,sample_list)
        return list(features)

if __name__ == "__main__":
    sample = np.random.rand(10,6,120)
    
    a=kinetic_feature(sample,n_jobs=1)
    b=1