import numpy as np
from multiprocessing.dummy import Pool as TreadPool
def _kinetic_feature(single_sample):
    pt_num = 95
    lat,lon,v,angle,timestep = single_sample[0:5]
    median_T = np.median(np.diff(timestep))
    median_sample_rate = 1/median_T
    points = lat+1j*lon
    # lat,lon,v,angle 为输入的单个轨迹的
    # 长度
    diff = np.diff(points)
    distances = np.abs(diff)
    dis = np.sum(distances)
    # 最大距离
    distances = np.abs(points[1:] - points[0])
    max_distance = np.max(distances)
    pt_distance = np.percentile(distances,pt_num)
    #主菜比
    zhucai_fea = dis / pt_distance
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
    max_v = np.max(v)
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
    
    #角度与位移角度偏差
    theta = angle*np.pi/180
    unit_complex = np.exp(1j * theta)
    angle_diff = np.angle(diff/unit_complex[1:])
    angle_diff_max = np.percentile(angle_diff,95)
    angle_diff_mean = np.mean(angle_diff)
    
    #频谱
    spectrum = np.fft.fftshift(np.fft.fft(points))
    pt_spectrum = np.percentile(spectrum,pt_num)
    pt_index = (np.abs(spectrum - pt_spectrum)).argmin()
    pt_freq = (pt_index-len(timestep)/2)/(len(timestep)/2)*median_sample_rate/2
    
    feature_list = [dis,zhucai_fea,start_1,start_2,mid_1,mid_2,
                    end_1,end_2,mean_v,max_v,pt_v,mean_rate_v,max_rate_v,pt_rate_v,var_rate_v,
                    min_rate_v,mid_11,mid_22,mid_111,mid_222,
                    angle_diff_max,angle_diff_mean,pt_freq]
    
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
    
    print(kinetic_feature(sample,n_jobs=1))