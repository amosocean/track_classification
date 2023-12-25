import numpy as np
from multiprocessing.dummy import Pool as TreadPool
import heapq
import torch
import time
def top5freqs(input_array):
    x = input_array
    fftResult = torch.fft.fft(x)

    # Get absolute value
    powerSpectrum = torch.abs(fftResult)

    # Find the indices of the 5 highest power frequencies
    idx = torch.topk(powerSpectrum, k=2, dim=-1).indices
    top_freqs = idx/input_array.shape[0]

    return top_freqs

def _kinetic_feature(single_sample):
    pt_num = 0.95
    lat, lon, height,v, timestep = single_sample[0:5]
    # lat = single_sample[:,0,:] 
    # lon = single_sample[:,1,:] 
    # v = single_sample[:,2,:] 
    # angle = single_sample[:,3,:] 
    # timestep = single_sample[:,4,:]
    timestep = timestep-timestep[0]
    time_diff = torch.diff(timestep)
    median_T = torch.median(time_diff)
    median_sample_rate = 1 / median_T
    #theta = angle * np.pi / 180
    points = lat + 1j * lon
    #v_vec = v * torch.exp(1j * theta)

    diff = torch.diff(points)
    # if torch.any(diff.imag > 350):
    #     index1 = torch.where(diff.imag > 350)
    #     diff.imag[index1] = diff.imag[index1] - 360
    # if torch.any(diff.imag < -350):
    #     index1 = torch.where(diff.imag < -350)
    #     diff.imag[index1] = diff.imag[index1] + 360

    points = torch.roll(points, 1)
    distances = torch.abs(diff)
    dis = torch.sum(distances)
    distances = torch.abs(points[1:] - points[0])
    pt_distances = torch.quantile(distances, pt_num)

    start_1 = lat[0]
    start_2 = lon[0]
    end = lat.shape[0] - 1
    mid1 = int(end / 2)
    mid2 = int(end / 2)
    mid_1 = lat[mid1]
    mid_2 = lon[mid2]
    end_1 = lat[-1]
    end_2 = lon[-1]

    mean_v = torch.mean(v)
    max_v = torch.quantile(v, 0.95)
    min_v = torch.min(v)
    v_50 = torch.quantile(v, 0.5)

    pt_v = torch.quantile(v, pt_num)

    rate_v = torch.diff(v)/time_diff
    mean_rate_v = torch.mean(rate_v)
    pt_rate_v = torch.quantile(rate_v, pt_num)
    var_rate_v = torch.var(rate_v)
    min_rate_v = torch.min(rate_v)

    mid_11 = lat[int(end / 4)]
    mid_22 = lon[int(end / 4)]
    mid_111 = lat[int(end / 4) * 3]
    mid_222 = lon[int(end / 4) * 3]

    # unit_complex = torch.exp(1j * theta)
    # angle_diff = torch.angle(diff / unit_complex[1:])
    # angle_diff_max = torch.quantile(angle_diff, 0.95)
    # angle_diff_mean = torch.mean(angle_diff)

    spectrum = torch.fft.fftshift(torch.fft.fft(points))
    pt_spectrum = torch.quantile(torch.abs(spectrum), pt_num)
    pt_index = (torch.abs(spectrum - pt_spectrum)).argmin()
    pt_freq = (pt_index - (timestep.shape[0]) / 2) / ((timestep.shape[0]) / 2) * median_sample_rate 

    spectrum = torch.fft.fftshift(torch.fft.fft(v))
    pt_spectrum = torch.quantile(torch.abs(spectrum), pt_num)
    pt_index = (torch.abs(spectrum - pt_spectrum)).argmin()
    pt_freq_v = (pt_index - (timestep.shape[0]) / 2) / ((timestep.shape[0]) / 2) * median_sample_rate 

    mean_height = torch.mean(height)
    max_height = torch.quantile(height, 0.95)
    min_height = torch.min(height)
    height_50 = torch.quantile(height, 0.5)

    pt_height = torch.quantile(height, pt_num)

    rate_height = torch.diff(height)/time_diff
    mean_rate_height = torch.mean(rate_height)
    pt_rate_height = torch.quantile(rate_height, pt_num)
    var_rate_height = torch.var(rate_height)
    min_rate_height = torch.min(rate_height)
    
    spectrum = torch.fft.fftshift(torch.fft.fft(height))
    pt_spectrum = torch.quantile(torch.abs(spectrum), pt_num)
    pt_index = (torch.abs(spectrum - pt_spectrum)).argmin()
    pt_freq_height = (pt_index - (timestep.shape[0]) / 2) / ((timestep.shape[0]) / 2) * median_sample_rate 
    
    # mean_angle = torch.mean(angle)
    # max_angle = torch.quantile(angle, 0.95)
    # var_angle = torch.var(angle)
    # min_angle = torch.min(angle)
    # angle_20 = torch.quantile(angle, 0.2)
    # angle_50 = torch.quantile(angle, 0.5)
    # angle_75 = torch.quantile(angle, 0.75)

    # rate_angle = torch.diff(angle) 
    # mean_rate_angle = torch.mean(rate_angle)
    # max_rate_angle = torch.max(rate_angle)
    # var_rate_angle = torch.var(rate_angle)
    # min_rate_angle = torch.min(rate_angle)

    fft_lat = top5freqs(lat)*median_sample_rate
    fft_lon = top5freqs(lon)*median_sample_rate
    fft_v = top5freqs(v)*median_sample_rate
    #fft_angle = top5freqs(angle)*median_sample_rate
    fft_height = top5freqs(height)*median_sample_rate

    feature_list = [
        dis, pt_distances, start_1, start_2, mid_1, mid_2, end_1, end_2,
        mean_v, max_v, min_v, v_50, mean_rate_v, var_rate_v, min_rate_v,
        pt_v, pt_rate_v, mid_11, mid_22, mid_111, mid_222,
        #angle_diff_max, angle_diff_mean, 
        pt_freq, pt_freq_v,
        # mean_angle, max_angle, var_angle, min_angle, angle_20, angle_50, angle_75,
        # mean_rate_angle, max_rate_angle, var_rate_angle, min_rate_angle,
        mean_height,max_height,min_height,height_50,pt_height,
        mean_rate_height,pt_rate_height,var_rate_height,min_rate_height,
        pt_freq_height,
        fft_lat[0], fft_lat[1], fft_lon[0], fft_lon[1], fft_v[0], fft_v[1],
        fft_height[0],fft_height[1]
        #fft_angle[0], fft_angle[1]
    ]
    feature = torch.stack(feature_list,dim=0)
    
    return feature.cpu()

def _kinetic_feature_vec(single_sample):
    lat = single_sample[...,:,0,:] 
    lon = single_sample[...,:,1,:] 
    v = single_sample[...,:,2,:] 
    angle = single_sample[...,:,3,:] 
    timestep = single_sample[...,:,4,:]
    timestep = timestep - timestep[...,:,0].unsqueeze(dim=-1)
    pt_num = 0.95
    median_T = torch.mean(torch.diff(timestep,dim=-1),dim=-1,keepdim=True)
    median_sample_rate = 1 / median_T
    theta = angle * np.pi / 180
    points = torch.complex(lat,lon)
    v_vec = v * torch.exp(1j * theta)

    diff = torch.diff(points,dim=-1)
    # if torch.any(diff.imag > 350):
    #     index1 = torch.where(diff.imag > 350)
    #     diff.imag[index1] = diff.imag[index1] - 360
    # if torch.any(diff.imag < -350):
    #     index1 = torch.where(diff.imag < -350)
    #     diff.imag[index1] = diff.imag[index1] + 360

    points = torch.roll(points, 1,dims=-1)
    distances = torch.abs(diff)
    dis = torch.sum(distances,dim=-1,keepdim=True)
    distances = torch.abs(points[...,1:] - points[...,0].unsqueeze(dim=-1))
    pt_distances = torch.quantile(distances, pt_num,dim=-1,keepdim=True)

    start_1 = lat[...,0].unsqueeze(dim=-1)
    start_2 = lon[...,0].unsqueeze(dim=-1)
    end = lat.shape[-1] - 1
    mid_index = int(end / 2)
    mid_1 = lat[...,mid_index].unsqueeze(dim=-1)
    mid_2 = lon[...,mid_index].unsqueeze(dim=-1)
    end_1 = lat[...,-1].unsqueeze(dim=-1)
    end_2 = lon[...,-1].unsqueeze(dim=-1)

    mean_v = torch.mean(v,dim=-1,keepdim=True)
    max_v = torch.quantile(v, 0.95,dim=-1,keepdim=True)
    min_v,_ = torch.min(v,dim=-1,keepdim=True)
    v_50 = torch.quantile(v, 0.5,dim=-1,keepdim=True)

    pt_v = torch.quantile(v, pt_num,dim=-1,keepdim=True)

    rate_v = torch.diff(v)
    mean_rate_v = torch.mean(rate_v,dim=-1,keepdim=True)
    pt_rate_v = torch.quantile(rate_v, pt_num,dim=-1,keepdim=True)
    var_rate_v = torch.var(rate_v,dim=-1,keepdim=True)
    min_rate_v,_ = torch.min(rate_v,dim=-1,keepdim=True)

    mid_11 = lat[...,int(end / 4)].unsqueeze(dim=-1)
    mid_22 = lon[...,int(end / 4)].unsqueeze(dim=-1)
    mid_111 = lat[...,int(end / 4) * 3].unsqueeze(dim=-1)
    mid_222 = lon[...,int(end / 4) * 3].unsqueeze(dim=-1)

    unit_complex = torch.exp(1j * theta)
    angle_diff = torch.angle(diff / unit_complex[...,1:])
    angle_diff_max = torch.quantile(angle_diff, 0.95,dim=-1,keepdim=True)
    angle_diff_mean = torch.mean(angle_diff,dim=-1,keepdim=True)

    spectrum = torch.fft.fftshift(torch.fft.fft(points))
    pt_spectrum = torch.quantile(torch.abs(spectrum), pt_num,dim=-1,keepdim=True)
    pt_index = (torch.abs(spectrum - pt_spectrum)).argmin()
    pt_freq = (pt_index - (timestep.shape[0]) / 2) / ((timestep.shape[0]) / 2) * median_sample_rate 

    spectrum = torch.fft.fftshift(torch.fft.fft(v_vec))
    pt_spectrum = torch.quantile(torch.abs(spectrum), pt_num,dim=-1,keepdim=True)
    pt_index = (torch.abs(spectrum - pt_spectrum)).argmin()
    pt_freq_v = (pt_index - (timestep.shape[0]) / 2) / ((timestep.shape[0]) / 2) * median_sample_rate 

    mean_angle = torch.mean(angle,dim=-1,keepdim=True)
    max_angle = torch.quantile(angle, 0.95,dim=-1,keepdim=True)
    var_angle = torch.var(angle,dim=-1,keepdim=True)
    min_angle,_ = torch.min(angle,dim=-1,keepdim=True)
    angle_20 = torch.quantile(angle, 0.2,dim=-1,keepdim=True)
    angle_50 = torch.quantile(angle, 0.5,dim=-1,keepdim=True)
    angle_75 = torch.quantile(angle, 0.75,dim=-1,keepdim=True)

    rate_angle = torch.diff(angle) 
    mean_rate_angle = torch.mean(rate_angle,dim=-1,keepdim=True)
    max_rate_angle,_ = torch.max(rate_angle,dim=-1,keepdim=True)
    var_rate_angle = torch.var(rate_angle,dim=-1,keepdim=True)
    min_rate_angle,_ = torch.min(rate_angle,dim=-1,keepdim=True)

    fft_lat = top5freqs(lat)*median_sample_rate
    fft_lon = top5freqs(lon)*median_sample_rate
    fft_v = top5freqs(v)*median_sample_rate
    fft_angle = top5freqs(angle)*median_sample_rate

    feature_list = [
        dis, pt_distances, start_1, start_2, mid_1, mid_2, end_1, end_2,
        mean_v, max_v, min_v, v_50, mean_rate_v, var_rate_v, min_rate_v,
        pt_v, pt_rate_v, mid_11, mid_22, mid_111, mid_222,
        angle_diff_max, angle_diff_mean, pt_freq, pt_freq_v,
        mean_angle, max_angle, var_angle, min_angle, angle_20, angle_50, angle_75,
        mean_rate_angle, max_rate_angle, var_rate_angle, min_rate_angle,
        fft_lat[...,0].unsqueeze(dim=-1), fft_lat[...,1].unsqueeze(dim=-1), 
        fft_lon[...,0].unsqueeze(dim=-1), fft_lon[...,1].unsqueeze(dim=-1), 
        fft_v[...,0].unsqueeze(dim=-1), fft_v[...,1].unsqueeze(dim=-1), 
        fft_angle[...,0].unsqueeze(dim=-1), fft_angle[...,1].unsqueeze(dim=-1)
    ]
    
    feature = torch.stack(feature_list,dim=-1).squeeze()
    
    return feature.cpu()    
    

def kinetic_feature(sample_list,n_jobs:int = 1):
    if n_jobs == 1: 
        sample_list = [torch.from_numpy(single_sample).to("cpu") for single_sample in sample_list]
        features = map(_kinetic_feature,sample_list)
        features_list = list(features)
        feature = np.stack(features_list)
        
        # 找到无穷大的位置
        inf_indices = np.isinf(feature)

        # 将无穷大的位置置零
        feature[inf_indices] = 0
        
        # 找到NaN的位置
        inf_indices = np.isnan(feature)

        # 将Nan的位置置零
        feature[inf_indices] = 0
        
        return feature
    else:
        sample_list = torch.from_numpy(sample_list).cuda()
        #_kinetic_feature_vmap = torch.vmap(_kinetic_feature,in_dims=0,out_dims=0)
        features=_kinetic_feature_vec(sample_list)
        return features.numpy()
    
if __name__ == "__main__":
    sample = np.random.rand(500,6,2000)
    t1 = time.time()
    a=kinetic_feature(sample,n_jobs=1)
    print(time.time()-t1)
    t1 = time.time()
    a=kinetic_feature(sample,n_jobs=1)
    print(time.time()-t1)
    t1 = time.time()
    a=kinetic_feature(sample,n_jobs=1)
    print(time.time()-t1)
   # print(a)