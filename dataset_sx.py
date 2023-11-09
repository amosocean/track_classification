import os
import numpy as np
import h5py
import torch
from typing import Dict,List,Tuple
dir_path="/home/amos/haitun/pycode/"
from torch.utils.data import Dataset
from torch.utils.data import DataLoader,ConcatDataset
class DatasetReader:

    def __init__(self) -> None:
        # Open the MATLAB v7.3 file using h5py
        self.dataset = h5py.File(os.path.join(dir_path,'source/matlab/savedData.mat'), 'r')
        self.dim_num=4
        self.category_sahpe=self.dataset['sample_list'].shape
        
    def get_trajectory(self,category_index:int)->Tuple[np.array]:
        #"""返回一个[5,timestep_num]的np数组，5包括time x y v angle , 以及一个区域编码"""
        """返回一个[5,timestep_num]的np数组，5包括x y v angle"""
        assert category_index <= self.category_sahpe[-1] and category_index>=0 , "category_index out of range"
        trajectory_shape=self.dataset[self.dataset['sample_list'][0,category_index]].shape
        
        
        res = []
        for index in range(trajectory_shape[-1]):

        #n=self.dim_num-1
            n = self.dim_num
            trajectory_ref=self.dataset[self.dataset[self.dataset['sample_list'][0,category_index]][0,index]][:,0]
            data_list=[self.dataset[trajectory_ref[_]] for _ in range(n)]
            #code_ref = self.dataset[self.dataset[self.dataset['sample_list'][0,category_index]][0,trajectory_index]][n,0]
            rtn= np.array(data_list,dtype=np.float32).squeeze()
            rtn = np.transpose(rtn)
            if len(rtn)<50:
                continue
            res.append(self.sliding_window(rtn, 50, 50))

        rres = np.concatenate(res,axis=0)
        # np.random.seed(10)
        # np.random.shuffle(rres)
        # rres_train = rres[0:int(len(rres)*0.7),:,:]
        # rres_test = rres[int(len(rres)*0.7):,:,:]
        #return rtn, np.array(self.dataset[code_ref],dtype=np.int32) 
        return rres            

    def get_category(self,category_index:int)->Tuple[(np.array,np.array)]:
        #"返回长度为轨迹数量的元组，每个元素是也是元组，第一个元素为轨迹array，第二个为区域编码"
        "返回长度为轨迹数量的元组，每个元素是也是元组，第一个元素为轨迹array"
        assert category_index <= self.category_sahpe[-1] and category_index>=0 , "category_index out of range"
        
        #n=self.dim_num-1
        n = self.dim_num
        trajectory_refs=self.dataset[self.dataset['sample_list'][0,category_index]][0,:]
        # for trajectory_ref in trajectory_refs:
        #     data_list=np.array([self.dataset[trajectory_ref[_]] for _ in range(n)]).squeeze()
            
        def get_array(trajectory_ref):
            #return np.array([self.dataset[self.dataset[trajectory_ref][_,0]] for _ in range(n)],dtype=np.float32).squeeze(), np.array(self.dataset[self.dataset[trajectory_ref][n,0]],dtype=np.int32)
            return np.array([self.dataset[self.dataset[trajectory_ref][_,0]] for _ in range(n)],dtype=np.float32).squeeze()
        
        return map(get_array,trajectory_refs)
    
    def get_length_category(self)->int:
        """返回类型数量"""
       
        return  self.category_sahpe[-1]
    
    def get_length_trajectory(self,category_index:int)->int:
        """返回某类型的轨迹数量"""
        
        assert category_index <= self.category_sahpe[-1] and category_index>=0 , "category_index out of range"
        trajectory_shape=self.dataset[self.dataset['sample_list'][0,category_index]].shape
        return trajectory_shape[-1]
    
    def sliding_window(self,matrix, window_len, n):
        new_shape = (1 + (matrix.shape[0] - window_len) // n, window_len, matrix.shape[1])
        new_matrix = np.zeros(new_shape)
        for i in range(new_shape[0]):
            new_matrix[i] = matrix[i * n : i * n + window_len]
        if (new_shape[0] - 1) * n + window_len < matrix.shape[0]:
            new_matrix = np.concatenate((new_matrix, matrix[-window_len:, :][np.newaxis, :, :]), axis=0)
        return new_matrix
class SubDataset(Dataset):
    """针对某一个类别的dataset"""
    def __init__(self,category_index) -> None:
        super().__init__()    
        self.datareader = DatasetReader()
        self.category_index = category_index
        self.trajectory_num = None
        self.read_data()
        
    def read_data(self):
        self.trajectory=self.datareader.get_trajectory(self.category_index)
        self.trajectory_num = self.trajectory.shape[0]
        
        
        

    def __getitem__(self,index)->(np.array,int):
        
        #(trajectory,zone_code)=self.datareader.get_trajectory(self.category_index,trajectory_index)
        
        trajectory = self.trajectory[index]
        #return (trajectory,zone_code) , self.category_index
        return trajectory , self.category_index

    def __len__(self):
        #return self.datareader.get_length_trajectory(self.category_index)
        return self.trajectory_num
    

if __name__ == "__main__":
    # dataset=Dataset()
    # print(dataset.get_trajectory(0,0))
    # x=list(dataset.get_category(0))
    dataset_list = [SubDataset(i) for i in range(14)]
    dataset1=ConcatDataset(datasets=dataset_list)
    mydata=DataLoader(dataset1,batch_size=1,shuffle=True)
    for index, (x, y) in enumerate(mydata):
        x=x
        y=y
        print(index)



    t=tuple(mydata)
    print(t[0])
    print(len(t))