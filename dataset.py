import os
import numpy as np
import h5py
import torch
from typing import Dict,List,Tuple
dir_path="/home/amos/haitun/pycode/"#os.path.dirname(os.path.abspath())
from torch.utils.data import Dataset
from torch.utils.data import DataLoader,ConcatDataset
class DatasetReader:

    def __init__(self) -> None:
        # Open the MATLAB v7.3 file using h5py
        self.dataset = h5py.File(os.path.join(dir_path,'source/matlab/savedData.mat'), 'r')
        self.dim_num=6
        self.category_sahpe=self.dataset['sample_list'].shape
        
    def get_trajectory(self,category_index:int,trajectory_index:int)->Tuple[np.array]:
        """返回一个[5,timestep_num]的np数组，5包括time x y v angle , 以及一个区域编码"""
        
        assert category_index <= self.category_sahpe[-1] and category_index>=0 , "category_index out of range"
        trajectory_shape=self.dataset[self.dataset['sample_list'][0,category_index]].shape
        assert trajectory_index <= trajectory_shape[-1] and trajectory_index>=0 , "trajectory_index out of range"
        
        n=self.dim_num-1
        trajectory_ref=self.dataset[self.dataset[self.dataset['sample_list'][0,category_index]][0,trajectory_index]][:,0]
        data_list=[self.dataset[trajectory_ref[_]] for _ in range(n)]
        code_ref = self.dataset[self.dataset[self.dataset['sample_list'][0,category_index]][0,trajectory_index]][n,0]
        rtn= np.array(data_list,dtype=np.float32).squeeze()
        return rtn, np.array(self.dataset[code_ref],dtype=np.int32)
                    

    def get_category(self,category_index:int)->Tuple[(np.array,np.array)]:
        "返回长度为轨迹数量的元组，每个元素是也是元组，第一个元素为轨迹array，第二个为区域编码"
        
        assert category_index <= self.category_sahpe[-1] and category_index>=0 , "category_index out of range"
        
        n=self.dim_num-1
        trajectory_refs=self.dataset[self.dataset['sample_list'][0,category_index]][0,:]
        # for trajectory_ref in trajectory_refs:
        #     data_list=np.array([self.dataset[trajectory_ref[_]] for _ in range(n)]).squeeze()
            
        def get_array(trajectory_ref):
            return np.array([self.dataset[self.dataset[trajectory_ref][_,0]] for _ in range(n)],dtype=np.float32).squeeze(), np.array(self.dataset[self.dataset[trajectory_ref][n,0]],dtype=np.int32)
        
        return map(get_array,trajectory_refs)
    
    def get_length_category(self)->int:
        """返回类型数量"""
       
        return  self.category_sahpe[-1]
    
    def get_length_trajectory(self,category_index:int)->int:
        """返回某类型的轨迹数量"""
        
        assert category_index <= self.category_sahpe[-1] and category_index>=0 , "category_index out of range"
        trajectory_shape=self.dataset[self.dataset['sample_list'][0,category_index]].shape
        return trajectory_shape[-1]
    
class SubDataset(Dataset):
    """针对某一个类别的dataset"""
    def __init__(self,category_index) -> None:
        super().__init__()    
        self.datareader = DatasetReader()
        self.category_index = category_index

    def __getitem__(self,index):
        trajectory_index=index
        (trajectory,zone_code)=self.datareader.get_trajectory(self.category_index,trajectory_index)
        
        return (trajectory,zone_code) , self.category_index

    def __len__(self):
        return self.datareader.get_length_trajectory(self.category_index)
    
      

if __name__ == "__main__":
    # dataset=Dataset()
    # print(dataset.get_trajectory(0,0))
    # x=list(dataset.get_category(0))
    dataset_list = [SubDataset(i) for i in range(14)]
    dataset1=ConcatDataset(datasets=dataset_list)
    mydata=DataLoader(dataset1,batch_size=1)
    t=tuple(mydata)
    print(t[0])