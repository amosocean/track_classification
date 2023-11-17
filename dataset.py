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
        self.dim_num=4
        self.category_sahpe=self.dataset['sample_list'].shape
        
    def get_trajectorys(self,category_index:int,trajectory_index:int)->Tuple[np.array]:
        #"""返回一个[5,timestep_num]的np数组，5包括time x y v angle , 以及一个区域编码"""
        """返回一个[5,timestep_num]的np数组，5包括x y v angle"""
        assert category_index <= self.category_sahpe[-1] and category_index>=0 , "category_index out of range"
        trajectory_shape=self.dataset[self.dataset['sample_list'][0,category_index]].shape
        assert trajectory_index <= trajectory_shape[-1] and trajectory_index>=0 , "trajectory_index out of range"
        
        #n=self.dim_num-1
        n = self.dim_num
        trajectory_ref=self.dataset[self.dataset[self.dataset['sample_list'][0,category_index]][0,trajectory_index]][:,0]
        data_list=[self.dataset[trajectory_ref[_]] for _ in range(n)]
        #code_ref = self.dataset[self.dataset[self.dataset['sample_list'][0,category_index]][0,trajectory_index]][n,0]
        rtn= np.array(data_list,dtype=np.float32).squeeze()
        #return rtn, np.array(self.dataset[code_ref],dtype=np.int32) 
        return rtn            

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
    
class SubDataset(Dataset):
    """针对某一个类别的dataset"""
    def __init__(self,category_index) -> None:
        super().__init__()    
        self.datareader = DatasetReader()
        self.category_index = category_index

    def __getitem__(self,index)->(np.array,int):
        trajectory_index=index
        #(trajectory,zone_code)=self.datareader.get_trajectorys(self.category_index,trajectory_index)
        trajectory=self.datareader.get_trajectorys(self.category_index,trajectory_index)
        
        #return (trajectory,zone_code) , self.category_index
        return trajectory , self.category_index

    def __len__(self):
        return self.datareader.get_length_trajectory(self.category_index)
    
      

if __name__ == "__main__":
    # dataset=Dataset()
    # print(dataset.get_trajectorys(0,0))
    # x=list(dataset.get_category(0))

    from aeon.utils.validation.collection import convert_collection
    import aeon.datasets
    from aeon.datasets import write_to_tsfile
    
    dataset_list = [SubDataset(i) for i in range(14)]
    dataset1=ConcatDataset(datasets=dataset_list)
    
    # 将数据集分割成训练集和验证集
    train_size = int(0.8 * len(dataset1))
    valid_size = len(dataset1) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset1, [train_size, valid_size])

    mydata=DataLoader(train_dataset,batch_size=1,shuffle=False)
    sample_list = []
    category_index_list = []
    for data in mydata:
        sample=data[0].squeeze().numpy()
        # t= np.isnan(sample)
        # assert not np.any(t) , "Has Nan!"
        sample_list.append(sample)
        category_index_list.append(int(data[1].numpy()))

    print(len(sample_list))
    aeon.datasets.write_to_tsfile(X=sample_list,path="./dataset",y=category_index_list,problem_name="haitun_TRAIN")
    #convert_collection(t,"df-list")
    
    mydata=DataLoader(valid_dataset,batch_size=1,shuffle=False)
    sample_list = []
    category_index_list = []
    for data in mydata:
        sample=data[0].squeeze().numpy()
        t=np.isnan(sample)
        assert not np.any(t) , "Has Nan!"
        sample_list.append(sample)
        category_index_list.append(int(data[1].numpy()))

    print(len(sample_list))
    aeon.datasets.write_to_tsfile(X=sample_list,path="./dataset",y=category_index_list,problem_name="haitun_TEST")