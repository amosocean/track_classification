import os
import numpy as np
import h5py
import utils.dataset
import torch
from typing import Dict,List,Tuple

from torch.utils.data import Dataset
from torch.utils.data import DataLoader,ConcatDataset

class DatasetReader:
    def __init__(self,dataset_dir:str) -> None:

        self.dataset = utils.dataset.Readcsv(dataset_dir)
        self.dim_num=8
        self.window_len = 128
        self.window_strip = 128
        self.category_sahpe=len(self.dataset)
    
    def get_trajectorys(self,category_index:int)->Tuple:
        """返回无滑窗的变长轨迹list"""
        assert category_index <= self.category_sahpe and category_index>=0 , "category_index out of range"
        trajectory_shape=len(self.dataset[category_index])
        
        n = self.dim_num
        lat_mean = []
        lon_mean = []
        file_name =[]
        res = self.dataset[category_index]
        
        return res,lat_mean,lon_mean,file_name
        
    
class SubDataset(Dataset):
    """针对某一个类别的dataset"""
    def __init__(self,category_index,datareader) -> None:
        super().__init__()    
        self.datareader = datareader 
        self.category_index = category_index
        self.trajectory_num = None
        self.read_data()
        
    def read_data(self):
        self.trajectory,lat,lon,self.file_names=self.datareader.get_trajectorys(self.category_index)
        self.extra_fea = np.array([lat,lon]).squeeze().T
        self.trajectory_num = len(self.trajectory)
        
        
    def __getitem__(self,index)->(np.array,int):
        
        #(trajectory,zone_code)=self.datareader.get_trajectorys(self.category_index,trajectory_index)
        
        trajectory = self.trajectory[index]
        # if self.category_index == 1:
        #     index = 1
        # else:
        #     index = 0
        return trajectory , self.category_index, self.category_index
    def __len__(self):
        #return self.datareader.get_length_trajectory(self.category_index)
        return self.trajectory_num
    
class SubDataset_split(SubDataset):
    def  __init__(self, category_index, datareader) -> None:
        super().__init__(category_index, datareader)
        
    def read_data(self):
        self.trajectory,lat,lon,self.file_names=self.datareader.get_trajectorys(self.category_index)
        self.extra_fea = np.array([lat,lon]).squeeze().T
        self.trajectory_num = len(self.trajectory)
        self.trajectory = self.split_trajectory([50,100,150,200])
        
    def split_trajectory(self, window_size_list):
        result = []
        for traj in self.trajectory:
            traj_length = traj.shape[1]
            split_traj = []
            for window_size in window_size_list:
                for i in range(0, 500, np.int64(np.log2(window_size)*8.87)):
                    if i + window_size <= 500 and i + window_size <= traj_length:
                        split_traj.append(traj[:, i: i + window_size])
                    else:  # edge case where window_size > remaining traj length
                        # Padding with last element when the remaining trajectory is
                        # shorter than the window_size
                        continue
                        padding = np.full((traj.shape[0], i + window_size - traj.shape[1]), 
                                        traj[:, -1].reshape(-1, 1))
                        split_traj.append(np.concatenate((traj[:, i:], padding), axis=1))
                        continue
            result.append(split_traj)
        return result
    
train_reader = DatasetReader("source/datasets")

class SubTrainDataset(SubDataset):
    def __init__(self, category_index) -> None:
        super().__init__(category_index,train_reader)

# class SubTestDataset(SubDataset):
#     def __init__(self, category_index) -> None:
#         super().__init__(category_index,DatasetReader("source/datasets"))

# class SubRaceDataset(SubDataset):
#     def __init__(self, category_index) -> None:
#         super().__init__(category_index,DatasetReader("source/datasets"))

#%%
def print_matrix(label_list,predict_list):
    from sklearn.metrics import confusion_matrix
    conf_mat = confusion_matrix(np.array(label_list), np.array(predict_list))

    print(conf_mat)
    
    from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score

    # 假设 y_true 是真实的标签，y_pred 是预测的标签
    y_true = np.array(label_list)
    y_pred = np.array(predict_list)

    precision = precision_score(y_true, y_pred, average='macro')
    accuraccy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    print('Precision: {}'.format(precision))
    print('Accuraccy: {}'.format(accuraccy))
    print('Recall: {}'.format(recall))
    print('F1 Score: {}'.format(f1))
    return accuraccy,f1

def predict_vote_func(predictions:List[np.array])->np.array:
    # 使用 numpy 的 unique 函数来获取所有唯一的预测值及它们的数量
    unique, counts = np.unique(predictions, return_counts=True)

    # 使用 numpy 的 argmax 函数来获取得票最多的预测值的索引
    max_votes_index = np.argmax(counts)

    # 使用索引返回得票最多的预测值
    return unique[max_votes_index]

def predict_prob_func(predictions:np.array)->np.array:
        # 计算每种class的总概率
    class_probabilities=np.sum(predictions,axis=0)
    # 找出概率最大的class的标签
    most_probable_class_label = np.argmax(class_probabilities)
    return most_probable_class_label

def valid_func(clf,X_list:List,y_list:List)->None:
    
    def get_01_label(label_list:List):
        """从14分类中直接获取2分类"""
        # 二分类处理
        label_01 = np.array([*label_list[:]], dtype=np.int32)
        indices = np.where((label_01 == 5) | (label_01 == 9) | (
            label_01 == 10) | (label_01 == 12))

        if not indices[0].size == 0:
            # 将它们置为类别1:
            label_01[indices] = 1

        # 将剩余的类别都置为0
        other_indices = np.where((label_01 != 1))
        label_01[other_indices] = 0
        rtn = label_01.tolist()
        return rtn
    

    
    pack = zip(X_list,y_list)
    label_list = []
    predict_list = []
    # for X,y in pack:

    #     result = clf.predict(X)
    #     result_prob = clf.predict_proba(X)
    #     label_list.append(y[0])
    #     predict_list.append(predict_prob_func(result_prob))
    dynamic_features = np.array(batch_kinetic(kinetic_feature)(sample_list))
    #dynamic_features = np.concatenate([dynamic_features,extra_feature],axis=-1)
    #catch22_features = np.array(tnf.fit_transform(X_list))
    
    #clf_all = WeightedEnsembleClassifier([clf_0,clf_2])
    
    #catch22_features = pca.fit_transform(catch22_features)
    # all_features = np.concatenate([dynamic_features,catch22_features],axis=-1)
    # predict_list = clf.predict(all_features)
    clf_0,clf_2 = clf
    prob1 = clf_0.predict_proba(dynamic_features)
    #prob2 = clf_2.predict_proba(catch22_features)
    prob = prob1
    
    predict_list = np.argmax(prob,axis=1)
    label_list = y_list
    label_list01 = get_01_label(label_list)
    predict_list01 = get_01_label(predict_list)
    
    acc_multi,f1_multi = print_matrix(label_list,predict_list)
    acc_bio,f1_bio = print_matrix(label_list01,predict_list01)
    print('One stage Score: {}'.format(((f1_bio+f1_multi)/2+(acc_bio+acc_multi)/2)/2))
    
    return predict_list, acc_multi,f1_multi

def pen_calculate(predict01, predict14, label01):
    N = len(label01)
    err1 = 0
    err2 = 0
    label11= [5,9,10,12]
    label00= [0,1,2,3,4,6,7,8,11,13]
    for i in range(N):
        if (predict01[i] != label01[i]):
            if predict01[i] == 0:
                if int(predict14[i]) in label11:
                    err1 += 1
            else:
                if int(predict14[i]) in label00:
                    err1 += 1
        else:
            if predict01[i] == 0:
                if int(predict14[i]) in label11:
                    err2 += 1
            else:
                if int(predict14[i]) in label00:
                    err2 += 1
    
    err1 = err1 / N
    err2 = err2 / N
    Pen = 0.5 * err1 + 0.2 * err2
    return Pen
    
def batch_kinetic(func):
    def wrapper(sample_list):
        from operator import itemgetter 
        #X=np.array(sample_list)
        # X = np.concatenate(sample_list,axis=0)
        #y = np.concatenate(category_index_list,axis=0)
        start_time = time.time()
        tensor_lengths = [t.shape[-1] for t in sample_list]
        tensor_indices = list(range(len(sample_list)))

        # 将上述列表合并为一个列表并按照length排序
        length_index_tensors = sorted(zip(tensor_lengths, tensor_indices, sample_list), key=itemgetter(0))

        # 将同样长度的tensor分别组装为batch
        batches = []
        current_length = length_index_tensors[0][0]
        current_batch_indices = []
        current_batch_tensors = []
        for length, index, tensor in length_index_tensors:
            if length != current_length:
                batches.append((current_length, current_batch_indices, current_batch_tensors))
                current_length = length
                current_batch_indices = []
                current_batch_tensors = []
            current_batch_indices.append(index)
            current_batch_tensors.append(tensor)
        batches.append((current_length, current_batch_indices, current_batch_tensors))

        outputs = []
        for length, indices, batch in batches:
            batch = np.stack(batch)
            batch_output = func(batch,n_jobs=2)
            outputs.extend(zip(indices, batch_output))
        
        # 根据index对计算结果进行排序
        outputs.sort(key=itemgetter(0))
        dynamic_features = [output for index, output in outputs]
        dynamic_features = np.stack(dynamic_features)
        
        end_time = time.time()

        # Calculate elapsed time
        elapsed_time = end_time - start_time
        print(f"The code took {elapsed_time} seconds to run.")
        
        return dynamic_features
    return wrapper

def get_tracksets(dataset):
        mydata=DataLoader(dataset,batch_size=1,shuffle=False)
        sample_list = []
        category_index_list = []
        category_index_01_list = []
        # filename_list = []
        # extra_feature_list = []
        for data in mydata:
            # filename_list.append(data[3])
            # extra_feature_list.append(data[4])
            category_index = int(data[1].numpy())
            #category_index_vector = np.tile(category_index, sample.shape[0])
            category_index_list.append(category_index)
            
            category_index_01 = int(data[2].numpy())
            #category_index_vector = np.tile(category_index, sample.shape[0])
            category_index_01_list.append(category_index_01)
            #category_index_list.append(category_index_vector)
            
            sample=data[0].squeeze(dim=0).numpy()
            # t= np.isnan(sample)
            # assert not np.any(t) , "Has Nan!"
            sample_list.append(sample)
            
        _0_index = np.where(np.array(category_index_01_list) == 0)
        _1_index =  np.where(np.array(category_index_01_list) == 1)  
        return sample_list,category_index_list,category_index_01_list,_0_index,_1_index
    
def get_tracksets_split(dataset):
    mydata=DataLoader(dataset,batch_size=1,shuffle=False)
    sample_list = []
    category_index_list = []
    category_index_01_list = []
    # filename_list = []
    # extra_feature_list = []
    for data in mydata:
        # filename_list.append(data[3])
        # extra_feature_list.append(data[4])
        category_index = int(data[1].numpy())
        category_index = [category_index]*len(data[0])
        #category_index_vector = np.tile(category_index, sample.shape[0])
        category_index_list.extend(category_index)
        
        category_index_01 = int(data[2].numpy())
        category_index_01 = [category_index_01]*len(data[0])
        #category_index_vector = np.tile(category_index, sample.shape[0])
        category_index_01_list.extend(category_index_01)
        #category_index_list.append(category_index_vector)
        
        window_list=data[0]
        window_list = [window.squeeze(dim=0).numpy() for window in window_list]
        
        # t= np.isnan(sample)
        # assert not np.any(t) , "Has Nan!"
        sample_list.extend(window_list)
        
    _0_index = np.where(np.array(category_index_01_list) == 0)
    _1_index =  np.where(np.array(category_index_01_list) == 1)  
    return sample_list,category_index_list,category_index_01_list,_0_index,_1_index

def get_tracksets_split_test(dataset):
    mydata=DataLoader(dataset,batch_size=1,shuffle=False)
    sample_list = []
    category_index_list = []
    category_index_01_list = []
    # filename_list = []
    # extra_feature_list = []
    for data in mydata:
        # for split in data[0]:
        #     split.squeeze(dim=0).shape[0].numpy()
        #     t = 1
        data[0] = [split for split in data[0] if split.shape[-1]==50]
        sample = random.choice(data[0]).squeeze(dim=0).numpy()
        # k = min(len(data[0]),3)
        # sample = random.sample(data[0],k).squeeze(dim=0).numpy()
        # filename_list.append(data[3])
        # extra_feature_list.append(data[4])
        category_index = int(data[1].numpy())
        #category_index_vector = np.tile(category_index, sample.shape[0])
        category_index_list.append(category_index)
        
        category_index_01 = int(data[2].numpy())
        #category_index_vector = np.tile(category_index, sample.shape[0])
        category_index_01_list.append(category_index_01)
        #category_index_list.append(category_index_vector)
        
        
        # t= np.isnan(sample)
        # assert not np.any(t) , "Has Nan!"
        sample_list.append(sample)
        
    _0_index = np.where(np.array(category_index_01_list) == 0)
    _1_index =  np.where(np.array(category_index_01_list) == 1)  
    return sample_list,category_index_list,category_index_01_list,_0_index,_1_index

#%%

if __name__ == "__main__":
    import time
    from torch.utils.data import random_split
    from torch.utils.data import Subset
    from sklearn.ensemble import RandomForestClassifier
    from utils.fea_gpu import kinetic_feature
    from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
    import random

    dataset_list = [SubTrainDataset(i) for i in iter([0,1,5])]
    #dataset_list.extend([SubTrainDataset(i) for i in [5,5,5,9,9,9,12,12,12]])
    datatset_all=ConcatDataset(datasets=dataset_list)
    
    # 假设 dataset 是你的大数据集
    # 下面的 num1 和 num2 是你想分割成的两个数据集的大小
    num1 = int(len(datatset_all) * 0.8)  # 比如你想让第一个数据集占80%的数据
    num2 = len(datatset_all) - num1  # 第二个数据集的大小等于总大小减去第一个的大小

    train_dataset, valid_dataset = random_split(datatset_all, [num1, num2])
    

#%% 
# """调试用，缩小数据集"""
#     def random_subset(dataset, fraction):
#         length = len(dataset)
#         indices = np.random.choice(np.arange(length), size=int(fraction * length), replace=False)
#         return Subset(dataset, indices)

#     # 假设 dataset 是你的原始数据集
#     dataset1 = random_subset(dataset1, 0.1)


    
    
    #sample_list,category_index_list,category_index_01_list,_0_index,_1_index = get_tracksets_split(train_dataset)
    sample_list,category_index_list,category_index_01_list,_0_index,_1_index = get_tracksets(train_dataset)
    # extra_feature = np.stack(extra_feature_list).squeeze()
    print(len(sample_list))

# %%
    param_grid = {
    'n_estimators': [50,80,150],
    'max_features': [*list(range(4,6,1)),*list(range(14,17,2)),None],
    'max_depth' : [*list(range(12,15,1)),None],
    'criterion' :['gini']
}


    clf_01 = RandomForestClassifier(n_estimators=120,n_jobs=-1,oob_score=True)
    # clf_01 = GridSearchCV(RandomForestClassifier(n_jobs=-1),param_grid=param_grid,cv=10,n_jobs=-1) 

    y = np.array(category_index_list)
    X=sample_list
    dynamic_features =kinetic_feature(sample_list,n_jobs=1)

    clf_01.fit(dynamic_features,np.array(category_index_01_list))
    print(clf_01.oob_score_)

 #%% validation   
    sample_list,category_index_list,category_index_01_list,_0_index,_1_index = get_tracksets(valid_dataset)
    print(len(sample_list))
    
    #%% 01分类
    dynamic_features = np.array(kinetic_feature(sample_list))
    #dynamic_features = np.concatenate([dynamic_features,extra_feature],axis=-1)
    # prob = clf_01.predict_proba(dynamic_features)
    # predict_list_01 = np.argmax(prob,axis=1)
    predict_list_01 = clf_01.predict(dynamic_features)
    f1_bio,acc_bio = print_matrix(predict_list_01,category_index_01_list)
    exit()
    #%% 比赛部分
    racedataset = SubRaceDataset(0)
    sample_list,dummy_category_index_list,dummy_category_index_01_list,dummy_0_index,dummy_1_index,file_name_list,extra_feature_list = get_tracksets(racedataset)
    print(len(sample_list))
     #%% 01分类
    dynamic_features = np.array(batch_kinetic(kinetic_feature)(sample_list))
    nan_index = np.where(np.isnan(dynamic_features))
    print(nan_index)
    #dynamic_features = np.concatenate([dynamic_features,extra_feature],axis=-1)
    direct_predict_list = clf0_14.predict(dynamic_features)
    direct_predict_list=np.int64(direct_predict_list)
    counts = np.bincount(direct_predict_list)

    print(counts)
    prob = clf_01.predict_proba(dynamic_features)
    predict_list_01 = np.argmax(prob,axis=1)
    #f1_bio,acc_bio = print_matrix(predict_list_01,dummy_category_index_01_list)
    prdict_0_index = np.where(predict_list_01 == 0)
    prdict_1_index = np.where(predict_list_01 == 1)
    #%% 
    # 0细分类
    predict_0_list = clf_0.predict(dynamic_features[prdict_0_index])
    
    # 1细分类
    predict_1_list = clf_1.predict(dynamic_features[prdict_1_index])
    
    predict_combine = np.zeros(len(sample_list))
    predict_combine[prdict_0_index] = predict_0_list
    predict_combine[prdict_1_index] = predict_1_list
    predict_combine = np.int64(predict_combine)

    # pen = pen_calculate(predict_list_01,predict_combine,category_index_01_list)
    # print('Two stage F1 Score: {}'.format(((f1_bio+f1_multi)/2+(acc_bio+acc_multi)/2)/2-pen))
    # print(pen)
    
    file_name_array = np.stack(file_name_list).squeeze()
    data = np.column_stack((file_name_array,predict_combine, predict_list_01))
    
    
    counts = np.bincount(predict_combine)

    print(counts)
    
    np.savetxt('hq-02-XXX.txt', data, fmt='%s')
    
    data = np.column_stack((file_name_array,direct_predict_list, predict_list_01))
    
    
    counts = np.bincount(predict_combine)

    print(counts)
    
    np.savetxt('hq-02-XXX-direct.txt', data, fmt='%s')
