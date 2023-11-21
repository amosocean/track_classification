import os
import numpy as np
import h5py
import torch
from typing import Dict,List,Tuple
dir_path="/home/amos/haitun/pycode/source/matlab/"
from torch.utils.data import Dataset
from torch.utils.data import DataLoader,ConcatDataset
from aeon.transformations.collection.pad import PaddingTransformer
class DatasetReader:

    def __init__(self,matfile_name:str) -> None:
        # Open the MATLAB v7.3 file using h5py
        self.dataset = h5py.File(os.path.join(dir_path,matfile_name), 'r')
        self.dim_num=5
        self.window_len = 128
        self.window_strip = 128
        self.category_sahpe=self.dataset['sample_list'].shape
        
    def get_trajectorys(self,category_index:int)->Tuple[np.array]:
        #"""返回一个[5,timestep_num]的np数组，5包括time x y v angle , 以及一个区域编码"""
        """返回一个[5,timestep_num]的np数组，5包括x y v angle"""
        assert category_index <= self.category_sahpe[-1] and category_index>=0 , "category_index out of range"
        trajectory_shape=self.dataset[self.dataset['sample_list'][0,category_index]].shape
        
        n = self.dim_num
        res = []
        for index in range(trajectory_shape[-1]):

            trajectory_ref=self.dataset[self.dataset[self.dataset['sample_list'][0,category_index]][0,index]][:,0]
            data_list=[self.dataset[trajectory_ref[_]] for _ in range(n)]
            #code_ref = self.dataset[self.dataset[self.dataset['sample_list'][0,category_index]][0,trajectory_index]][n,0]
            padder=PaddingTransformer(self.window_len,fill_value=0)
            rtn= np.array(data_list,dtype=np.float32).squeeze()
            rtn = np.transpose(rtn)
            if len(rtn)<self.window_len:
                rtn = rtn.T
                res.extend([rtn,])
            else:
                rtn = np.transpose(self.sliding_window(rtn, self.window_len, self.window_strip),[0,2,1])
                temp =[*rtn]
                res.extend(temp)

        res = padder.fit_transform(res)
        return res            

    def get_trajectorys(self,category_index:int)->Tuple:
        """返回一个元组，包含含有多个单一完整轨迹的SubDataset，对于某个类型"""
        assert category_index <= self.category_sahpe[-1] and category_index>=0 , "category_index out of range"
        trajectory_shape=self.dataset[self.dataset['sample_list'][0,category_index]].shape
        
        n = self.dim_num
        res = []
        for index in range(trajectory_shape[-1]):

            trajectory_ref=self.dataset[self.dataset[self.dataset['sample_list'][0,category_index]][0,index]][:,0]
            data_list=[self.dataset[trajectory_ref[_]] for _ in range(n)]
            #code_ref = self.dataset[self.dataset[self.dataset['sample_list'][0,category_index]][0,trajectory_index]][n,0]
            padder=PaddingTransformer(self.window_len,fill_value=0)
            rtn= np.array(data_list,dtype=np.float32).squeeze()
            rtn = np.transpose(rtn)
            if len(rtn)<self.window_len:
                rtn = np.transpose(rtn[np.newaxis,:,:],[0,2,1])
                rtn = padder.fit_transform(rtn)
                res.append(rtn)
            else:
                rtn = np.transpose(self.sliding_window(rtn, self.window_len, self.window_strip),[0,2,1])
                temp =rtn
                res.append(rtn)
        return res
    
    def get_trajectorys(self,category_index:int)->Tuple:
        """返回无滑窗的变长轨迹list"""
        assert category_index <= self.category_sahpe[-1] and category_index>=0 , "category_index out of range"
        trajectory_shape=self.dataset[self.dataset['sample_list'][0,category_index]].shape
        
        n = self.dim_num
        res = []
        for index in range(trajectory_shape[-1]):

            trajectory_ref=self.dataset[self.dataset[self.dataset['sample_list'][0,category_index]][0,index]][:,0]
            data_list=[self.dataset[trajectory_ref[_]] for _ in range(n)]
            #code_ref = self.dataset[self.dataset[self.dataset['sample_list'][0,category_index]][0,trajectory_index]][n,0]
            padder=PaddingTransformer(self.window_len,fill_value=0)
            rtn= np.array(data_list,dtype=np.float32).squeeze()
            rtn = np.transpose(rtn)
            rtn = np.transpose(rtn,[1,0])
            res.append(rtn)
        return res              
        
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
    def __init__(self,category_index,datareader) -> None:
        super().__init__()    
        self.datareader = datareader 
        self.category_index = category_index
        self.trajectory_num = None
        self.read_data()
        
    def read_data(self):
        self.trajectory=self.datareader.get_trajectorys(self.category_index)
        self.trajectory_num = len(self.trajectory)
        
        
        

    def __getitem__(self,index)->(np.array,int):
        
        #(trajectory,zone_code)=self.datareader.get_trajectorys(self.category_index,trajectory_index)
        
        trajectory = self.trajectory[index]
        #return (trajectory,zone_code) , self.category_index
        #return trajectory , self.
        #return trajectory , self.category_index
        # if self.category_index == 5 or self.category_index == 9 or self.category_index == 10 or self.category_index == 12:
        #     index = 1
        # else:
        #     index = 0
        # return trajectory , index
        return trajectory , self.category_index
    def __len__(self):
        #return self.datareader.get_length_trajectory(self.category_index)
        return self.trajectory_num
    
class Trajectory_Dataset(SubDataset):
    
    def read_data(self):
        self.trajectory=self.datareader.get_trajectorys(self.category_index)
        self.trajectory_num = len(self.trajectory)
    
class SubTrainDataset(SubDataset):
    def __init__(self, category_index) -> None:
        super().__init__(category_index,DatasetReader(matfile_name="train.mat"))

class SubTestDataset(SubDataset):
    def __init__(self, category_index) -> None:
        super().__init__(category_index,DatasetReader(matfile_name="test.mat"))
        
if __name__ == "__main__":
    import aeon.datasets
    from torch.utils.data import Subset
    from aeon.datasets import write_to_tsfile
    from aeon.classification.feature_based import Catch22Classifier,TSFreshClassifier
    from aeon.classification.hybrid import HIVECOTEV2
    from aeon.classification.shapelet_based import ShapeletTransformClassifier
    from aeon.classification.distance_based import ElasticEnsemble
    from sklearn.ensemble import RandomForestClassifier
    # dataset=Dataset()
    # print(dataset.get_trajectorys(0,0))
    # x=list(dataset.get_category(0))
    dataset_list = [SubTrainDataset(i) for i in range(14)]
    #dataset_list.extend([SubTrainDataset(i) for i in [5,5,5,9,9,9,12,12,12]])
    train_dataset=ConcatDataset(datasets=dataset_list)
    
    dataset_list = [SubTestDataset(i) for i in range(14)]
    valid_dataset=ConcatDataset(datasets=dataset_list)
    # mydata=DataLoader(dataset1,batch_size=1,shuffle=True)
    # for index, (x, y) in enumerate(mydata):
    #     x=x
    #     y=y
    #     print(index)

#%% 
# """调试用，缩小数据集"""
#     def random_subset(dataset, fraction):
#         length = len(dataset)
#         indices = np.random.choice(np.arange(length), size=int(fraction * length), replace=False)
#         return Subset(dataset, indices)

#     # 假设 dataset 是你的原始数据集
#     dataset1 = random_subset(dataset1, 0.1)
#%%
    # t=tuple(mydata)
    # print(t[0])
    # print(len(t))
    
     # 将数据集分割成训练集和验证集
    # train_size = int(0.8 * len(dataset1))
    # valid_size = len(dataset1) - train_size
    # train_dataset, valid_dataset = torch.utils.data.random_split(dataset1, [train_size, valid_size])

    mydata=DataLoader(train_dataset,batch_size=1,shuffle=False)
    sample_list = []
    category_index_list = []
    for data in mydata:
        sample=data[0].squeeze(dim=0).numpy()
        # t= np.isnan(sample)
        # assert not np.any(t) , "Has Nan!"
        sample_list.append(sample)
        category_index = int(data[1].numpy())
        category_index_vector = np.tile(category_index, sample.shape[0])
        category_index_list.append(category_index)
        #category_index_list.append(category_index_vector)

    print(len(sample_list))
    #aeon.datasets.write_to_tsfile(X=sample_list,path="./dataset",y=category_index_list,problem_name="haitun_TRAIN")
    #convert_collection(t,"df-list")
    clf = Catch22Classifier(
    estimator=RandomForestClassifier(n_estimators=5),
    outlier_norm=True,
    random_state=0,
    n_jobs=16,
)
    #clf = Catch22Classifier(estimator=RandomForestClassifier(n_estimators=5))
#     clf = ElasticEnsemble(
#     proportion_of_param_options=0.1,
#     proportion_train_for_test=0.1,
#     distance_measures = ["dtw","ddtw"],
#     majority_vote=True,
# )
    #X=np.array(sample_list)
    # X = np.concatenate(sample_list,axis=0)
    X=sample_list
    #y = np.concatenate(category_index_list,axis=0)
    y = np.array(category_index_list)
    clf.fit(X, y)   
    
    mydata=DataLoader(valid_dataset,batch_size=1,shuffle=False)
    sample_list = []
    category_index_list = []
    for data in mydata:
        sample=data[0].squeeze(dim=0).numpy()
        # t= np.isnan(sample)
        # assert not np.any(t) , "Has Nan!"
        sample_list.append(sample)
        category_index = int(data[1].numpy())
        category_index_vector = np.tile(category_index, sample.shape[0])
        category_index_list.append(category_index)
        #category_index_list.append(category_index_vector)

    print(len(sample_list))
#     #aeon.datasets.write_to_tsfile(X=sample_list,path="./dataset",y=category_index_list,problem_name="haitun_TEST")

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
        
        pack = zip(X_list,y_list)
        label_list = []
        predict_list = []
        # for X,y in pack:

        #     result = clf.predict(X)
        #     result_prob = clf.predict_proba(X)
        #     label_list.append(y[0])
        #     predict_list.append(predict_prob_func(result_prob))
        
        predict_list = clf.predict(X_list)
        label_list = y_list
            
    
    
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

    valid_func(clf=clf,X_list=sample_list,y_list=category_index_list)
