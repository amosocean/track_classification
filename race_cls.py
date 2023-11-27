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
        self.dim_num=8
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
        lat_mean = []
        lon_mean = []
        file_name =[]
        for index in range(trajectory_shape[-1]):

            trajectory_ref=self.dataset[self.dataset[self.dataset['sample_list'][0,category_index]][0,index]][:,0]
            data_list=[self.dataset[trajectory_ref[_]] for _ in range(n)]
            #code_ref = self.dataset[self.dataset[self.dataset['sample_list'][0,category_index]][0,trajectory_index]][n,0]
            rtn= np.array(data_list[0:5],dtype=np.float32).squeeze()
            rtn = np.transpose(rtn)
            rtn = np.transpose(rtn,[1,0])
            res.append(rtn)
            lat_mean.append(np.array(data_list[5]))
            lon_mean.append(np.array(data_list[6]))
            file_name.append(np.array(data_list[7]))
        return res,lat_mean,lon_mean,file_name     
        #return res         
        
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
        self.trajectory,lat,lon,self.file_names=self.datareader.get_trajectorys(self.category_index)
        self.extra_fea = np.array([lat,lon]).squeeze().T
        self.trajectory_num = len(self.trajectory)
        
        
        

    def __getitem__(self,index)->(np.array,int):
        
        #(trajectory,zone_code)=self.datareader.get_trajectorys(self.category_index,trajectory_index)
        
        trajectory = self.trajectory[index]
        file_name = self.file_names[index]
        zones = self.extra_fea[index,:]
        #return (trajectory,zone_code) , self.category_index
        #return trajectory , self.
        #return trajectory , self.category_index
        if self.category_index == 5 or self.category_index == 9 or self.category_index == 10 or self.category_index == 12:
            index = 1
        else:
            index = 0
        return trajectory , self.category_index, index, file_name, zones
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

class SubRaceDataset(SubDataset):
    def __init__(self, category_index) -> None:
        super().__init__(category_index,DatasetReader(matfile_name="race.mat"))
        
if __name__ == "__main__":
    import aeon.datasets
    from torch.utils.data import Subset
    from aeon.datasets import write_to_tsfile
    from aeon.classification.feature_based import Catch22Classifier,TSFreshClassifier,FreshPRINCEClassifier,MatrixProfileClassifier,SignatureClassifier,SummaryClassifier
    from aeon.classification.interval_based import CanonicalIntervalForestClassifier
    from aeon.classification.hybrid import HIVECOTEV2
    from aeon.classification.shapelet_based import ShapeletTransformClassifier
    from aeon.classification.distance_based import ElasticEnsemble
    from aeon.transformations.collection import Catch22
    from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.decomposition import PCA
    from aeon.classification.compose import WeightedEnsembleClassifier
    from aeon.transformations.collection.tsfresh import *
    from utils.fea import kinetic_feature
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

# #%% 
#     """调试用，缩小数据集"""
#     def random_subset(dataset, fraction):
#         length = len(dataset)
#         indices = np.random.choice(np.arange(length), size=int(fraction * length), replace=False)
#         return Subset(dataset, indices)

#     # 假设 dataset 是你的原始数据集
#     train_dataset = random_subset(train_dataset, 0.1)
#     valid_dataset = random_subset(valid_dataset, 0.2)


    def get_tracksets(dataset):
        mydata=DataLoader(dataset,batch_size=1,shuffle=True)
        sample_list = []
        category_index_list = []
        category_index_01_list = []
        filename_list = []
        extra_feature_list = []
        for data in mydata:
            filename_list.append(data[3])
            extra_feature_list.append(data[4])
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
        return sample_list,category_index_list,category_index_01_list,_0_index,_1_index,filename_list,extra_feature_list
    
    sample_list,category_index_list,category_index_01_list,_0_index,_1_index,file_name_list,extra_feature_list = get_tracksets(train_dataset)
    extra_feature = np.stack(extra_feature_list).squeeze()
    print(len(sample_list))
    #aeon.datasets.write_to_tsfile(X=sample_list,path="./dataset",y=category_index_list,problem_name="haitun_TRAIN")
    #convert_collection(t,"df-list")
#     clf = Catch22Classifier(
#     estimator= RandomForestClassifier(max_depth=50, n_estimators=10, max_features=2, random_state=42),
#     outlier_norm=True,
#     random_state=0,
#     n_jobs=16,
# )
# # %%
#     sample_list = sample_list[0:3]
#     category_index_list = category_index_list[0:3]
# %%
    tnf = Catch22(outlier_norm=True,catch24=True,replace_nans=True,n_jobs=-1,parallel_backend="loky")
    clf_01 = RandomForestClassifier(n_estimators=500,n_jobs=-1)
    clf0_14 = RandomForestClassifier(n_estimators=500,n_jobs=-1)
    clf_0 = RandomForestClassifier(n_estimators=500,n_jobs=-1)
    clf_1 = RandomForestClassifier(n_estimators=500,n_jobs=-1)
    clf_2 = RandomForestClassifier(n_estimators=50,n_jobs=-1)
    pca = PCA(n_components=1)
    #clf = Catch22Classifier(estimator=RandomForestClassifier(n_estimators=5))
    clf = TSFreshFeatureExtractor(default_fc_parameters="efficient",disable_progressbar=False,chunksize=10)

    #X=np.array(sample_list)
    # X = np.concatenate(sample_list,axis=0)
    X=sample_list
    #y = np.concatenate(category_index_list,axis=0)
    y = np.array(category_index_list)
    from tsfresh import extract_features, select_features
    from tsfresh.feature_extraction import EfficientFCParameters
    import tsfresh
    
    import pandas as pd

    def list2df(sample_list:List):
        
        # 假设你的向量列表是：
        arrays = sample_list

        dataframes = []

        for i, array in enumerate(arrays):
            array =  array.T
            df = pd.DataFrame(array, columns=['latitude', 'longitude', 'velocity', 'angle','time'])
            df['id'] = i  # 增加 'id' 列以区分不同的向量
            dataframes.append(df)
        
        tsfresh_df = pd.concat(dataframes).reset_index(drop=True)
        # tsfresh_df = tsfresh_df.drop('velocity',axis=1)
        # tsfresh_df = tsfresh_df.drop('angle',axis=1)
        return tsfresh_df
    
    tsfresh_df=list2df(X)
    
    tsfeatures = extract_features(tsfresh_df, column_id='id', column_sort='time',chunksize=None,default_fc_parameters = EfficientFCParameters())
    tsfeatures = tsfeatures.dropna(axis=1)
    #T=tsfresh.feature_selection.relevance.calculate_relevance_table(tsfeatures, y[0:22])
    tsfeatures=select_features(tsfeatures, y[0:len(X)],ml_task="classification",multiclass=True,n_significant=1)
    kind_to_fc_parameters=tsfresh.feature_extraction.settings.from_columns(tsfeatures)
    #tsfeatures = extract_features(tsfresh_df, column_id='id', column_sort='time',chunksize=None,kind_to_fc_parameters=kind_to_fc_parameters)
    # T=tsfresh.feature_selection.significance_tests.target_real_feature_real_test(tsfeatures, y[0:6])
    dynamic_features = tsfeatures.to_numpy()
    
    # dynamic_features = np.array(kinetic_feature(X,n_jobs=1))
    # dynamic_features = np.concatenate([dynamic_features,extra_feature],axis=-1)
    
    #catch22_features = np.array(tnf.fit_transform(X))
    #catch22_features = pca.fit_transform(catch22_features)
    #all_features = np.concatenate([dynamic_features,catch22_features],axis=-1)
    
    clf_01.fit(dynamic_features,np.array(category_index_01_list))
    clf_0.fit(dynamic_features[_0_index], y[_0_index])
    clf_1.fit(dynamic_features[_1_index], y[_1_index])
    clf0_14.fit(dynamic_features,y)
    #clf_2.fit(catch22_features,y)   
    
    sample_list,category_index_list,category_index_01_list,_0_index,_1_index,file_name_list,extra_feature_list = get_tracksets(valid_dataset)
    extra_feature = np.stack(extra_feature_list).squeeze()
    print(len(sample_list))
#     #aeon.datasets.write_to_tsfile(X=sample_list,path="./dataset",y=category_index_list,problem_name="haitun_TEST")

    #%%
    def print_matrix(label_list,predict_list):
        from sklearn.metrics import confusion_matrix
        conf_mat = confusion_matrix(np.array(label_list), np.array(predict_list))

        #print(conf_mat)
        
        from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score

        # 假设 y_true 是真实的标签，y_pred 是预测的标签
        y_true = np.array(label_list)
        y_pred = np.array(predict_list)

        precision = precision_score(y_true, y_pred, average='macro')
        accuraccy = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        # print('Precision: {}'.format(precision))
        # print('Accuraccy: {}'.format(accuraccy))
        # print('Recall: {}'.format(recall))
        #print('F1 Score: {}'.format(f1))
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
        dynamic_features = np.array(kinetic_feature(X_list,n_jobs=1))
        dynamic_features = np.concatenate([dynamic_features,extra_feature],axis=-1)
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
        
    tsfresh_df_test=list2df(sample_list)
    
    tsfeatures_test = extract_features(tsfresh_df_test, column_id='id', column_sort='time',chunksize=None,kind_to_fc_parameters=kind_to_fc_parameters)
    tsfeatures_test = tsfeatures_test.dropna(axis=1)
    tsfeatures = tsfeatures[tsfeatures_test.columns]
    
    dynamic_features = tsfeatures.to_numpy()

    clf_01 = RandomForestClassifier(n_estimators=500,n_jobs=-1)
    clf0_14 = RandomForestClassifier(n_estimators=500,n_jobs=-1)
    clf_0 = RandomForestClassifier(n_estimators=500,n_jobs=-1)
    clf_1 = RandomForestClassifier(n_estimators=500,n_jobs=-1)
    
    sample_list,category_index_list,category_index_01_list,_0_index,_1_index,file_name_list,extra_feature_list = get_tracksets(train_dataset)
    print(len(sample_list))
    print("临时切换train")
    clf_01.fit(dynamic_features,np.array(category_index_01_list))
    clf_0.fit(dynamic_features[_0_index], y[_0_index])
    clf_1.fit(dynamic_features[_1_index], y[_1_index])
    clf0_14.fit(dynamic_features,y)
    
    sample_list,category_index_list,category_index_01_list,_0_index,_1_index,file_name_list,extra_feature_list = get_tracksets(valid_dataset)
    extra_feature = np.stack(extra_feature_list).squeeze()
    print(len(sample_list))
    print("换回test")
    # T=select_features(tsfeatures, y[0:6])
    # T=tsfresh.feature_selection.significance_tests.target_real_feature_real_test(tsfeatures, y[0:6])
    dynamic_features = tsfeatures_test.to_numpy()
    #%% 01分类
    # dynamic_features = np.array(kinetic_feature(sample_list,n_jobs=1))
    # dynamic_features = np.concatenate([dynamic_features,extra_feature],axis=-1)
    prob = clf_01.predict_proba(dynamic_features)
    predict_list_01 = np.argmax(prob,axis=1)
    f1_bio,acc_bio = print_matrix(predict_list_01,category_index_01_list)
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
    predict_combine = np.int64(predict_combine).tolist()
    f1_multi,acc_multi = print_matrix(predict_combine,category_index_list)
    pen = pen_calculate(predict_list_01,predict_combine,category_index_01_list)
    print('Two stage F1 Score: {}'.format(((f1_bio+f1_multi)/2+(acc_bio+acc_multi)/2)/2-pen))
    print(pen)
    
    
    # pen = pen_calculate(predict_list_01,direct_predict_list,category_index_01_list)
    # print('Direct Combine F1 Score with penalty: {}'.format(((f1_bio+direct_f1)/2+(acc_bio+direct_acc)/2)/2-pen))
    # print(pen)
    
    # file_name_array = np.stack(file_name_list).squeeze()
    # data = np.column_stack((file_name_array,direct_predict_list, predict_list_01))
    
    # np.savetxt('result.txt', data, fmt='%s')
    
    exit(0)
    
    #%% 比赛部分
    racedataset = SubRaceDataset(0)
    sample_list,dummy_category_index_list,dummy_category_index_01_list,dummy_0_index,dummy_1_index,file_name_list,extra_feature_list = get_tracksets(racedataset)
    print(len(sample_list))
     #%% 01分类
    dynamic_features = np.array(kinetic_feature(sample_list,n_jobs=1))
    dynamic_features = np.concatenate([dynamic_features,extra_feature],axis=-1)
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
    
    np.savetxt('hq-02-XXX.txt', data, fmt='%s')