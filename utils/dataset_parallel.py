import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor

# 定义处理函数，用于解析文件并转换为numpy数组
def process_file(file_path):
    
    df = pd.read_csv(file_path, delimiter=' ', header=None)
    df[0] = df[0].str.replace('T', '')   
    # 解析每行数据并将其转换为numpy数组
    # 将DataFrame转换为numpy数组
    numpy_array = df.to_numpy(dtype=np.float64,na_value=0)
    numpy_array = numpy_array[:, np.concatenate([np.arange(1, numpy_array.shape[1]), [0]])]
    return numpy_array.T

def Readcsv(dataset_folder):
    """
    返回嵌套列表，第一层是类别，第二层是某一类的轨迹样本
    """
    # 遍历标签文件夹
    label_folders = [label_folder for label_folder in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, label_folder))]
    label_folders.sort(key=lambda x:int(x.split('.')[0]))
    # 使用线程池处理每个标签文件夹下的文件
    with ProcessPoolExecutor(max_workers=16) as executor:
        result = []
        for label_folder in label_folders:
            label_folder_path = os.path.join(dataset_folder, label_folder)
            files = [os.path.join(label_folder_path, file) for file in os.listdir(label_folder_path) if os.path.isfile(os.path.join(label_folder_path, file))]
            res = list(executor.map(process_file, files))
            result.append(res)

    # result列表包含了所有文件的numpy数组

    return result

if __name__ == "__main__":
    print(len(Readcsv("/mnt/d/haitundata")[0]))