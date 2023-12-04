import os
import numpy as np
import pandas as pd
from multiprocessing import Pool

# 定义处理函数，用于解析文件并转换为numpy数组
def process_file(file_path):
    
    df = pd.read_csv(file_path, delimiter=' ', header=None)
    df[0] = df[0].str.replace('T', '')   
    # 解析每行数据并将其转换为numpy数组
    # 将DataFrame转换为numpy数组
    numpy_array = df.values[:, 1:].astype(np.float64)
    
    return numpy_array

def Readcsv(dataset_folder):
    """
    返回嵌套列表，第一层是类别，第二层是某一类的轨迹样本
    """
    # # 设置数据集文件夹路径
    # dataset_folder = '/mnt/d/haitundata'

    # 遍历标签文件夹
    label_folders = [label_folder for label_folder in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, label_folder))]
    label_folders.sort(key=lambda x:int(x.split('.')[0]))
    # 使用多进程处理每个标签文件夹下的文件
    pool = Pool()

    result = []
    for label_folder in label_folders:
        label_folder_path = os.path.join(dataset_folder, label_folder)
        files = [os.path.join(label_folder_path, file) for file in os.listdir(label_folder_path) if os.path.isfile(os.path.join(label_folder_path, file))]
        result.append(pool.map(process_file, files))

    pool.close()
    pool.join()
    # result列表包含了所有文件的numpy数组

    return result

if __name__ == "__main__":
    print(len(Readcsv("/mnt/d/haitundata")[0]))