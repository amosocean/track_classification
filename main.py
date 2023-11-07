from scipy.io import loadmat
import os
dir_path=os.path.dirname(os.path.abspath(__file__))
mat_contents = loadmat(os.path.join(dir_path,'source/matlab/savedData.mat'))
print(mat_contents.keys())
print(type(mat_contents))