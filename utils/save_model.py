import pickle

def save(classifiers):
    
    with open("clfs.plk","wb") as file:
        pickle.dump(classifiers,file)

def read():
        
    with open("clfs.plk","rb") as file:
        clf_list=pickle.load(file)

    return clf_list


# %%
