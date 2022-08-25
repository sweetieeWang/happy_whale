# coding=utf-8
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
import pandas as pd
from PIL import Image
# from sklearn.model_selection import train_test_split

def convert_label(data):
    columns=['species','individual_id']
    for f in columns:
        data[f]=data[f].map(dict(zip(data[f].unique(),range(0,data[f].nunique()))))
    return data

def file_split(data_path, csv_name, train_classes, species, special_id,percentage):
    '''
    special_id:新类个数
    percentage:训练集百分比
    '''
    data = pd.read_csv(os.path.join(data_path,csv_name))
    data['species']=data['species'].map(dict(zip(data['species'].unique(),range(0,data['species'].nunique()))))
    data_copy=data.drop_duplicates(subset=['individual_id'] , keep='first', inplace=False)
    test_id=[]
    # 训练集类数+新类数
    numbers=random.sample(range(train_classes+special_id),special_id)
    for i in range(special_id):
        item=str(data_copy.iloc[numbers[i],2])
        test_id.append(item)
    #print(test_id)
    special=data[data['individual_id'].isin(test_id)]
    other=data[~data['individual_id'].isin(test_id)]
    other = convert_label(other)
    # 训练集spieces数 暂不用
    special.reset_index(drop=True, inplace=True)
    special['species'] =pd.Series((species)*np.ones(special.shape[0]))
    # 训练集类数
    special['individual_id'] =pd.Series((train_classes)*np.ones(special.shape[0]))

    other = other.sample(frac=1.0)  # 全部打乱
    cut_idx = int(round(percentage* other.shape[0]))
    other_all, special_all = other.iloc[:cut_idx], other.iloc[cut_idx:]
    # for i in special_all.itertuples():
    #     if i[3] ==358:
    #         print(i)
    # print('------')
    # for i in other_all.itertuples():
    #     if i[3] ==358:
    #         print(i)
    # special_all=special_all.append(special)
    
    return other_all,special_all

#print(data)
class WhaleDataset(Dataset):
    '''
    数据集，存在3个特征
    image，piece, label
    '''
    def __init__(self, data, data_path, transform= None):
        self.data_path = data_path

        data = np.array(data).tolist()
        self.data = []
        for line in data:
            # line = np.char.rsplit(np.char.strip(line), ' ')
            self.data.append(line)
        self.transform = transform
        self.target_transform = None
        print('init final!')
        #print(self.data)
    def __getitem__(self, index):
        fn, piece, label = self.data[index]
        # print(type(fn),type(piece),type(label))
        fn, piece, label = fn, np.array(int(piece)), np.array(int(label))
        image = Image.open(os.path.join(self.data_path,'train_images',fn)).convert('RGB')
        w,h = image.size
        if w>=h:
            image=image.resize([224,int(224*h/w)])
        else:
            image=image.resize([int(224*w/h),224])
        #image.show()
        #image = np.array(image)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return {'image':image, 'species':torch.from_numpy(piece), 'label':torch.from_numpy(label)}
    def __len__(self):
        return len(self.data)

#train_ds, val_ds = file_split('/home/public/happy-whale-and-dolphin', 'train.csv', 15287, 30, 300, 0.7)
# train_transform = transforms.Compose([
#             transforms.RandomCrop([224, 224],pad_if_needed=True,padding_mode=symmetric),
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.Resize([224,224]),

#             # 归一化 均值 方差待修正
#             transforms.ToTensor(),
#             transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
#             # [0.656,0.487,0.411], [1., 1., 1.]
#             ])

# ds_train = WhaleDataset(train_ds, '/home/public/happy-whale-and-dolphin',transform=train_transform)
# plt.figure("Image") # 图像窗口名称
# for i in range(20):
#     image = ds_train[i]['image']
#     img = transforms.ToPILImage()(image)
#     plt.imshow(img)
#     plt.axis('on') # 关掉坐标轴为 off
#     plt.title('img') # 图像题目
#     plt.show()