#!/usr/bin/env python3
import rospy, rospkg
import numpy as np
import pickle
from os.path import join
from PIL import Image
import torch 


from data_recorder import DataRecorded

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pickle

transform_image = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_array = transforms.Compose([
    transforms.ToTensor(),
])

class SAP_DataSet(Dataset):
    def __init__(self, filename: str, i_use_action:bool = True, i_use_wrench: bool = True, i_use_position:bool = False, o_use_wrench:bool = True, o_use_position:bool = False):
        rospack = rospkg.RosPack()
        path = join(rospack.get_path('robot_driver'), "data")
        file_path = join(path, filename)
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)
            
        self.i_use_action = i_use_action
        self.i_use_wrench = i_use_wrench
        self.i_use_position = i_use_position
        self.o_use_wrench = o_use_wrench
        self.o_use_position = o_use_position
        
        '''
            data_norm = (data - mean)/std
        '''
        wrench_std = []
        wrench_mean  = []
        for i in range(6):
            _std= np.std([item.wrench[i] for item in self.data])
            _mean = np.mean([item.wrench[i] for item in self.data])
            wrench_std.append(_std)
            wrench_mean.append(_mean)
        
        for item in self.data:
            for i in range(6):
                item.wrench[i] = (item.wrench[i] - wrench_mean[i])/wrench_std[i]
                item.wrench_plus1[i] = (item.wrench_plus1[i] - wrench_mean[i])/wrench_std[i]
        
        '''
        position_std = []
        position_mean  = []
        for i in range(3):
            _std = np.std([item.pose[i] for item in self.data])
            _mean = np.mean([item.pose[i] for item in self.data])
            position_std.append(_std)
            position_mean.append(_mean)
        
        for item in self.data:
            for i in range(3):
                item.pose[i] = (item.pose[i] - position_mean[i])/position_std[i]
                item.pose_plus1[i] = (item.pose_plus1[i] - position_mean[i])/position_std[i]        
        '''
        position_max = [] #x,y,z
        position_min  = [] #x,y,z
        for i in range(3):
            _max = np.max([item.pose[i] for item in self.data])
            _min = np.min([item.pose[i] for item in self.data])
            position_max.append(_max)
            position_min.append(_min)
        
        for item in self.data:
            for i in range(3):
                item.pose[i] = (item.pose[i] - position_min[i])/(position_max[i] - position_min[i])
                item.pose_plus1[i] = (item.pose_plus1[i] - position_min[i])/(position_max[i] - position_min[i])   
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if self.data[index].image.shape[2] == 3:
            image = Image.fromarray(self.data[index].image).convert('L')
            image_plus1 = Image.fromarray(self.data[index].image_plus1).convert('L')
        
        info = np.array(self.data[index].action).reshape(-1) if self.i_use_action else np.array([])
        info_plus1 = np.array([])
        
        if self.i_use_wrench:
            info = np.concatenate((info, self.data[index].wrench))
            
        if self.i_use_position:
            info = np.concatenate((info, self.data[index].pose[0:3]))
            
        if self.o_use_wrench:
            info_plus1 = np.concatenate((info_plus1, self.data[index].wrench_plus1))
            
        if self.o_use_position:
            info_plus1 = np.concatenate((info_plus1, self.data[index].pose_plus1[0:3]))            
            
        return transform_image(image), torch.tensor(info.astype(np.float32)), transform_image(image_plus1), torch.tensor(info_plus1.astype(np.float32))

# if __name__ == "__main__":
#     dataset = MyDataset()
#     i = iter(dataset)
#     print(next(i).shape)
#     print(len(dataset))


FILE_NAME = "data_20240729_0928_L1097.pkl"
rospack = rospkg.RosPack()
path = join(rospack.get_path('robot_driver'), "data")
file_path = join(path, FILE_NAME)

if __name__ == "__main__":
    # with open(file_path, 'rb') as f:
    #     data = pickle.load(f)
    #     if data[0].image.shape[2] == 3:
    #         image = Image.fromarray(data[0].image).convert('L')
    #         image_plus1 = Image.fromarray(data[0].image_plus1).convert('L')
    #         image.show()
    #         image_plus1.show()
    #         print(np.array(image))
    #         print(np.array(image_plus1))
            

        # print(data)
    
    dataset = SAP_DataSet(FILE_NAME, i_use_position=True, i_use_wrench=True, o_use_position=True, o_use_wrench=True)
    data = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(next(iter(dataset)))
    for idx, (image, info, image_plus1, info_plus1) in enumerate(data):  
        print(image.shape)