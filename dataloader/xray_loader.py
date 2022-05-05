import torch
import torchvision
from PIL import Image
import os
import numpy as np

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe,INPUT_PATH = "drive/MyDrive/dataset/griffith/xray/",num_class=16):
        self.dataframe = dataframe
        self.INPUT_PATH = INPUT_PATH

        self.transform = torchvision.transforms.Compose([torchvision.transforms.Resize((256, 256)),
                                        torchvision.transforms.ToTensor()
                                        ])
        self.num_class = num_class
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image = Image.open(os.path.join(self.INPUT_PATH,row["image_path"]))
        # torchvision.transforms.functional.to_tensor()
        target = int(row["Target"])
        b = np.zeros(self.num_class)
        b[target]=1
        #print(b[target])
        return self.transform(image), b