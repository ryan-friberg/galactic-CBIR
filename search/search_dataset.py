import numpy as np
import os
from tqdm import tqdm 
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_pil_image

'''
This file defines a dataset that is built after model training. It essentially serves
as the pre-compution for all the feature extractions in the dataset. The main purpose of 
this dataset is to speed up search by removing search-time model inference
'''

class SearchDataset(Dataset):
    def __init__(self, data_dir, model, associated_dataset, device, extract_features=False):
        self.data_dir = data_dir
        self.device = device

        if (not os.path.exists(data_dir)):
            os.mkdir(data_dir)

        if extract_features:
            self.extract_and_save_features(model, associated_dataset)
        
        galaxy_dataset_indices, tensor_files = self.get_filenames()
        self.galaxy_dataset_indices = np.array(galaxy_dataset_indices)
        self.tensor_files = np.array(tensor_files)
        self.num_files = len(self.galaxy_dataset_indices)

    def __getitem__(self, idx):
        galaxy_idx = self.galaxy_dataset_indices[idx]
        features = torch.load(self.tensor_files[idx], map_location=self.device)
        return galaxy_idx, features

    def __len__(self):
        return self.num_files

    def extract_and_save_features(self, model, associated_dataset):
        model.eval()    
    
        for idx in tqdm(range(len(associated_dataset))):
            img_tensor, labels = associated_dataset[idx]
            img_tensor = img_tensor.unsqueeze(0).to(self.device)

            # define file structure/names
            tensor_file = os.path.join(self.data_dir, str(int(idx)) + '_features.pt')
            
            # get image/features
            feature_tensor = model(img_tensor)
            torch.save(feature_tensor, tensor_file)

    def get_filenames(self):
        galaxy_dataset_indices = []
        tensor_files = []

        files = os.listdir(self.data_dir)
        for i, file in enumerate(files):
            file = os.path.join(self.data_dir, file)
            galaxy_dataset_indices.append(i)
            tensor_files.append(file)

        return galaxy_dataset_indices, tensor_files