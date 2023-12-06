from PIL import Image
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_pil_image

'''
This file defines a dataset that is built after model training. It essentially serves
as the pre-compution for all the feature extractions in the dataset. The main purpose of 
this dataset is to speed up search by removing search-time model inference
'''

class SearchDataset(Dataset):
    def __init__(self, data_dir, model, associated_dataset, device, extract_features=True):
        self.data_dir = data_dir
        self.device = device

        if (not os.path.exists(data_dir)):
            os.mkdir(data_dir)

        if extract_features:
            self.extract_and_save_features(model, associated_dataset)
        
        self.image_files, self.tensor_files = self.get_filenames(self.data_dir)
        self.num_files = len(self.image_files)

    def __getitem__(self, idx):
        # returns a tuple of the PIL image (for visualization) and torch feature tensor
        try:
            image = Image.open(self.image_files[idx]).convert('RGB')
            features = torch.load(self.features_files, map_location=self.device)
            return image, features
        except:
            return None

    def __len__(self):
        return self.num_files

    def extract_and_save_features(self, model, associated_dataset):
        model.eval()
        loader = DataLoader(associated_dataset, batch_size=1, shuffle=False)        
        
        for i, img_tensor, labels in enumerate(loader):
            # define file structure/names
            pair_dir = os.path.join(self.data_dir, str(int(i)))
            img_file = os.path.join(pair_dir, 'img.jpg')
            features_file = os.path.join(pair_dir, 'features.pt')
            
            if (not os.path.exists(pair_dir)):
                os.mkdir(pair_dir)
            
            # get image/features
            img = to_pil_image(img_tensor)
            feature_tensor = model(img_tensor)
            
            # save to filesystem
            img.save(img_file)
            torch.save(feature_tensor, features_file)

    def get_filenames(self):
        image_files   = []
        tensor_files = []

        pair_dirs = os.listdir(self.data_dir)
        for dir in pair_dirs:
            dir = os.path.join(self.data_dir, dir)
            img_file = os.path.join(dir, 'img.jpg')
            features_file = os.path.join(dir, 'features.pt')
            image_files.append(img_file)
            tensor_files.append(features_file)

        return image_files, tensor_files