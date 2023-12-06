#   # this may throw a tensorflow error
import glob
import h5py
import itertools
import numpy as np
import os
from PIL import Image
import torch

from torch.utils.data import Dataset
import torchvision.transforms as transforms


'''
This file defines our custom pytorch Dataset class using the
Galaxy10 dataset.

"labels" are really indices into the dataset to ensure negative
samples do not sample the same image during train/test

Datasets used:
https://astronn.readthedocs.io/en/latest/galaxy10.html
'''

# dataset of 10 classes of galaxies, stored in the filesystem as 3x256x256 images in
# g, r, and z bands (as commonly done with astronomical images)
class GalaxyCBRDataSet(Dataset):
    def __init__(self, images_dir, transforms, force_download=False, h5_file=''):
        self.transforms = transforms
        self.supported_file_types = ['/*.jpg']
        self.images_dir = images_dir 

        if (not os.path.exists(images_dir)):
            os.mkdir(images_dir)

        if force_download:
            self.download_galaxy10_data(h5_file)
        
        image_files, labels = self.get_image_filenames_with_labels(self.images_dir)
        self.image_files = np.array(image_files)
        self.labels = np.array(labels).astype("int")
        self.num_images = len(self.image_files)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_files[idx]).convert('RGB')
            if image.size != (224,224):
                image = image.resize((224,224))
            label = self.labels[idx]
            image = self.transforms(image)
            return image, label
        except:
            return None

    def __len__(self):
        return self.num_images 

    def download_galaxy10_data(self, h5_filename):
        if (h5_filename != ''):
            with h5py.File(h5_filename, 'r') as F:
                images = np.array(F['images'])
                labels = np.array(F['ans'])
        else:
            # awkwardly put this here to avoid repeat tf cuda warnings
            from astroNN.datasets import load_galaxy10

            images, labels = load_galaxy10()

        label_set = np.unique(labels)
        for unique_label in label_set:
            label_dir = os.path.join(self.images_dir, str(int(unique_label)))
            if (not os.path.exists(label_dir)):
                os.mkdir(label_dir)

            indices = np.where(labels == unique_label)[0]
            image_idx = 0
            print("Saving class", str(unique_label), "images...")
            for idx in indices:
                file_name = os.path.join(label_dir, str(image_idx) + ".jpg")
                image = Image.fromarray(images[idx].astype('uint8'))
                image.save(file_name)
                image_idx += 1

    def get_image_filenames_with_labels(self, images_dir):
        image_files = []
        labels = []

        files = os.listdir(images_dir)
        for name in files:
            if name == ".DS_Store":
                continue
            image_class_dir = os.path.join(images_dir, name)
            image_class_files = list(itertools.chain.from_iterable(
                [glob.glob(image_class_dir + file_type) for file_type in self.supported_file_types]))
            image_files += image_class_files
        labels = np.arange(0, len(image_files), dtype=int)
        return image_files, labels


# simple collation function to be used in the future for the DataLoader
# (I believe this is the same as the default collate_fn)
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    images = torch.stack([b[0] for b in batch])
    labels = torch.LongTensor([b[1] for b in batch])
    return images, labels