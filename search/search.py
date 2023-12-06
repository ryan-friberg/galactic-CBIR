import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import ToTensor

'''
This file defines the process by which visual search happens. This includes
the similarity scoring methodolody, and the actual querying on the dataset.
'''

def cosine_similarity(features1, features2):
    return F.cosine_similarity(features1, features2, dim=1)


def euclidean_distance(features1, features2, keepdim=True):
    return F.pairwise_distance(features1, features2, keepdim=keepdim)

def search(dataloader, model, scoring_fn, query_img_file, k=3):
    to_tensor = ToTensor()
    model.eval()

    # create batch out of the query image and extract query features
    query_batch = to_tensor(Image.open(query_img_file)).unsqueeze(0)
    query_features = model(query_batch)
    
    best_k = np.array([])
    best_imgs = np.array([])
    for img_file, feature_tensor in dataloader:
        sim_score = scoring_fn(query_features, feature_tensor)
        if (best_k < k):
            best_k = np.append(best_k, sim_score)
            best_imgs = np.append(best_imgs, img_file)

            sort_indices = np.argsort(best_k)
            best_k = best_k[sort_indices]
            best_imgs = best_imgs[sort_indices]
        elif (sim_score < best_k[-1]):
            best_k[-1] = sim_score
            best_imgs[-1] = img_file
            
            sort_indices = np.argsort(best_k)
            best_k = best_k[sort_indices]
            best_imgs = best_imgs[sort_indices]
    
    return best_imgs