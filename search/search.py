import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms

'''
This file defines the process by which visual search happens. This includes
the similarity scoring methodolody, and the actual querying on the dataset.
'''


# the chosen scoring function to be used during training
def similarity_scoring_function(features1, features2):
    cosine_sim = torch.mean(F.cosine_similarity(features1, features2, dim=1))
    l2_dist    = torch.mean(F.pairwise_distance(features1, features2, p=2))
    minkowsi   = torch.mean(F.pairwise_distance(features1, features2, p=2))
    similarity_score =  cosine_sim + (1 / (1 + l2_dist)) + (1 / (1 + minkowsi))
    return similarity_score


# walk through the pre-computed search database and maintain a top-k results list
def search(dataset, model, scoring_fn, query_img_file, device, k=3):
    model.eval()

    # create batch out of the query image and extract query features
    to_batch = transforms.Compose([transforms.ToTensor(), transforms.Resize(224, antialias=False)])
    query_batch = to_batch(Image.open(query_img_file)).unsqueeze(0).to(device)
    query_features = model(query_batch)
    
    best_k = np.array([])
    best_imgs = np.array([])
    for idx in tqdm(range(len(dataset))):
        img_file, feature_tensor = dataset[idx]
        feature_tensor = feature_tensor.to(device)
        sim_score = scoring_fn(query_features, feature_tensor).item()
        if (len(best_k) < k):
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


def save_search_results(results_file, query_image, top_k):
    with open(results_file, 'w') as file:
        file.write(f"{query_image}\n")
        for img_file in top_k:
            file.write(f"{img_file}\n")


# parse the results file and return a dictionary of images
def get_search_results(results_file):
    res = []
    
    with open(results_file) as f:
        lines = [line.rstrip() for line in f]
    
    for i, img_file in enumerate(lines):
        res.append(Image.open(img_file))
    
    return lines, res