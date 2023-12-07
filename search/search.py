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


def get_index(arr, target):
    indices = np.where(arr == target)[0]
    return indices[0] if indices.any() else -1


# walk through the pre-computed search database and maintain a top-k results list
def search(search_dataset, image_dataset, model, scoring_fn, query_img_file, device, k=3):
    model.eval()

    # create batch out of the query image and extract query features
    to_batch = transforms.Compose([transforms.ToTensor(), transforms.Resize(224, antialias=False)])
    query_batch = to_batch(Image.open(query_img_file)).unsqueeze(0).to(device)
    query_features = model(query_batch)
    query_idx = get_index(image_dataset.image_files, query_img_file)
    
    best_k = np.array([])
    best_img_files = np.array([])
    for idx in tqdm(range(len(search_dataset))):
        img_idx, feature_tensor = search_dataset[idx]
        
        # if running search with element from the dataset, make sure
        # to not have the query image in the results
        if (img_idx == query_idx):
            continue        
        
        img_file = image_dataset.image_files[img_idx]
        feature_tensor = feature_tensor.to(device)
        sim_score = scoring_fn(query_features, feature_tensor).item()
        if (len(best_k) < k):
            best_k = np.append(best_k, sim_score)
            best_img_files = np.append(best_img_files, img_file)

            sort_indices = np.argsort(best_k)
            best_k = best_k[sort_indices]
            best_img_files = best_img_files[sort_indices]
        elif (sim_score > best_k[0]):
            best_k[0] = sim_score
            best_img_files[0] = img_file
            
            sort_indices = np.argsort(best_k)
            best_k = best_k[sort_indices]
            best_img_files = best_img_files[sort_indices]
    return best_img_files


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
    
    file_names = []
    for line in lines:
        file_names.append(line.replace('./data/galaxy_dataset', ''))

    return file_names, res