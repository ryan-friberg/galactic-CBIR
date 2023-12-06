import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

'''
This file defines the process by which visual search happens. This includes
the similarity scoring methodolody, and the actual querying on the dataset.
'''

def cosine_similarity(features1, features2):
    return F.cosine_similarity(features1, features2, dim=1)


def euclidean_distance(features1, features2, keepdim=True):
    return F.pairwise_distance(features1, features2, keepdim=keepdim)


def get_top_k_indices(query_features, all_features, k):
    # Compute similarity scores between query features and all features
    similarities = cosine_similarity(query_features.unsqueeze(0), all_features)
    # Find top-k indices based on similarity scores
    _, top_k_indices = torch.topk(similarities, k, largest=True)
    return top_k_indices


def retrieve_images(dataset, indices):
    # Fetch images from the dataset based on indices
    images = [dataset[idx] for idx in indices]
    return images


def save_query_and_similar_images(query_image, similar_images, save_path, file_name):
    # TODO: this function assumes that 
    # query_image and similar_images are PIL images or tensors that can be converted to PIL images
    fig, axs = plt.subplots(1, len(similar_images) + 1, figsize=(15, 5))
    axs[0].imshow(query_image.permute(1, 2, 0))  # Convert tensor to image
    axs[0].set_title("Query Image")
    axs[0].axis('off')

    for i, img in enumerate(similar_images):
        axs[i + 1].imshow(img.permute(1, 2, 0))  # Convert tensor to image
        axs[i + 1].set_title(f"Similar {i+1}")
        axs[i + 1].axis('off')

    plt.savefig(f"{save_path}/{file_name}")
    plt.close()


def search_dataset(features_dataset, model):
    pass