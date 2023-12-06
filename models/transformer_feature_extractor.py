import numpy as np
import torch
import torch.nn as nn

'''
This file's purpose is to define the learning-based models used for feature extraction. Example
architectures include a vision transformer, and a CNN-based variational autoencoder.

This file may potentially include pre-trained and from-scratch models.

Implementation based architecture diagram from: https://arxiv.org/pdf/2010.11929.pdf 
'''

# backward pass is handled by the autograd functionality of pytorch
class FeatureExtractorViT(nn.Module):
    # n_patches is number of patches along a dimension
    def __init__(self, batch_shape, n_patches=14, hidden_size=1024, num_blocks=3, num_heads=4, output_feature_size=2048):
        super(FeatureExtractorViT, self).__init__()
        assert (batch_shape[1:] == (3,224,224)) # the econding works only for specific image shape
        
        ### assign class variables
        self.shape = batch_shape
        self.n_patches = n_patches

        ### define image resizing based on the image patch size/shape
        self.new_shape = (batch_shape[0], n_patches**2, (batch_shape[1]*batch_shape[2]*batch_shape[3]) // n_patches**2)
        self.p_shape   = (batch_shape[2]/n_patches, batch_shape[3]/n_patches)

        ### define image patching size
        self.p_size  = self.shape[2] // self.n_patches

        ### define the actual transformer model architecture
        self.linear         = nn.Linear(int(batch_shape[1] * self.p_shape[0]**2), hidden_size)
        attn_layers         = [ViTEncoderBlock(hidden_size, num_heads) for _ in range(num_blocks)]
        self.encoder_blocks = nn.Sequential(*attn_layers)
        self.output_layer   = nn.Linear(hidden_size, output_feature_size)
        
        ### define positional embedding, mark it untrainable, and add it to the state_dict
        self.pos_embed = nn.Parameter(self.positional_embedding(n_patches ** 2, hidden_size))
        self.pos_embed.requires_grad = False
        self.register_buffer('pos_embedding', self.pos_embed, persistent=False)

    def positional_embedding(self, sequence_len, embed_dim):
        position = torch.arange(0, sequence_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(np.log(10000.0) / embed_dim))
        embeddings = torch.zeros(sequence_len, embed_dim)
        embeddings[:, 0::2] = torch.cos(position * div_term)
        embeddings[:, 1::2] = torch.sin(position * div_term)
        return embeddings

    def forward(self, batch):
        # unfold the images in the batch into tensors
        # first reshape the patches to have shape (batch_size, num_patches, channels, patch_size, patch_size)
        # then reshape to (batch_size, num_patches_per_image, num_channels * num_elements_in_patch)
        patches = batch.unfold(2, self.p_size, self.p_size).unfold(3, self.p_size, self.p_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(batch.size(0), -1, batch.size(1), self.p_size, self.p_size)
        patches = patches.view(batch.size(0), -1, batch.shape[1] * self.p_size * self.p_size)

        z = self.linear(patches)
        pos_embed = self.pos_embed.repeat(batch.shape[0], 1, 1)
        z = z + pos_embed
        z = self.encoder_blocks(z)
        z = self.output_layer(z)
        return z


class MultiHeadAttentionEncoder(nn.Module):
    def __init__(self, dim, num_heads):
        super(MultiHeadAttentionEncoder, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = int(dim // num_heads)

        # create the query, key, value layers (each list is the same length the entries at index i represent attn head i)
        self.multihead_q = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for q in range(self.num_heads)])
        self.multihead_k = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for k in range(self.num_heads)])
        self.multihead_v = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for v in range(self.num_heads)])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, linear_embed):
        # linear embedding has shape of 
        output = []
        for img in linear_embed:
            # img is of shape num_patches x patch_len
            attn_res = []
            for head in range(self.num_heads):
                head_input = img[:, head * self.head_dim: (head + 1) * self.head_dim]
                q = self.multihead_q[head](head_input)
                k = self.multihead_k[head](head_input) 
                v = self.multihead_v[head](head_input)

                # matmul k and q, (technically scale, then mask) then apply softmax
                # matmul the total result with value tensor
                attn = self.softmax(q @ k.T / np.sqrt(self.head_dim)) @ v
                attn_res.append(attn)

        output.append(torch.hstack(attn_res))
        return torch.cat([torch.unsqueeze(z, dim=0) for z in output])


class ViTEncoderBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(ViTEncoderBlock, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention = nn.Sequential(nn.LayerNorm(self.hidden_size),
                                       MultiHeadAttentionEncoder(self.hidden_size, self.num_heads))
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.mlp = nn.Sequential(nn.Linear(self.hidden_size, 2*self.hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(2*self.hidden_size, self.hidden_size))
    
    def forward(self, x):
        z = x + self.attention(x)
        z = self.layer_norm(z)
        z = z + self.mlp(z)
        return z