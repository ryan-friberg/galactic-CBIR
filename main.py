import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from search.search import cosine_similarity, search
from search.search_dataset import SearchDataset
from torch.utils.data import DataLoader, random_split
from data.dataset import GalaxyCBRDataSet, collate_fn
from models.train import train, test
from models.transformer_feature_extractor import FeatureExtractorViT

'''
This file is the centerpiece of the CBIR pipeline. With the arguments below, this file can create
and/or load all the necessary components to train/test a machine learning model or run search. 
'''


# define CLI arugments, add more as needed
parser = argparse.ArgumentParser(description='Content-based image retrieval pipeline')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--device', default='cuda', type=str, help='cpu or cuda')
parser.add_argument('--optim', default='adam', type=str, help='training optimizer: sgd, sgd_nest, adagrad, adadelta, or adam')
parser.add_argument('--num_workers', default=1, type=int, help='number of workers')
parser.add_argument('--train_split', default=0.8, type=float, help='percentage of dataset to be used for training')
parser.add_argument('--epochs', default=5, type=int, help='numper of training epochs')
parser.add_argument('--batch_size', default=8, type=int, help='size of batch')
parser.add_argument('--checkpoint', default='', type=str, help='model checkpoint path')
parser.add_argument('--load', action='store_true', help='load from the given checkpoint')
parser.add_argument('--model', default='transformer', type=str, help='select which model architecture')
parser.add_argument('--data_dir', default='./data/galaxy_dataset/', type=str, help='location of data files')
parser.add_argument('--train', action='store_true', help='train model')
parser.add_argument('--test', action='store_true', help='test model')
parser.add_argument('--force_download', action='store_true', help='rebuild dataset filesytem')
parser.add_argument('--h5_file', default='', type=str, help='location of data source file')
parser.add_argument('--num_augmentations', default=3, type=int, help='number of augmentations during training')
parser.add_argument('--build_search', action='store_true', help='pre-compute the search database to save on inference time')
parser.add_argument('--search', action='store_true', help='run search with a query image and model checkpoint')
parser.add_argument('--query_image', default='', type=str, help='search query image filename')
parser.add_argument('--search_data_dir', default='', type=str, help='directory of search data files')
parser.add_argument('--search_output', default='./search/results.txt', type=str, help='output file for search results')
parser.add_argument('--k', default=3, type=int, help='number of search results to return')


# create the model given CLI arguments (only really useful when more than one model type is available)
def build_model(arch_name, batch_size=None):
    model = None
    default_checkpoint = None
    if (arch_name == 'transformer'):
        print("=> Transformer")
        model = FeatureExtractorViT((batch_size, 3, 224, 224))
        default_checkpoint = 'best_transformer.pt'
    return model, default_checkpoint


# build the corresponding optimizer based on what was requested via the CLI
def build_optim(optim_name, model, lr):
    # pick the model from the arguments
    optim = None
    if (optim_name == 'sgd'):
        print("=> SGD")
        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif (optim_name == 'sgd_nest'):
        print("=> SGD w/ Nesterov")
        optim = torch.optim.SGD(model.parameters(), nesterov=True, lr=lr, momentum=0.9, weight_decay=5e-4)
    elif (optim_name == 'adam'):
        print("=> Adam")     
        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    elif (optim_name == 'adagrad'):
        print("=> Adagrad")
        optim = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=5e-4)
    elif (optim_name == 'adadelta'):
        print("=> Adadelta")
        optim = torch.optim.Adadelta(model.parameters(), lr=lr, weight_decay=5e-4)
    return optim


# search for a hardware accelerator, if available
def determine_device(requested_device_name):
    print("===> Determining device...")
    if requested_device_name == 'cuda':
        print("=> GPU requested")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=> Using device:", device)
    return device


# load a model from a previous training run
def load_checkpoint(model, optimizer, filename):
    if (filename == ''):
        print("=> No checkpoint path specified!")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return (checkpoint['epoch'] + 1), checkpoint['best_loss']


# run the pipeline
def main():
    args = parser.parse_args()

    print('===> Building model...')
    model, default_checkpoint = build_model(args.model, args.batch_size)
    total_params = sum(p.numel() for p in model.parameters())
    print("=> Model parameter count:", total_params)

    device = determine_device(args.device)
    model.to(device)

    print("===> Building optimizer...")
    optim = build_optim(args.optim, model, args.lr)
    scoring_fn = cosine_similarity

    start = 0
    best_loss = np.Inf
    if (args.load):
        print('===> Loading checkpoint...')
        start, best_loss = load_checkpoint(model, optim, args.checkpoint)
        print("=> Loaded!")

    print("===> Building dataset and dataloaders...")
    data_transforms = transforms.ToTensor()
    galaxy_dataset = GalaxyCBRDataSet(args.data_dir, data_transforms, force_download=args.force_download, h5_file=args.h5_file)

    train_size = int(args.train_split * len(galaxy_dataset))
    test_size  = int((len(galaxy_dataset) - train_size) / 2)
    val_size   = (len(galaxy_dataset) - train_size - test_size)
    print("=> train size: %d, test size: %d, val_size: %d" % (train_size, test_size, val_size))

    train_dataset, test_dataset, val_dataset = random_split(galaxy_dataset,[train_size, test_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    if ((model == None) or (optim == None)):
        print("Invalid parameters. Please check optimizer and model spelling!")
        return -1

    if (args.test):
        print("===> Testing...")
        test(model, test_loader, args.num_augmentations, scoring_fn, device)
    if (args.train):
        print("===> Training...")
        if (args.checkpoint == ''):
            checkpoint = default_checkpoint
        else:
            checkpoint = args.checkpoint
        print("=> Checkpoint file:", checkpoint)
        
        train(model, train_loader, val_loader, train_dataset, val_dataset, optim, scoring_fn, device, start_epoch=start, 
              num_epochs=args.epochs, num_augmentations=args.num_augmentations, validate_interval=5, best_loss=best_loss,
              checkpoint_filename=checkpoint)
        print("=> Training complete!")

        # reload the best performing model
        start, best_loss = load_checkpoint(model, optim, checkpoint)
    if (args.build_search):
        print("===> Building search dataset...")
        if (not args.load):
            if (args.checkpoint == ''):
                print("=> Model not loaded or no trained model specified")
                return -1
            else:
                # load the best model (as it would not have happened earlier)
                start, best_loss = load_checkpoint(model, optim, checkpoint)

        search_dataset = SearchDataset(args.search_data_dir, model, galaxy_dataset, device, extract_features=True)
        print("=> Search dataset build complete!")
    if (args.search):
        print("===> Searching...")
        if ((args.query_image == '') or (args.search_data_dir == '')):
            print("=> No query image and/or search dataset directory specified!")
            return -1
        elif (not args.load):
            if (args.checkpoint == ''):
                print("=> Model not loaded or no trained model specified")
                return -1
            else:
                # load the best model (as it would not have happened earlier)
                start, best_loss = load_checkpoint(model, optim, checkpoint)

        search_dataset = SearchDataset(args.search_data_dir, model, galaxy_dataset, device, extract_features=False)
        search_loader  = DataLoader(search_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
        top_k = search(search_loader, model, scoring_fn, args.query_image, k=args.k)

        print("=> Top %d closest images found, saving to %s..." % (args.k, args.search_output))
        with open(args.search_output, 'w') as file:
            file.write(f"{args.query_image}\n")
            for img_file in top_k:
                file.write(f"{img_file}\n")
    print("===> Pipeline completed!")
    return 0


if __name__=='__main__':
    main()