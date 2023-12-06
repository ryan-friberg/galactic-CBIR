import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from search.search import cosine_similarity
from search.search_dataset import SearchDataset
from torch.utils.data import DataLoader, random_split
from data.dataset import GalaxyCBRDataSet, collate_fn
from models.train import train, test
from models.transformer_feature_extractor import FeatureExtractorViT

# define CLI arugments, add more as needed
parser = argparse.ArgumentParser(description='Content-based image retrieval pipeline')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--device', default='cuda', type=str, help='cpu or cuda')
parser.add_argument('--optim', default='adam', type=str, help='training optimizer: sgd, sgd_nest, adagrad, adadelta, or adam')
parser.add_argument('--num_workers', default=1, type=int, help='number of workers')
parser.add_argument('--train_split', default=0.8, type=float, help='percentage of dataset to be used for training')
parser.add_argument('--epochs', default=5, type=int, help='numper of training epochs')
parser.add_argument('--batch_size', default=8, type=int, help='size of batch')
parser.add_argument('--load', default='', type=str, help='model checkpoint path')
parser.add_argument('--model', default='transformer', type=str, help='select which model architecture')
parser.add_argument('--data_dir', default='./data/galaxy_dataset/', type=str, help='location of data files')
parser.add_argument('--train', action='store_true', help='train model')
parser.add_argument('--test', action='store_true', help='test model')
parser.add_argument('--force_download', action='store_true', help='rebuild dataset filesytem')
parser.add_argument('--h5_file', default='', type=str, help='location of data source file')
parser.add_argument('--num_augmentations', default=3, type=int, help='number of augmentations during training')

# TODO: implement main.py search functionality
parser.add_argument('--search', action='store_true', help='run search with a query image and model checkpoint\
                                                                      instead of train/test')
parser.add_argument('--query_image', default='', type=str, help='search query image filename')
parser.add_argument('--search_database', default='', type=str, help='directory of search data files')


def build_model(arch_name, batch_size=None):
    model = None
    checkpoint_path = None
    if (arch_name == 'transformer'):
        print("=> Transformer")
        model = FeatureExtractorViT((batch_size, 3, 224, 224))
        checkpoint_path = 'best_transformer.pt'
    elif (arch_name == 'cnn'):
        print("=> CNN")
        model = None
        checkpoint_path = 'best_cnn.pt'
    return model, checkpoint_path


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


def determine_device(requested_device_name):
    print("===> Determining device...")
    if requested_device_name == 'cuda':
        print("=> GPU requested")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=> Using device:", device)
    return device


def load_checkpoint(model, optimizer, filename='checkpoint.pth.tar'):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return (checkpoint['epoch'] + 1), checkpoint['best_loss']


def main():
    args = parser.parse_args()

    # build appropriate model
    print('===> Building model...')
    model, checkpoint_path = build_model(args.model, args.batch_size)
    total_params = sum(p.numel() for p in model.parameters())
    print("=> Model parameter count:", total_params)

    device = determine_device(args.device)
    model.to(device)

    print("===> Building optimizer...")
    optim = build_optim(args.optim, model, args.lr)
    scoring_fn = cosine_similarity

    # load pre-trained checkpoint, if specified
    start = 0
    best_loss = np.Inf
    if args.load != '': # model needs to be loaded to the same device it was saved from
        print('===> Loading checkpoint...')
        start, best_loss = load_checkpoint(model, optim, args.load)
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
        # train(model, train_loader, val_loader, train_dataset, val_dataset, optim, scoring_fn, device, start_epoch=start, 
        #       num_epochs=args.epochs, num_augmentations=args.num_augmentations, validate_interval=5, best_loss=best_loss,
        #       checkpoint_filename=args.checkpoint_path)
        print("=> Training complete!")

        print("===> Building search dataset...")
        search_data_dir = "./search/search_data"
        search_dataset  = SearchDataset(search_data_dir, model, galaxy_dataset, device, extract_features=True)
        print("=> Search dataset build complete!")
    if (args.search):
        if ((args.query_image == '') or (args.load == '')):
            print("Search requires both a query image file and a model checkpoint file!")
            return -1
        else:
            # run search
            pass

if __name__=='__main__':
    main()