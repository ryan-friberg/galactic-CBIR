### GalaxIDNet: A custom transformer-based pipeline for galactic content-based image retrieval

Given a query image of a galaxy, this pipeline is able to retreivel the top-k most similar images belonging to the dataset. This repository houses a custom dataset using data from the [Galaxy10 Dataset](https://astronn.readthedocs.io/en/latest/galaxy10.html), a custom from-scratch transformer neural network, and a second, learned dataset for image search.

The main.py file allow for CLI training, testing, and/or searching as well as specification for various hyper-parameters, please see main.py for details or run with "--help"

Examples:
```rb
python3 main.py --force_download --train
python3 main.py --search --query_image [path-to-image] --search_database [path-to-search-parent-dir]
```

PLEASE NOTE: The python package astroNN seems to be unsupported for Apple silicon silicon computers. If this affects you, please download the .h5 file from the website manually and pass the file path to main.py via --h5_file to build the dataset instead.

Dependencies:
- torch
- numpy
- torchvision
- tqdm
- h5py or astroNN
- PIL
