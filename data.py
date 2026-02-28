"""
https://github.com/seulkiyeom/LRP_pruning/blob/master/modules/data.py

data.py

Purpose
-------
Dataset utilities and split logic for pruning experiments. Provides dataset constructors that
return (train_dataset, test_dataset) pairs used by PruningFineTuner.

Project-specific usage (CelebA)
-------------------------------
For CelebA pruning runs, we deliberately:
- Use ONLY the official CelebA TRAIN split as the universe to avoid leakage from official val/test.
- Create reproducible internal splits:
    * downstream_test (withheld, not touched during pruning/fine-tuning; used later)
    * ft_train (used for ranking + fine-tuning during pruning)
    * prune_val (used as “test” during pruning iterations)

These splits must be reproducible by seed and fractions so downstream evaluation can re-create
the withheld subset exactly.

Dataset wrappers
----------------
- XYOnly: wraps a dataset returning (x, y, ...) and exposes only (x, y) to match the pruning
  code which assumes (data, target) from DataLoader.

Other datasets
--------------
Also contains loaders for MNIST/CIFAR/ImageNet and various custom ImageFolder subsets used
historically in pruning experiments.

Assumptions / notes
-------------------
- ImageNet normalization is used for most transforms (mean/std hardcoded).
- Many paths are cluster-specific (/n/fs/ncp/...); adjust for new environments.
- Some older functions use random_split; ensure seeds are set for reproducibility.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fnmatch
import os
from functools import lru_cache
from pathlib import Path

import imageio
import numpy
import pandas as pd
import torch
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import random_split, DataLoader

class ImageNetDatasetValidation(torch.utils.data.Dataset):
    """ This class represents the ImageNet Validation Dataset"""

    def __init__(self, trans=None, root_dir=None):

        # validation data paths
        if root_dir is None:
            self.baseDir = '/n/fs/ncp/NCP.v2/data/images/'
        else:
            self.baseDir = root_dir
        self.validationDir = os.path.join(self.baseDir, 'validation')
        self.validationLabelsDir = os.path.join(self.validationDir, 'info.csv')
        self.validationImagesDir = os.path.join(self.validationDir, 'images')

        # read the validation labels
        self.dataInfo = pd.read_csv(self.validationLabelsDir)
        self.labels = self.dataInfo['label'].values
        self.imageNames = self.dataInfo['imageName'].values
        self.labelID = self.dataInfo['labelWNID'].values

        self.len = self.dataInfo.shape[0]

        self.transforms = trans

    # we use an lru cache in order to store the most recent
    # images that have been read, in order to minimize data access times
    @lru_cache(maxsize=128)
    def __getitem__(self, index):

        # get the filename of the image we will return
        filename = self.imageNames[index]

        # create the path to that image
        imgPath = os.path.join(self.validationImagesDir, filename)

        # load the image an an numpy array (imageio uses numpy)
        img = imageio.imread(imgPath)

        # if the image is b&w and has only one colour channel
        # create two duplicate colour channels that have the
        # same values
        if (img.ndim == 2):
            img = numpy.stack([img] * 3, axis=2)

        # convert the array to a pil image, so that we can apply transformations
        img = Image.fromarray(img)

        # apply any transformations necessary
        if self.transforms is not None:
            img = self.transforms(img)

        # get the label
        labelIdx = int(self.labels[index])

        return img, labelIdx

    def __len__(self):
        return self.len

class XYOnly(torch.utils.data.Dataset):
    """Wrap a dataset that returns (x, y, ...) and expose only (x, y)."""
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        # item could be tuple/list; assume first two are (x, y)
        return item[0], item[1]


def get_mnist(datapath='../data/mnist/', download=True):
    '''
    The MNIST dataset in PyTorch does not have a development set, and has its own format.
    We use the first 5000 examples from the training dataset as the development dataset. (the same with TensorFlow)
    Assuming 'datapath/processed/training.pt' and 'datapath/processed/test.pt' exist, if download is set to False.
    '''
    # MNIST Dataset
    train_dataset = datasets.MNIST(root=datapath,
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=download)

    test_dataset = datasets.MNIST(root=datapath,
                                  train=False,
                                  transform=transforms.ToTensor())
    return train_dataset, test_dataset


def get_cifar10(datapath='../../data/', download=True):
    '''
    Get CIFAR10 dataset
    '''
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    # Cifar-10 Dataset
    train_dataset = datasets.CIFAR10(root=datapath,
                                     train=True,
                                     transform=transforms.Compose([
                                         transforms.RandomCrop(32, padding=4),
                                         # transforms.Resize(256),
                                         # transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         normalize
                                     ]),
                                     download=download)

    test_dataset = datasets.CIFAR10(root=datapath,
                                    train=False,
                                    transform=transforms.Compose([
                                        # transforms.Resize(224),
                                        transforms.ToTensor(),
                                        normalize
                                    ]))
    return train_dataset, test_dataset


def get_imagenet(transform=None, root_dir=None):
    if root_dir is None:
        root_dir = '/ssd7/skyeom/data/imagenet'
    root_dir = Path(root_dir)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize])

    val_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize])

    # we can load the training data as an ImageFolder
    train = datasets.ImageFolder(root_dir / "train", train_transform)

    # but not the validation data
    # we use the custom made ImageNetDatasetValidation class for that
    val = ImageNetDatasetValidation(val_transform, root_dir=root_dir)

    return train, val

def get_basketball_imagenet(transform=None, root_dir=None):
    if root_dir is None:
        root_dir = '/n/fs/ncp/NCP.v2/data/images/imagenet_430_binary'
    root_dir = Path(root_dir)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize])

    # we can load the training data as an ImageFolder
    #train = datasets.ImageFolder(root_dir / "train", transform)

    # but not the validation data
    # we use the custom made ImageNetDatasetValidation class for that
    #val = ImageNetDatasetValidation(transform, root_dir=root_dir)

    dataset = datasets.ImageFolder('n/fs/ncp/NCP.v2/data/images/imagenet_430_binary', transform=transform)

    print(dataset.class_to_idx) #  should show: {'basketball': 0, 'not_basketball': 1} or vice versa
    # 80/20 split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train, test = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    
    return train, test

def get_crate_imagenet(transform=None, root_dir=None):
    if root_dir is None:
        root_dir = '/n/fs/ncp/NCP.v2/data/images/imagenet_crate_packet_prune_set_0p5_wm'
    root_dir = Path(root_dir)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize])

    # we can load the training data as an ImageFolder
    #train = datasets.ImageFolder(root_dir / "train", transform)

    # but not the validation data
    # we use the custom made ImageNetDatasetValidation class for that
    #val = ImageNetDatasetValidation(transform, root_dir=root_dir)

    dataset = datasets.ImageFolder(root_dir, transform=transform)

    print(dataset.class_to_idx) #  should show: {'non_target': 0, 'target': 1}
    # 80/20 split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train, test = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    
    return train, test

def get_carton_imagenet(transform=None, root_dir=None):
    if root_dir is None:
        root_dir = '/n/fs/ncp/NCP.v2/data/images/carton_dugong/test_set' # remember to change train size if using for pruning!!!!
    root_dir = Path(root_dir)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([#transforms.Resize(256),
                                        #transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize])

    dataset = datasets.ImageFolder(root_dir, transform=transform)

    print(dataset.class_to_idx) #  should show: {'non_target': 0, 'target': 1}
    # 80/20 split
    train_size = int(0.1 * len(dataset)) ## remember to set this back to 0.8
    test_size = len(dataset) - train_size
    train, test = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    
    return train, test

import hashlib
from pathlib import Path
from torchvision import transforms
import torch
from torch.utils.data import random_split

def _hash01(s: str) -> float:
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return int(h[:15], 16) / float(16**15)

def get_celeba_attribute_splits_for_pruning(
    attr_name="Wearing_Lipstick",
    celeba_root="/n/fs/ncp/NCP.v2/data/images/celeba",
    seed=42,
    downstream_test_frac=0.50,        # withheld for later downstream eval
    prune_val_frac_of_remaining=0.30, # used as "test" during pruning
    transform=None,
    require_exists=True,
    gender_as01=False,
):
    """
    Returns:
      ft_train_xy, prune_val_xy

    Where BOTH are drawn ONLY from the OFFICIAL CelebA TRAIN split.

    Also deterministically defines a withheld downstream_test split
    (not returned / not used in pruning), reproducible via seed + fracs.
    """
    from celeba import CelebAIndex, CelebAAttributeDataset

    if transform is None:
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    idx = CelebAIndex(str(celeba_root))

    # Universe = official TRAIN filenames only
    train_fnames = idx.filenames(split="train", require_exists=require_exists)

    downstream_f, remaining_f = [], []
    for fn in train_fnames:
        r = _hash01(f"{seed}:{fn}")
        (downstream_f if r < downstream_test_frac else remaining_f).append(fn)

    prune_val_f, ft_train_f = [], []
    for fn in remaining_f:
        r = _hash01(f"{seed+1}:{fn}")
        (prune_val_f if r < prune_val_frac_of_remaining else ft_train_f).append(fn)

    ft_train = CelebAAttributeDataset(
        idx=idx,
        split="train",
        target_attr=attr_name,
        transform=transform,
        gender_as01=gender_as01,
        filenames=ft_train_f,
    )
    prune_val = CelebAAttributeDataset(
        idx=idx,
        split="train",
        target_attr=attr_name,
        transform=transform,
        gender_as01=gender_as01,
        filenames=prune_val_f,
    )

    # Important: do NOT return downstream_test here (by design).
    # But log its size so you can sanity-check and recreate later.
    print(
        f"[CelebA pruning splits from official TRAIN] "
        f"train_total={len(train_fnames)} | "
        f"downstream_test(withheld)={len(downstream_f)} | "
        f"ft_train={len(ft_train)} | prune_val={len(prune_val)} | "
        f"seed={seed} | fracs: downstream={downstream_test_frac}, prune_val_of_remaining={prune_val_frac_of_remaining}"
    )

    # Pruning pipeline expects (x,y); drop g,fname
    ft_train = XYOnly(ft_train)
    prune_val = XYOnly(prune_val)
    return ft_train, prune_val

