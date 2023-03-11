from __future__ import print_function

import PIL.Image as Image
import numpy as np
import torch
import torchvision.datasets as datasets


class ImageFolderInstance(datasets.ImageFolder):
    """Folder dataset which returns the index of the image as well
    """

    def __init__(self, root, transform=None, target_transform=None, two_crop=False):
        super(ImageFolderInstance, self).__init__(root, transform, target_transform)
        self.two_crop = two_crop

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        image = self.loader(path)
        if self.transform is not None:
            img = self.transform(image)
        else:
            img = image
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.two_crop:
            img2 = self.transform(image)
            img = torch.cat([img, img2], dim=0)

        return img, target


class CIFAR10Instance(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, two_crop=False):
        super(CIFAR10Instance, self).__init__(root, train=train, transform=transform,
                                     target_transform=target_transform, download=download)
        self.two_crop = two_crop

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if self.two_crop:
            img2 = self.transform(image)
            img = torch.cat([img, img2], dim=0)

        return img, target


class STL10Instance(datasets.STL10):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False, two_crop=False):
        super(STL10Instance, self).__init__(root, split=split, transform=transform,
                                     target_transform=target_transform, download=download)
        self.two_crop = two_crop
    

    def __getitem__(self, index):
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if self.two_crop:
            img2 = self.transform(image)
            img = torch.cat([img, img2], dim=0)

        return img, target

