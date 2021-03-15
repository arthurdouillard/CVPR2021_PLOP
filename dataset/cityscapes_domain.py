import copy
import glob
import os

import numpy as np
import torch.utils.data as data
import torchvision as tv
from PIL import Image
from torch import distributed

from .utils import Subset, group_images


# Converting the id to the train_id. Many objects have a train id at
# 255 (unknown / ignored).
# See there for more information:
# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
id_to_trainid = {
    0: 255,
    1: 255,
    2: 255,
    3: 255,
    4: 255,
    5: 255,
    6: 255,
    7: 0,   # road
    8: 1,   # sidewalk
    9: 255,
    10: 255,
    11: 2,  # building
    12: 3,  # wall
    13: 4,  # fence
    14: 255,
    15: 255,
    16: 255,
    17: 5,  # pole
    18: 255,
    19: 6,  # traffic light
    20: 7,  # traffic sign
    21: 8,  # vegetation
    22: 9,  # terrain
    23: 10, # sky
    24: 11, # person
    25: 12, # rider
    26: 13, # car
    27: 14, # truck
    28: 15, # bus
    29: 255,
    30: 255,
    31: 16, # train
    32: 17, # motorcycle
    33: 18, # bicycle
    -1: 255
}

city_to_id = {
    "aachen": 0, "bremen": 1, "darmstadt": 2, "erfurt": 3, "hanover": 4,
    "krefeld": 5, "strasbourg": 6, "tubingen": 7, "weimar": 8, "bochum": 9,
    "cologne": 10, "dusseldorf": 11, "hamburg": 12, "jena": 13,
    "monchengladbach": 14, "stuttgart": 15, "ulm": 16, "zurich": 17,
    "frankfurt": 18, "lindau": 19, "munster": 20
}


def filter_images(dataset, labels):
    # Filter images without any label in LABELS (using labels not reordered)
    idxs = []

    print(f"Filtering images...")
    for i in range(len(dataset)):
        domain_id = dataset.__getitem__(i, get_domain=True)  # taking domain id
        if domain_id in labels:
            idxs.append(i)
        if i % 1000 == 0:
            print(f"\t{i}/{len(dataset)} ...")
    return idxs


class CityscapesSegmentationDomain(data.Dataset):

    def __init__(self, root, train=True, transform=None, domain_transform=None):
        root = os.path.expanduser(root)
        annotation_folder = os.path.join(root, 'gtFine')
        image_folder = os.path.join(root, 'leftImg8bit')

        self.images = [  # Add train cities
            (
                path,
                os.path.join(
                    annotation_folder,
                    "train",
                    path.split("/")[-2],
                    path.split("/")[-1][:-15] + "gtFine_labelIds.png"
                ),
                city_to_id[path.split("/")[-2]]
            ) for path in sorted(glob.glob(os.path.join(image_folder, "train/*/*.png")))
        ]
        self.images += [  # Add validation cities
            (
                path,
                os.path.join(
                    annotation_folder,
                    "val",
                    path.split("/")[-2],
                    path.split("/")[-1][:-15] + "gtFine_labelIds.png"
                ),
                city_to_id[path.split("/")[-2]]
            ) for path in sorted(glob.glob(os.path.join(image_folder, "val/*/*.png")))
        ]

        self.transform = transform
        self.domain_transform = domain_transform

    def __getitem__(self, index, get_domain=False):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        if get_domain:
            domain = self.images[index][2]
            if self.domain_transform is not None:
                domain = self.domain_transform(domain)
            return domain

        try:
            img = Image.open(self.images[index][0]).convert('RGB')
            target = Image.open(self.images[index][1])
        except Exception as e:
            raise Exception(f"Index: {index}, len: {len(self)}, message: {str(e)}")

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.images)


class CityscapesSegmentationIncrementalDomain(data.Dataset):
    """Labels correspond to domains not classes in this case."""
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        labels=None,
        idxs_path=None,
        masking=True,
        overlap=True,
        **kwargs
    ):
        full_data = CityscapesSegmentationDomain(root, train)

        # take index of images with at least one class in labels and all classes in labels+labels_old+[255]
        if idxs_path is not None and os.path.exists(idxs_path):
            idxs = np.load(idxs_path).tolist()
        else:
            idxs = filter_images(full_data, labels)
            if idxs_path is not None and distributed.get_rank() == 0:
                np.save(idxs_path, np.array(idxs, dtype=int))

        rnd = np.random.RandomState(1)
        rnd.shuffle(idxs)
        train_len = int(0.8 * len(idxs))
        if train:
            idxs = idxs[:train_len]
            print(f"{len(idxs)} images for train")
        else:
            idxs = idxs[train_len:]
            print(f"{len(idxs)} images for val")

        target_transform = tv.transforms.Lambda(
            lambda t: t.
            apply_(lambda x: id_to_trainid.get(x, 255))
        )
        # make the subset of the dataset
        self.dataset = Subset(full_data, idxs, transform, target_transform)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """

        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
