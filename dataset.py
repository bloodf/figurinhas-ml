from torch.utils.data import Dataset
from PIL import Image
import PIL.ImageOps
import random

"""## Custom Dataset Class
This dataset generates a pair of images. 0 for geniune pair and 1 for imposter pair
"""


class SiameseNetworkDataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):

        # Get an image
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        # Get an image from the same class
        while True:
            # keep looping till the same class image is found
            img1_tuple = random.choice(self.imageFolderDataset.imgs)
            if img0_tuple[1] == img1_tuple[1]:
                break

        # Get an image from a different class
        while True:
            # keep looping till a different class image is found

            img2_tuple = random.choice(self.imageFolderDataset.imgs)
            if img0_tuple[1] != img2_tuple[1]:
                break

        width, height = (244, 244)

        pathList = [(img0_tuple[0], img1_tuple[0], img2_tuple[0])]

        img0 = Image.open(img0_tuple[0]).resize((width, height))
        img1 = Image.open(img1_tuple[0]).resize((width, height))
        img2 = Image.open(img2_tuple[0]).resize((width, height))

        img0 = img0.convert("L")
        img1 = img1.convert("L")
        img2 = img2.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)
            img2 = PIL.ImageOps.invert(img2)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # anchor, positive image, negative image
        return img0, img1, img2, pathList

    def __len__(self):
        return len(self.imageFolderDataset.imgs)
