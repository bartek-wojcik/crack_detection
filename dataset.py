from os import listdir
from os.path import join, isfile

import cv2
from torch.utils.data import Dataset


class CrackDataset(Dataset):
    def __init__(self, root, transform=None, label=0):
        self.image_paths = [join(root, f) for f in listdir(root) if isfile(join(root, f))]
        self.transform = transform
        self.label = label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (32, 32))

        image = cv2.GaussianBlur(image, (9, 9), 0)

        image = cv2.adaptiveThreshold(
            image,
            maxValue=255.0,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=9,
            C=9
        )

        if self.transform is not None:
            image = self.transform(image)

        return image, self.label
