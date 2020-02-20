import glob
import os
from torch.utils.data import Dataset
from PIL import Image


class StandardDataset(Dataset):
    """
    Works with generic dataset where each subdirectory contain a class images
    """
    def __init__(self, root_dir, transform):
        """
        :param root_dir (string): Directory with classes subdirectories.
        :param transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        self.samples = []

        # Read file system and save num classes and tuples (image path, gt)
        self.classes = sorted(next(os.walk(self.root_dir))[1])

        for class_index, class_name in enumerate(self.classes):

            for file in sorted(glob.glob(os.path.join(self.root_dir, class_name) + "/*.jpg")):

                self.samples.append((file, class_index))

    def get_classes(self):

        return self.classes

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        image_path = sample[0]
        gt = sample[1]

        # Read as PIL image
        image = Image.open(image_path)

        # Transform image
        image = self.transform(image)

        return {'image': image, 'gt': gt, 'file': image_path}
