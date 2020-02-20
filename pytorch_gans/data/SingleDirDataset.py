import glob
from torch.utils.data import Dataset
from PIL import Image


class SingleDirDataset(Dataset):
    """
    Works with dataset that contains all the images in the same directory and no ground truth is provided
    """
    def __init__(self, root_dir, transform):
        """
        :param root_dir (string): Directory containing images
        :param transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        self.samples = []

        # Read file system and save image paths

        for file in sorted(glob.glob(self.root_dir + "/*.jpg")):

            self.samples.append(file)

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        image_path = self.samples[idx]

        # Read as PIL image
        image = Image.open(image_path)

        # Transform image
        image = self.transform(image)

        return {'image': image, 'file': image_path}
