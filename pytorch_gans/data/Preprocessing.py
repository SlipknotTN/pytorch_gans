from torchvision import transforms


class Preprocessing(object):

    def __init__(self, config):

        # TODO: Test Inception preprocessing

        if config.input_channels == 3:
            # ToTensor converts HWC PIL.Image to CHW float tensor.
            self.data_transform_train = transforms.Compose([
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])
            ])

            self.data_transform_val = transforms.Compose([
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])
            ])

        else:

            self.data_transform_train = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[1.0])
            ])

            self.data_transform_val = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[1.0])
            ])

    def get_transforms_train(self):

        return self.data_transform_train

    def get_transforms_val(self):

        return self.data_transform_val
