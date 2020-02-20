import argparse

from torch import nn, optim
from torch.utils.data.dataloader import DataLoader

from pytorch_gans.config.ConfigParams import ConfigParams
from pytorch_gans.data.Preprocessing import Preprocessing
from pytorch_gans.data.StandardDataset import StandardDataset
from pytorch_gans.model.ModelsFactory import ModelsFactory

def do_parsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_train_dir", required=True, type=str, help="Dataset train directory")
    parser.add_argument("--dataset_val_dir", required=True, type=str, help="Dataset validation directory")
    parser.add_argument("--config_file", required=True, type=str, help="Config file path")
    parser.add_argument("--model_output_dir", required=False, type=str, default="./export/model.pth",
                        help="Directory where to save G and D models")
    parser.add_argument("--single_dir_dataset", action="store_true", help="If dataset not includes classes subdirs")
    args = parser.parse_args()
    return args


def main():
    args = do_parsing()
    print(args)

    # Load config file with model, hyperparameters and preprocessing
    config = ConfigParams(args.config_file)

    # Prepare preprocessing transform pipeline
    preprocessing_transforms = Preprocessing(config)
    preprocessing_transforms_train = preprocessing_transforms.get_transforms_train()
    preprocessing_transforms_val = preprocessing_transforms.get_transforms_val()

    # Read Dataset
    dataset_train = StandardDataset(args.dataset_train_dir, preprocessing_transforms_train)
    print("Train - Classes: {0}, Samples: {1}".format(str(len(dataset_train.get_classes())), str(len(dataset_train))))
    dataset_val = StandardDataset(args.dataset_val_dir, preprocessing_transforms_val)
    print("Validation - Classes: {0}, Samples: {1}".
          format(str(len(dataset_val.get_classes())), str(len(dataset_val))))
    print("Classes " + str(dataset_train.get_classes()))

    # Load model and apply .train() and .cuda()
    G, D = ModelsFactory.create(config, len(dataset_train.get_classes()))
    print(G)
    print(D)
    G.cuda()
    G.train()
    D.cuda()
    D.train()

    # Create a PyTorch DataLoader from CatDogDataset (two of them: train + val)
    train_loader = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(dataset_val, batch_size=config.batch_size, shuffle=False, num_workers=8)

    criterion = nn.CrossEntropyLoss()
    optimizer_G = optim.Adam(G.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999))

if __name__ == "__main__":
    main()
