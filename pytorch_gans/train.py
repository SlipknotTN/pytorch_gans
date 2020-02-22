import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader

from pytorch_gans.config.ConfigParams import ConfigParams
from pytorch_gans.data.Preprocessing import Preprocessing
from pytorch_gans.data.StandardDataset import StandardDataset
from pytorch_gans.model.ModelsFactory import ModelsFactory


def save_generated_images(validation_inputs, G, epoch, results_dir):
    # Save images generated at the end of the epoch, validation like,
    gen_images_t = G(validation_inputs)
    gen_images = gen_images_t.cpu().data.numpy()
    # Rescale from -1 1 to 0 1
    gen_images = (gen_images + 1.0) / 2.0
    # From NCHW to NHWC
    gen_images = np.transpose(gen_images, (0, 2, 3, 1))
    # TODO: Calculate r using validation inputs size
    r = 4
    c = 8
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_images[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(results_dir + "/genimages_%d.png" % epoch)
    plt.close()


def do_parsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_train_dir", required=True, type=str, help="Dataset train directory")
    parser.add_argument("--dataset_val_dir", required=True, type=str, help="Dataset validation directory")
    parser.add_argument("--config_file", required=True, type=str, help="Config file path")
    parser.add_argument("--model_output_dir", required=True, type=str,
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
    classes = sorted(next(os.walk(args.dataset_train_dir))[1])
    print(f"Classes: {classes}")
    dataset_train = StandardDataset(args.dataset_train_dir, preprocessing_transforms_train)
    print("Train - Classes: {0}, Samples: {1}".format(str(len(dataset_train.get_classes())), str(len(dataset_train))))
    dataset_val = StandardDataset(args.dataset_val_dir, preprocessing_transforms_val)
    print("Validation - Classes: {0}, Samples: {1}".
          format(str(len(dataset_val.get_classes())), str(len(dataset_val))))
    print("Classes " + str(dataset_train.get_classes()))

    # Load model and apply .train() and .cuda()
    G, D = ModelsFactory.create(config, len(dataset_train.get_classes()))
    device = torch.device("cuda:0")
    print(G)
    print(D)
    G.cuda()
    G.train()
    D.cuda()
    D.train()

    results_dir = os.path.join(args.model_output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Create a PyTorch DataLoader from CatDogDataset (two of them: train + val)
    train_loader = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True, num_workers=8)
    # Validation not used typically
    val_loader = DataLoader(dataset_val, batch_size=config.batch_size, shuffle=False, num_workers=8)
    validation_inputs = torch.randn(config.batch_size, config.zdim).to(device)

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(G.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999))

    for epoch in range(config.epochs):

        G.train()
        running_d_loss = 0.0
        running_d_real_loss = 0.0
        running_d_fake_loss = 0.0
        running_g_loss = 0.0

        # Iterate on train batches and update weights using loss
        for batch_i, data in enumerate(train_loader):

            # Batch of generator images
            latents = torch.randn(len(data["image"]), config.zdim).to(device)
            g_out = G(latents)

            ############ Discriminator update over real images

            # Discriminator prediction over real images
            d_out_real = D(data["image"].to(device))

            # zero the parameter (weight) gradients for discriminator
            D.zero_grad()

            # calculate the loss between predicted and target class
            d_real_loss = criterion(d_out_real, torch.ones(size=(len(data["image"]), 1)).to(device))

            # backward pass to calculate the weight gradients
            d_real_loss.backward()

            # update D weights
            optimizer_D.step()

            ############ Discriminator update over fake images

            # Discriminator prediction over fake images (detaching generator to avoid gradients propagation)
            d_out_fake = D(g_out.detach())

            # calculate the loss between predicted and target class
            d_fake_loss = criterion(d_out_fake, torch.zeros(size=(len(data["image"]), 1)).to(device))

            # backward pass to calculate the weight gradients
            d_fake_loss.backward()

            # update D weights
            optimizer_D.step()

            ############ Generator update

            # zero the parameter (weight) gradients for generator
            G.zero_grad()

            # Discriminator over fake images another time keeping generator for gradients
            d_out_fake_with_G = D(g_out)

            # Calculate generator loss and gradients, we want discriminator output 1 for fake images
            g_loss = criterion(d_out_fake_with_G, torch.ones(size=(len(data["image"]), 1)).to(device))

            # backward pass to calculate the weight gradients
            g_loss.backward()

            # update G weights
            optimizer_G.step()

            ############ print loss statistics and output generated images

            running_d_loss += running_d_real_loss * 0.5 + running_d_fake_loss * 0.5
            running_d_real_loss += d_real_loss.item()
            running_d_fake_loss += d_fake_loss.item()
            running_g_loss += g_loss.item()
            if batch_i % 10 == 9:  # print every 10 batches
                print(f"Epoch: {epoch + 1}, "
                      f"Batch: {batch_i + 1}, "
                      f"D Avg. Loss: {running_d_loss / 10}, "
                      f"D Avg Real Loss {running_d_real_loss / 10}, "
                      f"D Avg Fake Loss {running_d_fake_loss / 10}, "
                      f"G Avg Loss {running_g_loss / 10}")
                running_d_loss = 0.0
                running_d_real_loss = 0.0
                running_d_fake_loss = 0.0
                running_g_loss = 0.0

        # eval() to disable BN train mode
        G.eval()
        save_generated_images(validation_inputs, G, epoch, results_dir)

    # Save model
    G.eval()
    torch.save(G.state_dict(), os.path.join(args.model_output_dir, "G.pth"))
    torch.save(D.state_dict(), os.path.join(args.model_output_dir, "D.pth"))

if __name__ == "__main__":
    main()
