import matplotlib.pyplot as plt
import numpy as np


def save_generated_images(gen_images, epoch, results_dir):
    # Rescale from [-1, 1] to [0, 1]
    gen_images = (gen_images + 1.0) / 2.0
    # From NCHW to NHWC
    gen_images = np.transpose(gen_images, (0, 2, 3, 1))
    # Calculate r using validation inputs size
    c = min(8, gen_images.shape[0])
    r = gen_images.shape[0] // c + int(bool(gen_images.shape[0] % c))
    fig, axs = plt.subplots(r, c)
    cmap = None
    if gen_images.shape[-1] == 1:
        cmap = "gray"
    cnt = 0
    for i in range(r):
        for j in range(c):
            if gen_images.shape[-1] == 1:
                axs[i, j].imshow(gen_images[cnt, :, :, 0], cmap=cmap)
            else:
                axs[i, j].imshow(gen_images[cnt, :, :, :], cmap=cmap)
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(results_dir + f"/genimages_{epoch + 1}.png")
    plt.close()