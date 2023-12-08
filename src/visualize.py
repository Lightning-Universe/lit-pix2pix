import matplotlib.pyplot as plt
import torch

def display_image_pairs(real_images, generated_images, file_path):
    num_images = real_images.size(0)

    fig, axes = plt.subplots(num_images, 2, figsize=(8, 2 * num_images))
    axes = axes.flatten()

    for i in range(num_images):
        # Display real image on the left
        axes[2 * i].imshow(real_images[i].permute(1, 2, 0).cpu().numpy())
        axes[2 * i].axis('off')

        # Display generated image on the right
        axes[2 * i + 1].imshow(generated_images[i].permute(1, 2, 0).cpu().numpy())
        axes[2 * i + 1].axis('off')

    plt.savefig(file_path)
    plt.close()