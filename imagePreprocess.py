'''
A Python routine to resize and augment RGB images. The images are
taken using a smartphone with high-resolution settings. With this 
routine, the images can be resized up to pixel-level size to
as small as 1x1 pixel dimensions. The input images also are
augmented in batches to different positioning and angles, and
the resulting augmented images are resized before the augmentation
to be in small pixel dimensions. 

On top of that, the augmented images can be viewed from the
plotting graphs where the augmented images are compared side by side
with the original image. The simple interactive plotter window
can let the user traverse the augmented images back and
forth for easy viewing.

This routine can help in the production of training and testing dataset
in the CNN systems, where the input images can be considered as the
input features. 


Author: Yusnaidi Md Yusof
Email: yusnaidi.kl@utm.my
Date: 3.1.2024
Copyright RFTI@UTM.

Download the source code at: https://github.com/YusnaidiMY/image-preprocessing
'''

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


# Create interactive plotting window with 'next' and 'back' buttons 
# for easy output image navigation
class ImagePlotter:
    def __init__(self, original_image, augmented_images):
        self.original_image = original_image
        self.augmented_images = augmented_images
        self.current_index = 0

        self.fig, self.ax = plt.subplots(1, 2, figsize=(10, 5))
        self.plot_original()
        self.plot_augmented()

        img_width, img_height = original_image.size

        # Adjust the button positions
        button_width = 0.1
        button_height = 0.05
        margin = 0.02
        self.next_button_ax = plt.axes([1 - button_width - margin, 0.01, button_width, button_height])
        self.next_button = Button(self.next_button_ax, 'Next', color='lightgoldenrodyellow')
        self.next_button.on_clicked(self.next_callback)

        self.back_button_ax = plt.axes([1 - 2 * (button_width + margin), 0.01, button_width, button_height])
        self.back_button = Button(self.back_button_ax, 'Back', color='lightgoldenrodyellow')
        self.back_button.on_clicked(self.back_callback)

    def plot_original(self):
        self.ax[0].imshow(self.original_image)
        self.ax[0].set_title('Original Image')

    def plot_augmented(self):
        self.ax[1].clear()
        self.ax[1].imshow(self.augmented_images[self.current_index])
        self.ax[1].set_title(f'Augmented Image {self.current_index + 1}')
        plt.draw()

    def back_callback(self, event):
        self.current_index = (self.current_index - 1) % len(self.augmented_images)
        self.plot_augmented()

    def next_callback(self, event):
        self.current_index = (self.current_index + 1) % len(self.augmented_images)
        self.plot_augmented()

# Resize then augment the input images (features images in the context CNN)
# at the pixel-level size. The image can even be resized to 1x1 pixels dimension.        
def resize_and_augment_images(input_folder, output_folder, target_size, batch_size):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create an ImageDataGenerator with the desired preprocessing options
    datagen = ImageDataGenerator(
        rescale=1./255,  # Rescale pixel values to [0, 1]
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Get the list of all files in the input folder
    input_files = os.listdir(input_folder)

    augmented_images = []  # List to store augmented images

    for file_name in input_files:
        # Load image
        img_path = os.path.join(input_folder, file_name)
        img = load_img(img_path)

        # Resize image
        img = img.resize(target_size)

        # Convert to numpy array and rescale
        img_array = img_to_array(img) / 255.0

        # Save resized and rescaled image
        resized_file_name = f"{os.path.splitext(file_name)[0]}_{target_size[0]}x{target_size[1]}.jpg"
        output_path = os.path.join(output_folder, resized_file_name)

        # Convert NumPy array to PIL Image
        img_pil = Image.fromarray((img_array * 255).astype(np.uint8))

        # Save the PIL Image
        img_pil.save(output_path)

        # Augment images directly without using flow_from_directory
        augmented_batch = []  # List to store augmented images for the current batch

        for i in range(batch_size):
            # Apply data augmentation on the resized image
            augmented = datagen.random_transform(img_array)

            # Convert augmented NumPy array to PIL Image
            augmented_pil = Image.fromarray((augmented * 255).astype(np.uint8))

            # Save the augmented PIL Image
            augmented_file_name = f"{os.path.splitext(file_name)[0]}_{target_size[0]}x{target_size[1]}_aug_{i + 1}.jpg"
            augmented_output_path = os.path.join(output_folder, augmented_file_name)
            augmented_pil.save(augmented_output_path)

            augmented_batch.append(augmented_pil)

        augmented_images.append(augmented_batch)

    print("Image resizing and augmentation complete.")

    # Flatten the list of augmented images for plotting
    flattened_augmented_images = [augmented for batch in augmented_images for augmented in batch]

    # Notify user
    print(f"Resized and augmented images are saved in: {output_folder}")

    # Plot all the original and augmented images
    plotter = ImagePlotter(img_pil, flattened_augmented_images)
    plt.show()

    print("Image plotting complete.")

if __name__ == "__main__":
    # Input and output folders
    # Path example using Windows machine: C:\Users\username\Documents\image_input_folder
    input_folder = r"PATH_TO_YOUR_IMAGE_INPUT_FOLDER"
    output_folder = r"PATH_TO_YOUR_IMAGE_OUTPUT_FOLDER"

    # Get user input for target size
    target_size_str = input("Enter the target size (e.g., '320x320'): ")
    target_size = tuple(map(int, target_size_str.split('x')))

    # Get user input for batch size
    batch_size = int(input("Enter the batch size: "))

    # Resize and augment images
    img_path = os.path.join(input_folder, os.listdir(input_folder)[0])
    img = load_img(img_path)
    img = img.resize(target_size)
    img_array = img_to_array(img) / 255.0
    resize_and_augment_images(input_folder, output_folder, target_size, batch_size)

    print("Image resizing, augmentation, and plotting complete.")
