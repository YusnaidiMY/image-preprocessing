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

Sample outputs:
<img width="748" alt="output-4x4" src="https://github.com/YusnaidiMY/image-preprocessing/assets/8178236/19d4dfbb-19af-4e20-bf52-aa979eb78992">

<img width="747" alt="output-320x320" src="https://github.com/YusnaidiMY/image-preprocessing/assets/8178236/95c5b582-e5f8-40f6-a192-903d0499e04c">
