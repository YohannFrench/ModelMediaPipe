
Person Dataset - v1 2022-07-25 9:25pm
==============================

This dataset was exported via roboflow.com on December 1, 2022 at 11:35 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

It includes 3193 images.
Person are annotated in COCO format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 416x416 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise
* Random rotation of between -15 and +15 degrees
* Random Gaussian blur of between 0 and 4.25 pixels

The following transformations were applied to the bounding boxes of each image:
* Random brigthness adjustment of between -25 and +25 percent


