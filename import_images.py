import os

import matplotlib.image as image
import numpy as np
from skimage.transform import resize


def import_directory(resolution, path: str):
    """Importing and downsizing images to 100x100 px"""
    dirlist = os.listdir(path)
    images_array = np.empty((len(dirlist), resolution, resolution, 3), dtype=float)
    for i, im_name in enumerate(dirlist):
        print(i, f'. {path}')
        img = image.imread(f'{path}/{im_name}')
        images_array[i] = resize(img, (resolution, resolution), anti_aliasing=True)
    return images_array


def save_images(resolution):
    """Save images and categories to .npy files"""
    baby_images = import_directory(resolution, 'pictures/learning/baby')
    adult_images = import_directory(resolution, 'pictures/learning/adult')
    senior_images = import_directory(resolution, 'pictures/learning/senior')
    test_samples_img = import_directory(resolution, './pictures/test')

    images = np.concatenate((baby_images, adult_images, senior_images))

    cat_baby = np.full(len(baby_images), 'baby')
    cat_adult = np.full(len(adult_images), 'adult')
    cat_senior = np.full(len(senior_images), 'senior')
    categories = np.concatenate((cat_baby, cat_adult, cat_senior))

    np.save('data/categories', categories)
    np.save('data/images', images)
    np.save('data/test_samples_img', test_samples_img)
    return [categories, images, test_samples_img]
