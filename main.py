import numpy as np
from sklearn.neural_network import MLPClassifier  # mainly all good or 1 mistake, 280s
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # all good or 2,3 mistakes, 40s
from sklearn.ensemble import RandomForestClassifier  # 2/3 mistakes, 24s
from sklearn.tree import DecisionTreeClassifier  # 50% successfully, the worst method, 80-138s, DO WYJEBANIA
import matplotlib.pyplot as plt

from classification import FaceAgeClassifier  # plik classification.py
import import_images as ii  # plik import_images.py


def main():
    resolution = 100  # don't set to more than 100, rozdzielczość zdjęć - dajemy ją na 100

    # ii.save_images(resolution)  # comment out if new pictures have been added, wczytanie obrazów

    categories = np.load('data/categories.npy')  # wczytanie tablic numpy z folderu data
    images = np.load('data/images.npy')
    test_samples = np.load('data/test_samples_img.npy')

    classifier1 = FaceAgeClassifier(DecisionTreeClassifier())  # change Class place
    # stworzony obiekt klasyfikatora na podstawie dowolnej ilości klasyfikatorów scikit-learn
    classifier1.fit(images, categories, resolution)
    classifier1.predict_and_plot(test_samples, resolution)
    # classifier1.print_metrics()
    # classifier1.fit(images, categories, resolution)
    # classifier1.learning_curve(images, categories, resolution, 8)
    return


if __name__ == '__main__':
    main()