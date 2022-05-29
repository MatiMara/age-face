import time as t

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_ms


class FaceAgeClassifier:
    """Banana ripeness classification based on the photos."""
    def __init__(self, *classifier):  # konstruktor(argumenty)
        """Constructor initializes classifier object."""
        self.classifier = classifier  # self to obiekt, self.classifier to atrybut obiektu
        self.learning_time = 0.0
        self.predicting_time = 0.0
        self.images = np.array([])
        self.categories = np.array([])
        self.resolution = np.array([])
        self.reshaped_images = np.array([])
        self.predictions = np.array([])
        self.reshaped_test_samples = np.array([])
        self.test_samples = np.array([])
        self.y_true = np.array(['senior', 'adult', 'baby', 'adult', 'senior', 'senior', 'adult', 'baby',
                                'baby', 'adult', 'adult', 'senior', 'baby', 'baby', 'senior'])  # change later całość!
        self.y_pred = np.array([])  # to też
        return

    def fit(self, images: np.ndarray, categories: np.ndarray, resolution: int):  # learning function
        """Learn classifier to classify bananas based on photos."""
        self.images = images
        self.categories = categories
        self.resolution = resolution
        self.reshaped_images = self.images.reshape(len(self.images), 3 * self.resolution ** 2)
        # reshape zmienia kształt tablic, wypłaszcza je

        start_time = t.time()

        for clf in self.classifier:
            clf.fit(self.reshaped_images, self.categories)
            # wykonaj metode fit z biblioteki sklearn dla każdego przekazanego obiektu

        end_time = t.time()
        self.learning_time = end_time - start_time
        return

    def predict(self, test_samples: np.ndarray, resolution: int):
        """Predict if bananaXD is baby, adult or senior(black)."""
        np.random.shuffle(test_samples)  # pomieszaj foty
        self.test_samples = test_samples
        self.reshaped_test_samples = test_samples.reshape(len(test_samples), 3 * resolution ** 2)
        for i, _ in enumerate(self.reshaped_test_samples):
            all_probas = [None] * len(self.classifier)

            start_time = t.time()  # zjebane, czas dać wyżej

            for y, _ in enumerate(self.classifier):
                all_probas[y] = self.classifier[y].predict_proba(self.reshaped_test_samples[i].reshape(1, -1))
                # zwraca prawdopodobieństwo bycia starym dziadem, albo baby itd.

            end_time = t.time()
            self.predicting_time = end_time - start_time

            avg = (sum(all_probas) / len(all_probas))[0]  # getting array out of array XD 0 bo warning
            # obliczanie średniej z kilku klacyfikatorów
            if avg[0] == max(avg):  # dodawanie etykiet na podstawie klasyfikacji
                self.predictions = np.append(self.predictions, 'adult')
            elif avg[1] == max(avg):
                self.predictions = np.append(self.predictions, 'baby')
            elif avg[2] == max(avg):
                self.predictions = np.append(self.predictions, 'senior')
        return self.predictions  # funkcja zwraca etykiety

    def plot(self, title=""):
        """Display categorized photos of the bananas."""
        n_rows = 3  # ilość wierszy
        n_cols = 5  # ilość kolumn
        _, axes = plt.subplots(n_rows, n_cols)
        for i in range(n_rows):  # wyświetlanie picturesów
            for j in range(n_cols):
                samples_index = i * n_cols + j
                axes[i][j].imshow(self.test_samples[samples_index])
                axes[i][j].axis('off')  #pretty wyświetlanie
                axes[i][j].set_title(self.predictions[samples_index])
        if title:  # wyświetlanie tytułu figury
            plt.suptitle(title)
        else:  # jak brak tytułu wyświetl nazwy klas przekazanych klasyfikatorów sklearn
            classifiers_str = ""
            for obj in self.classifier:
                if classifiers_str:
                    classifiers_str = f"{classifiers_str}, {obj.__str__()}"
                else:
                    classifiers_str = obj.__str__()
            plt.suptitle(f"{classifiers_str}\n "
                         f"Learning time: {round(self.learning_time, 3)} [s]\n"
                         f"Predicting time: {round(1000 * self.predicting_time, 3)} [ms]")
        plt.show()
        return

    def predict_and_plot(self, test_samples: np.ndarray, resolution: int, title=""):
        """Run predict and plot functions one after another."""
        predictions = self.predict(test_samples, resolution)
        self.plot(title)
        return predictions

    def print_metrics(self): # wyświetl wskaźnik jakości, działa dla bananów Ignasia
        self.y_pred = self.predictions
        print(f"Accuracy: {round(sk_metrics.accuracy_score(self.y_true, self.y_pred), 3)}")
        print(f"F1 score: {round(sk_metrics.f1_score(self.y_true, self.y_pred, average='weighted'), 3)}")
        print(f"Precision: {round(sk_metrics.precision_score(self.y_true, self.y_pred, average='weighted'), 3)}")
        # micro/macro/weighted
        print(f"Recall: {round(sk_metrics.recall_score(self.y_true, self.y_pred, average='weighted'), 3)}")
        print(f"Confusion matrix:\n{sk_metrics.confusion_matrix(self.y_true, self.y_pred)}")
        return

    def learning_curve(self, images: np.ndarray, categories: np.ndarray, resolution: int, n_points: int):
        # krzywa uczenia, działa długo ale to nie jest konieczne IMO
        reshaped_images = self.images.reshape(len(images), 3 * resolution ** 2)
        train_sizes, train_scores, test_scores = {}, {}, {}
        for clf in self.classifier:
            train_sizes[clf], train_scores[clf], test_scores[clf] = sk_ms.learning_curve(
                clf,
                reshaped_images,
                categories,
                train_sizes=np.linspace(0.1, 1.0, n_points)
            )
        avg_train_sizes = sum(train_sizes.values()) / len(train_sizes)
        avg_train_scores = sum(train_scores.values()) / len(train_scores)
        avg_test_scores = sum(test_scores.values()) / len(test_scores)

        a_train_s_array = np.empty(len(avg_train_scores), dtype=float)
        for i, a_train_s in enumerate(avg_train_scores):
            a_train_s_array[i] = sum(a_train_s) / len(a_train_s)

        plt.figure()
        plt.plot(avg_train_sizes, a_train_s_array)
        # plt.figure()
        # plt.plot(avg_train_sizes, avg_test_scores)
        plt.show()
        return

    def predict_that_photo(self, path: str):  # only idea narazie
        pass  # nic nie robi XD