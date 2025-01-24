import matplotlib.pyplot as plt
import numpy as np
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from slugify import slugify
from abc import abstractmethod
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from tabulate import tabulate

from helpers import log_execution, print_file_saved


class MachineLearningAlgorithm:
    def __init__(self, train_df, test_df, label_columns, model_name):
        self.train_df = train_df
        self.test_df = test_df
        self.label_columns = label_columns

        self._model_name = model_name
        self._model = None
        self._evaluation = None
        self._train_features = None
        self._train_labels = None
        self._test_features = None
        self._test_labels = None

        self._prepare_data()

    @property
    def model(self):
        return self._model

    @property
    def evaluation(self):
        return self._evaluation

    @abstractmethod
    def train(self):
        pass

    @log_execution("Eksportowanie modelu")
    def export_model(self, save_path):
        if self._model is None:
            raise Exception("Model nie jest wytrenowany!")

        os.makedirs(save_path, exist_ok=True)

        file_path = os.path.join(save_path, f'{slugify(self._model_name)}.joblib')
        joblib.dump(self._model, file_path)

        print_file_saved(file_path)

    @log_execution("Ocena modelu")
    def evaluate(self):
        if self._model is None:
            raise Exception("Model nie jest wytrenowany!")

        prediction = self._model.predict(self._test_features)
        probability = self._model.predict_proba(self._test_features)[:, 1]
        accuracy = accuracy_score(self._test_labels, prediction)
        roc_auc = roc_auc_score(self._test_labels, probability)
        error = 1 - accuracy

        self._evaluation = {
            "prediction": prediction,
            "probability": probability,
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "error": error
        }

    @log_execution("Eksportowanie wyników oceny modelu")
    def export_evaluation(self, save_path):
        if self._evaluation is None:
            raise Exception("Model nie jest oceniony!")

        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, f'{slugify(self._model_name)}_evaluation.txt')

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"### Ocena modelu: {self._model_name}\n")
            f.write(f"Dokładność modelu: {self._evaluation['accuracy']:.4f}\n")
            f.write(f"Błąd: {self._evaluation['error']:.4f}\n\n")

            f.write("### Raport klasyfikacji:\n")
            classification_data = classification_report(
                self._test_labels, self._evaluation['prediction'], output_dict=True
            )
            headers = ["Klasa", "Precision", "Recall", "F1-Score", "Support"]
            rows = [
                [key] + [f"{value['precision']:.2f}", f"{value['recall']:.2f}", f"{value['f1-score']:.2f}",
                         int(value['support'])]
                for key, value in classification_data.items()
                if key not in ("accuracy", "macro avg", "weighted avg")
            ]
            rows.append(["accuracy", "-", "-", f"{self._evaluation['accuracy']:.4f}", len(self._test_labels)])
            rows.append(["macro avg"] + [f"{classification_data['macro avg'][metric]:.2f}" for metric in
                                         ["precision", "recall", "f1-score"]] + [len(self._test_labels)])
            rows.append(["weighted avg"] + [f"{classification_data['weighted avg'][metric]:.2f}" for metric in
                                            ["precision", "recall", "f1-score"]] + [len(self._test_labels)])

            classification_table = tabulate(rows, headers=headers, tablefmt="grid")
            f.write(classification_table)
            f.write("\n\n")

            f.write("### Opis wyników:\n")
            for key, value in classification_data.items():
                if key in ("accuracy", "macro avg", "weighted avg"):
                    continue
                f.write(f"Wyniki dla klasy {key}:\n")
                f.write(f"  - Precision: {value['precision']:.2f}\n")
                f.write(f"  - Recall: {value['recall']:.2f}\n")
                f.write(f"  - F1-Score: {value['f1-score']:.2f}\n")
                f.write(f"  - Liczba przykładów (Support): {int(value['support'])}\n\n")

            f.write(f"Dokładność (Accuracy): {self._evaluation['accuracy']:.4f}\n\n")

            f.write("Średnie wyniki:\n")
            f.write(f"  - Macro avg (Średnia arytmetyczna dla klas):\n")
            f.write(f"    - Precision: {classification_data['macro avg']['precision']:.2f}\n")
            f.write(f"    - Recall: {classification_data['macro avg']['recall']:.2f}\n")
            f.write(f"    - F1-Score: {classification_data['macro avg']['f1-score']:.2f}\n\n")

            f.write(f"  - Weighted avg (Średnia ważona z uwzględnieniem liczby przykładów):\n")
            f.write(f"    - Precision: {classification_data['weighted avg']['precision']:.2f}\n")
            f.write(f"    - Recall: {classification_data['weighted avg']['recall']:.2f}\n")
            f.write(f"    - F1-Score: {classification_data['weighted avg']['f1-score']:.2f}\n\n")

            f.write(f"ROC AUC Score: {self._evaluation['roc_auc']:.4f}\n\n")

            f.write("### Macierz konfuzji:\n")
            cm = confusion_matrix(self._test_labels, self._evaluation['prediction'])

            headers = ["", "Przewidywana klasa 0", "Przewidywana klasa 1"]
            rows = [
                ["Rzeczywista klasa 0"] + list(cm[0]),
                ["Rzeczywista klasa 1"] + list(cm[1])
            ]

            confusion_matrix_table = tabulate(rows, headers=headers, tablefmt="grid")
            f.write(confusion_matrix_table)
            f.write("\n")
            f.write("UWAGA: Wiersze odnoszą się do rzeczywistych klas, a kolumny do przewidywanych klas przez model.\n")
            f.write("Wartości w komórkach oznaczają liczbę przykładów w danej kombinacji.\n\n")

            f.write("### Wyjaśnienie metryk:\n")
            f.write(
                "- Precision: Miara precyzji, określa, jaki procent przykładów przewidzianych jako dana klasa faktycznie do niej należał.\n")
            f.write(
                "- Recall: Miara czułości, określa, jaki procent przykładów danej klasy został poprawnie zaklasyfikowany.\n")
            f.write(
                "- F1-Score: Średnia harmoniczna precyzji i czułości, równoważąca te dwie miary. Wartość F1-Score jest wysoka, gdy zarówno precyzja, jak i czułość są wysokie.\n")
            f.write("- Support: Liczba przykładów rzeczywiście należących do danej klasy w zbiorze testowym.\n")
            f.write(
                "- Accuracy: Dokładność klasyfikacji, procent poprawnie zaklasyfikowanych przykładów w całym zbiorze testowym.\n")
            f.write(
                "- Macro avg: Średnia precyzja, czułość i F1-Score dla wszystkich klas, traktowanych równoważnie (bez uwzględnienia liczebności klas).\n")
            f.write(
                "- Weighted avg: Średnia precyzja, czułość i F1-Score z uwzględnieniem liczby przykładów w każdej klasie.\n")

        print_file_saved(file_path)

    @log_execution("Eksportowanie wykresu macierzy konfuzji")
    def export_confusion_matrix_plot(self, save_path):
        if self._evaluation is None:
            raise Exception("Model nie jest oceniony!")

        os.makedirs(save_path, exist_ok=True)
        cm = confusion_matrix(self._test_labels, self._evaluation['prediction'])

        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(8, 6))
        plt.imshow(cm_normalized, interpolation='nearest', cmap="Blues")
        plt.title(f'Macierz konfuzji - {self._model_name}')
        plt.colorbar()
        plt.xlabel('Predykcja')
        plt.ylabel('Prawdziwe etykiety')

        thresh = cm_normalized.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, f"{cm[i, j]}",
                         horizontalalignment="center",
                         color="white" if cm_normalized[i, j] > thresh else "black")

        plot_path = os.path.join(save_path, f'{slugify(self._model_name)}_confusion_matrix.png')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        print_file_saved(plot_path)

    @log_execution("Przygotowanie danych")
    def _prepare_data(self):
        self._train_features = self.train_df.drop(self.label_columns, axis=1)
        self._train_labels = self.train_df[self.label_columns].values.ravel()

        self._test_features = self.test_df.drop(self.label_columns, axis=1)
        self._test_labels = self.test_df[self.label_columns].values.ravel()


class RandomForest(MachineLearningAlgorithm):
    def __init__(self, train_df, test_df, label_columns):
        super().__init__(train_df, test_df, label_columns, 'Random Forest')

    @log_execution("Trenowanie modelu", measure_time=True)
    def train(self, estimators=1000, seed=2137):
        model = RandomForestClassifier(n_estimators=estimators, random_state=seed)
        model.fit(self._train_features, self._train_labels)
        self._model = model


class LogisticRegressionModel(MachineLearningAlgorithm):
    def __init__(self, train_df, test_df, label_columns):
        super().__init__(train_df, test_df, label_columns, 'Logistic Regression')

    @log_execution("Trenowanie modelu", measure_time=True)
    def train(self, iterations=1000, seed=2137, penalty='l2', C=1.0):
        model = LogisticRegression(random_state=seed, max_iter=iterations, penalty=penalty, C=C)
        model.fit(self._train_features, self._train_labels)
        self._model = model


class GradientBoosting(MachineLearningAlgorithm):
    def __init__(self, train_df, test_df, label_columns):
        super().__init__(train_df, test_df, label_columns, 'Gradient Boosting')

    @log_execution("Trenowanie modelu", measure_time=True)
    def train(self, estimators=1000, seed=2137, learning_rate=0.1, max_depth=3):
        model = GradientBoostingClassifier(
            n_estimators=estimators,
            random_state=seed,
            learning_rate=learning_rate,
            max_depth=max_depth
        )
        model.fit(self._train_features, self._train_labels)
        self._model = model


def model_training_pipeline(results_path, summaries_path, plots_path, train_df, test_df, seed):
    label_columns = ['satisfaction']

    random_forest_estimators = 200
    logistic_regression_iterations = 1000
    gradient_boosting_estimators = 300

    random_forest = RandomForest(train_df, test_df, label_columns)
    random_forest.train(estimators=random_forest_estimators, seed=seed)
    random_forest.evaluate()
    random_forest.export_evaluation(summaries_path)
    random_forest.export_confusion_matrix_plot(plots_path)
    random_forest.export_model(results_path)

    logistic_regression = LogisticRegressionModel(train_df, test_df, label_columns)
    logistic_regression.train(iterations=logistic_regression_iterations, seed=seed)
    logistic_regression.evaluate()
    logistic_regression.export_evaluation(summaries_path)
    logistic_regression.export_confusion_matrix_plot(plots_path)
    logistic_regression.export_model(results_path)

    gradient_boosting = GradientBoosting(train_df, test_df, label_columns)
    gradient_boosting.train(estimators=gradient_boosting_estimators, seed=seed)
    gradient_boosting.evaluate()
    gradient_boosting.export_evaluation(summaries_path)
    gradient_boosting.export_confusion_matrix_plot(plots_path)
    gradient_boosting.export_model(results_path)
