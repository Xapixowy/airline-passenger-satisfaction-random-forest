import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from tabulate import tabulate

from helpers import log_execution, print_file_saved


class RandomForest:
    def __init__(self, train_df, test_df, label_columns):
        self.train_df = train_df
        self.test_df = test_df
        self.label_columns = label_columns

        self.__model = None
        self.__evaluation = None
        self.__train_features = None
        self.__train_labels = None
        self.__test_features = None
        self.__test_labels = None

        self.__prepare_data()

    @property
    def model(self):
        return self.__model

    @property
    def evaluation(self):
        return self.__evaluation

    @log_execution("Trenowanie modelu")
    def train(self, trees=100, seed=2137):
        model = RandomForestClassifier(n_estimators=trees, random_state=seed)
        model.fit(self.__train_features, self.__train_labels)
        self.__model = model

    @log_execution("Eksportowanie modelu")
    def export_model(self, save_path):
        if self.__model is None:
            raise Exception("Model not trained.")

        os.makedirs(save_path, exist_ok=True)

        file_path = os.path.join(save_path, 'random_forest.joblib')
        joblib.dump(self.__model, file_path)

        print_file_saved(file_path)

    @log_execution("Ocena modelu")
    def evaluate(self):
        if self.__model is None:
            raise Exception("Model not trained.")

        prediction = self.__model.predict(self.__test_features)
        probability = self.__model.predict_proba(self.__test_features)[:, 1]
        accuracy = accuracy_score(self.__test_labels, prediction)
        roc_auc = roc_auc_score(self.__test_labels, probability)
        error = 1 - accuracy

        self.__evaluation = {
            "prediction": prediction,
            "probability": probability,
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "error": error
        }

    @log_execution("Eksportowanie wyników oceny modelu")
    def export_evaluation(self, save_path):
        if self.__evaluation is None:
            raise Exception("Model is not evaluated.")

        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, 'random_forest_evaluation.txt')

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"### Ocena modelu: Random Forest\n")
            f.write(f"Dokładność modelu: {self.__evaluation['accuracy']:.4f}\n")
            f.write(f"Błąd: {self.__evaluation['error']:.4f}\n\n")

            f.write("### Raport klasyfikacji:\n")
            classification_data = classification_report(
                self.__test_labels, self.__evaluation['prediction'], output_dict=True
            )
            headers = ["Klasa", "Precision", "Recall", "F1-Score", "Support"]
            rows = [
                [key] + [f"{value['precision']:.2f}", f"{value['recall']:.2f}", f"{value['f1-score']:.2f}",
                         int(value['support'])]
                for key, value in classification_data.items()
                if key not in ("accuracy", "macro avg", "weighted avg")
            ]
            rows.append(["accuracy", "-", "-", f"{self.__evaluation['accuracy']:.4f}", len(self.__test_labels)])
            rows.append(["macro avg"] + [f"{classification_data['macro avg'][metric]:.2f}" for metric in
                                         ["precision", "recall", "f1-score"]] + [len(self.__test_labels)])
            rows.append(["weighted avg"] + [f"{classification_data['weighted avg'][metric]:.2f}" for metric in
                                            ["precision", "recall", "f1-score"]] + [len(self.__test_labels)])

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

            f.write(f"Dokładność (Accuracy): {self.__evaluation['accuracy']:.4f}\n\n")

            f.write("Średnie wyniki:\n")
            f.write(f"  - Macro avg (Średnia arytmetyczna dla klas):\n")
            f.write(f"    - Precision: {classification_data['macro avg']['precision']:.2f}\n")
            f.write(f"    - Recall: {classification_data['macro avg']['recall']:.2f}\n")
            f.write(f"    - F1-Score: {classification_data['macro avg']['f1-score']:.2f}\n\n")

            f.write(f"  - Weighted avg (Średnia ważona z uwzględnieniem liczby przykładów):\n")
            f.write(f"    - Precision: {classification_data['weighted avg']['precision']:.2f}\n")
            f.write(f"    - Recall: {classification_data['weighted avg']['recall']:.2f}\n")
            f.write(f"    - F1-Score: {classification_data['weighted avg']['f1-score']:.2f}\n\n")

            f.write(f"ROC AUC Score: {self.__evaluation['roc_auc']:.4f}\n\n")

            f.write("### Macierz konfuzji:\n")
            cm = confusion_matrix(self.__test_labels, self.__evaluation['prediction'])

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
        if self.__evaluation is None:
            raise Exception("Model is not evaluated.")

        os.makedirs(save_path, exist_ok=True)
        cm = confusion_matrix(self.__test_labels, self.__evaluation['prediction'])

        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(8, 6))
        plt.imshow(cm_normalized, interpolation='nearest', cmap="Blues")
        plt.title('Macierz konfuzji - Random Forest')
        plt.colorbar()
        plt.xlabel('Predykcja')
        plt.ylabel('Prawdziwe etykiety')

        thresh = cm_normalized.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, f"{cm[i, j]}",
                         horizontalalignment="center",
                         color="white" if cm_normalized[i, j] > thresh else "black")

        plot_path = os.path.join(save_path, 'random_forest_confusion_matrix.png')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        print_file_saved(plot_path)

    @log_execution("Przygotowanie danych")
    def __prepare_data(self):
        self.__train_features = self.train_df.drop(self.label_columns, axis=1)
        self.__train_labels = self.train_df[self.label_columns].values.ravel()

        self.__test_features = self.test_df.drop(self.label_columns, axis=1)
        self.__test_labels = self.test_df[self.label_columns].values.ravel()


def model_training_pipeline(results_path, summaries_path, plots_path, train_df, test_df, trees, random_seed):
    label_columns = ['satisfaction']

    model_training = RandomForest(train_df, test_df, label_columns)
    model_training.train(trees, random_seed)
    model_training.evaluate()
    model_training.export_evaluation(summaries_path)
    model_training.export_confusion_matrix_plot(plots_path)
    model_training.export_model(results_path)

    return model_training.model
