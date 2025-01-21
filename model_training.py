import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from tabulate import tabulate

from helpers import print_function_execution, print_file_saved


def prepare_data(train_df, test_df, label_columns):
    train_features = train_df.drop(label_columns, axis=1)
    train_labels = train_df[label_columns].values.ravel()

    test_features = test_df.drop(label_columns, axis=1)
    test_labels = test_df[label_columns].values.ravel()

    return {
        'train_features': train_features,
        'train_labels': train_labels,
        'test_features': test_features,
        'test_labels': test_labels
    }


def train_random_forest(train_features, train_labels, n_estimators=100, random_state=42):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(train_features, train_labels)

    return model


def evaluate_model(model, test_features, test_labels):
    prediction = model.predict(test_features)
    probability = model.predict_proba(test_features)[:, 1]
    accuracy = accuracy_score(test_labels, prediction)
    roc_auc = roc_auc_score(test_labels, probability)
    error = 1 - accuracy

    return {
        "prediction": prediction,
        "probability": probability,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "error": error
    }


def export_model_evaluation(evaluation, test_labels, model_name, save_path):
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, f'{model_name.lower()}_evaluation.txt')

    with open(file_path, "w") as f:
        f.write(f"### Ocena modelu: {model_name}\n")
        f.write(f"Dokładność modelu: {evaluation['accuracy']:.4f}\n")
        f.write(f"Błąd: {evaluation['error']:.4f}\n\n")

        f.write("### Raport klasyfikacji:\n")
        classification_data = classification_report(
            test_labels, evaluation['prediction'], output_dict=True
        )
        headers = ["Klasa", "Precision", "Recall", "F1-Score", "Support"]
        rows = [
            [key] + [f"{value['precision']:.2f}", f"{value['recall']:.2f}", f"{value['f1-score']:.2f}",
                     int(value['support'])]
            for key, value in classification_data.items()
            if key not in ("accuracy", "macro avg", "weighted avg")
        ]
        rows.append(["accuracy", "-", "-", f"{evaluation['accuracy']:.4f}", len(test_labels)])
        rows.append(["macro avg"] + [f"{classification_data['macro avg'][metric]:.2f}" for metric in
                                     ["precision", "recall", "f1-score"]] + [len(test_labels)])
        rows.append(["weighted avg"] + [f"{classification_data['weighted avg'][metric]:.2f}" for metric in
                                        ["precision", "recall", "f1-score"]] + [len(test_labels)])

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

        f.write(f"Dokładność (Accuracy): {evaluation['accuracy']:.4f}\n\n")

        f.write("Średnie wyniki:\n")
        f.write(f"  - Macro avg (Średnia arytmetyczna dla klas):\n")
        f.write(f"    - Precision: {classification_data['macro avg']['precision']:.2f}\n")
        f.write(f"    - Recall: {classification_data['macro avg']['recall']:.2f}\n")
        f.write(f"    - F1-Score: {classification_data['macro avg']['f1-score']:.2f}\n\n")

        f.write(f"  - Weighted avg (Średnia ważona z uwzględnieniem liczby przykładów):\n")
        f.write(f"    - Precision: {classification_data['weighted avg']['precision']:.2f}\n")
        f.write(f"    - Recall: {classification_data['weighted avg']['recall']:.2f}\n")
        f.write(f"    - F1-Score: {classification_data['weighted avg']['f1-score']:.2f}\n\n")

        f.write(f"ROC AUC Score: {evaluation['roc_auc']:.4f}\n\n")

        f.write("### Macierz konfuzji:\n")
        cm = confusion_matrix(test_labels, evaluation['prediction'])

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


def export_confusion_matrix_plot(test_labels, predictions, model_name, save_path):
    os.makedirs(save_path, exist_ok=True)
    cm = confusion_matrix(test_labels, predictions)

    # Normalizacja macierzy do zakresu [0, 1] dla lepszej widoczności
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_normalized, interpolation='nearest', cmap="Blues")
    plt.title(f"Macierz konfuzji - {model_name.lower()}")
    plt.colorbar()
    plt.xlabel('Predykcja')
    plt.ylabel('Prawdziwe etykiety')

    # Dodanie wartości do komórek macierzy
    thresh = cm_normalized.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]}",
                     horizontalalignment="center",
                     color="white" if cm_normalized[i, j] > thresh else "black")

    # Zapisanie wykresu
    plot_path = os.path.join(save_path, f'{model_name.lower()}_confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print_file_saved(plot_path)


def export_model(path, model, model_name):
    filename = f"{model_name}.joblib"
    file_path = os.path.join(path, filename)

    os.makedirs(path, exist_ok=True)

    joblib.dump(model, file_path)

    print_file_saved(file_path)


def model_training_pipeline(results_path, summaries_path, plots_path, train_df, test_df, trees=100, random_seed=2137):
    label_columns = ['satisfaction']

    data = print_function_execution(f"Przygotowanie danych", prepare_data, train_df, test_df, label_columns)

    model = print_function_execution(f"Trenowanie modelu Random Forest", train_random_forest, data['train_features'],
                                     data['train_labels'], trees, random_seed)

    evaluation = print_function_execution(f"Ocena modelu", evaluate_model, model, data['test_features'],
                                          data['test_labels'])

    print_function_execution(f"Eksportowanie wyników ewaluacji", export_model_evaluation, evaluation,
                             data['test_labels'], 'random_forest', summaries_path)

    print_function_execution(f"Eksportowanie wykresu macierzy konfuzji", export_confusion_matrix_plot,
                             data['test_labels'], evaluation['prediction'], 'random_forest', plots_path)

    print_function_execution(f"Eksportowanie modelu", export_model, results_path, model, 'random_forest')

    return model
