import os
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

from helpers import print_function_execution, print_file_saved


def get_missing_values(data, dataset_name):
    missing_data = []
    for idx, column in enumerate(data.columns, start=1):
        missing_values = data[column].isnull().sum()
        missing_percent = (missing_values / len(data)) * 100
        missing_data.append([idx, column, missing_values, missing_percent])
    missing_df = pd.DataFrame(missing_data,
                              columns=['LP', 'Nazwa kolumny', 'Liczba brakujących wartości', '% brakujących wartości'])

    return missing_df


def export_missing_values(missing_df, dataset_name, save_path):
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, f'{dataset_name.lower()}_missing_values.txt')

    with open(file_path, "w") as f:
        f.write(f"### Brakujące dane w zbiorze {dataset_name}:\n")

        if not missing_df.empty:
            table = tabulate(
                missing_df,
                headers="keys",
                tablefmt="grid",
                showindex=False
            )
            f.write(table)
        else:
            f.write("Brak brakujących danych w zbiorze.\n")

        f.write("\n\n")

        total_missing = missing_df['Liczba brakujących wartości'].sum()
        total_percent = (total_missing / (missing_df.shape[0] * len(missing_df))) * 100

        f.write(f"### Podsumowanie dla zbioru {dataset_name}:\n")
        f.write(f"- Całkowita liczba brakujących wartości: {total_missing}\n")
        f.write(f"- Procent brakujących danych w całym zbiorze: {total_percent:.2f}%\n")

    print_file_saved(file_path)


def save_missing_values_plot(missing_df, dataset_name, save_path):
    os.makedirs(save_path, exist_ok=True)

    file_path = os.path.join(save_path, f'{dataset_name.lower()}_missing_values.png')

    plt.figure(figsize=(12, 6))
    plt.bar(missing_df['Nazwa kolumny'], missing_df['Liczba brakujących wartości'], color='skyblue')
    plt.xlabel('Nazwa kolumny')
    plt.ylabel('Liczba brakujących wartości')
    plt.title(f'Brakujące dane w zbiorze {dataset_name}')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

    print_file_saved(file_path)


def data_analysis_pipeline(train_df, test_df, summaries_directory_path, plots_directory_path):
    train_dataset_name = 'Train'
    train_missing_df = print_function_execution(f"Wykonanie analizy brakujących danych dla zbioru {train_dataset_name}",
                                                get_missing_values, train_df, train_dataset_name)
    print_function_execution(f"Zapisywanie wyników brakujących danych dla zbioru {train_dataset_name}",
                             export_missing_values, train_missing_df, train_dataset_name, summaries_directory_path)
    print_function_execution(f"Zapisywanie wykresu brakujących danych dla zbioru {train_dataset_name}",
                             save_missing_values_plot, train_missing_df, train_dataset_name, plots_directory_path)

    test_dataset_name = 'Test'
    test_missing_df = print_function_execution(f"Wykonanie analizy brakujących danych dla zbioru {test_dataset_name}",
                                               get_missing_values, test_df, test_dataset_name)
    print_function_execution(f"Zapisywanie wyników brakujących danych dla zbioru {test_dataset_name}",
                             export_missing_values, test_missing_df, test_dataset_name, summaries_directory_path)
    print_function_execution(f"Zapisywanie wykresu brakujących danych dla zbioru {test_dataset_name}",
                             save_missing_values_plot, test_missing_df, test_dataset_name, plots_directory_path)
