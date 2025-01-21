import os
import pandas as pd
import matplotlib.pyplot as plt
from slugify import slugify
from tabulate import tabulate

from helpers import log_execution, print_file_saved


class DataAnalysis:
    def __init__(self, name, dataframe):
        self.name = name
        self.df_original = dataframe
        self.df_missing_values = self.__get_missing_values()

    @log_execution("Eksportowanie raportu brakujących danych")
    def export_summary(self, save_path):
        os.makedirs(save_path, exist_ok=True)

        file_path = os.path.join(save_path, f'{slugify(self.name)}_missing_values.txt')

        with open(file_path, "w", encoding="utf-8") as f:
            rows_count = self.df_original.shape[0]
            columns_count = self.df_original.shape[1]
            numeric_columns_count = self.df_original.select_dtypes(include=['int', 'float']).shape[1]
            categorical_columns_count = self.df_original.select_dtypes(include=['object']).shape[1]
            numeric_data_count = self.df_original.select_dtypes(include=['int', 'float']).count().sum()
            categorical_data_count = self.df_original.select_dtypes(include=['object']).count().sum()
            numeric_data_percent = (numeric_data_count / (rows_count * columns_count)) * 100
            categorical_data_percent = (categorical_data_count / (rows_count * columns_count)) * 100
            summary_data = [
                ["Całkowita liczba wierszy", rows_count],
                ["Całkowita liczba kolumn", columns_count],
                ["Liczba kolumn z danymi liczbowymi", numeric_columns_count],
                ["Liczba kolumn z danymi kategorycznymi", categorical_columns_count],
                ["Liczba wartości liczbowych", numeric_data_count],
                ["Liczba wartości kategorycznych", categorical_data_count],
                ["Procent wartości liczbowych", f"{numeric_data_percent:.4f}%"],
                ["Procent wartości kategorycznych", f"{categorical_data_percent:.4f}%"],
            ]

            f.write(f"### Dane w zbiorze {self.name}:\n")
            tabulated_summary = tabulate(summary_data, headers=["Opis", "Wartość"], tablefmt="grid")
            f.write(tabulated_summary)
            f.write("\n\n")

            f.write("### Brakujące dane w zbiorze:\n")

            if not self.df_missing_values.empty:
                table = tabulate(
                    self.df_missing_values,
                    headers="keys",
                    tablefmt="grid",
                    showindex=False
                )
                f.write(table)
            else:
                f.write("Brak brakujących danych w zbiorze.\n")

            f.write("\n\n")

            total_missing = self.df_missing_values['Liczba brakujących wartości'].sum()
            total_percent = (total_missing / (self.df_original.shape[0] * self.df_original.shape[1])) * 100
            missing_summary_data = [
                ["Całkowita liczba brakujących wartości", total_missing],
                ["Procent brakujących danych w całym zbiorze", f"{total_percent:.4f}%"],
            ]

            f.write(f"### Podsumowanie dla zbioru {self.name}:\n")
            tabulated_missing_summary = tabulate(missing_summary_data, headers=["Opis", "Wartość"], tablefmt="grid")
            f.write(tabulated_missing_summary)

        print_file_saved(file_path)

    @log_execution("Eksportowanie wykresu brakujących danych")
    def export_missing_values_plot(self, save_path):
        os.makedirs(save_path, exist_ok=True)

        file_path = os.path.join(save_path, f'{slugify(self.name)}_missing_values.png')

        plt.figure(figsize=(12, 6))
        plt.bar(self.df_missing_values['Nazwa kolumny'], self.df_missing_values['Liczba brakujących wartości'],
                color='skyblue')
        plt.xlabel('Nazwa kolumny')
        plt.ylabel('Liczba brakujących wartości')
        plt.title(f'Brakujące dane w zbiorze {self.name}')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

        print_file_saved(file_path)

    @log_execution("Eksportowanie histogramów kolumn")
    def export_histograms(self, save_path):
        histograms_save_path = f"{save_path}/{slugify(self.name)}_histograms"
        os.makedirs(histograms_save_path, exist_ok=True)

        for column in self.df_original.columns:
            if pd.api.types.is_numeric_dtype(self.df_original[column]):
                plt.figure(figsize=(10, 6))
                self.df_original[column].dropna().hist(bins=30, color='skyblue', edgecolor='black')
                plt.title(f'Histogram: {column} (zbiór {self.name})')
                plt.xlabel(column)
                plt.ylabel('Liczność')
                plt.grid(axis='y', alpha=0.75)

                file_path = os.path.join(histograms_save_path, f'{slugify(self.name)}_histogram_{slugify(column)}.png')
                plt.savefig(file_path)
                plt.close()

                print_file_saved(file_path)

    @log_execution("Eksportowanie wykresów kołowych")
    def export_pie_charts(self, save_path):
        pie_charts_save_path = f"{save_path}/{slugify(self.name)}_pie_charts"
        os.makedirs(pie_charts_save_path, exist_ok=True)

        for column in self.df_original.columns:
            if not pd.api.types.is_numeric_dtype(self.df_original[column]):
                plt.figure(figsize=(10, 6))
                value_counts = self.df_original[column].value_counts()

                value_counts.plot.pie(
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=plt.cm.Paired.colors,
                    wedgeprops={'edgecolor': 'black'}
                )
                plt.title(f'Diagram kołowy: {column} (zbiór {self.name})')
                plt.ylabel('')

                file_path = os.path.join(pie_charts_save_path, f'{self.name.lower()}_pie_chart_{slugify(column)}.png')
                plt.savefig(file_path)
                plt.close()

                print_file_saved(file_path)

    @log_execution("Wyliczanie brakujących danych")
    def __get_missing_values(self):
        missing_data = []

        for idx, column in enumerate(self.df_original.columns, start=1):
            missing_values = self.df_original[column].isnull().sum()
            missing_percent = (missing_values / len(self.df_original)) * 100
            missing_data.append([idx, column, missing_values, missing_percent])

        missing_df = pd.DataFrame(missing_data,
                                  columns=['LP', 'Nazwa kolumny', 'Liczba brakujących wartości',
                                           '% brakujących wartości'])

        return missing_df


def data_analysis_pipeline(train_df, test_df, summaries_directory_path, plots_directory_path):
    datasets = [
        {
            'name': 'Train',
            'dataframe': train_df
        },
        {
            'name': 'Test',
            'dataframe': test_df
        }
    ]

    for dataset in datasets:
        data_analysis = DataAnalysis(dataset['name'], dataset['dataframe'])
        data_analysis.export_summary(summaries_directory_path)
        data_analysis.export_missing_values_plot(plots_directory_path)
        data_analysis.export_histograms(plots_directory_path)
        data_analysis.export_pie_charts(plots_directory_path)
