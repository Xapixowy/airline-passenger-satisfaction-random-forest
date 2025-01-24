import os
import sys
import pandas as pd
from dotenv import load_dotenv

from helpers import Stepper, print_header, generate_filename_with_timestamp, log_execution
from data_analysis import data_analysis_pipeline
from data_preprocessing import data_preprocessing_pipeline
from model_training import model_training_pipeline


def check_environment_variables(variables):
    for variable in variables:
        if not variable:
            print(f"Błąd: Brak wartości dla zmiennej {variable} w pliku .env.")
            sys.exit(1)


@log_execution("Wczytywanie zmiennych środowiskowych")
def load_environment_variables():
    load_dotenv()
    env_variables = {
        "data_directory": os.getenv('DATA_DIRECTORY'),
        "plots_directory": os.getenv('PLOTS_DIRECTORY'),
        "summaries_directory": os.getenv('SUMMARIES_DIRECTORY'),
        "results_directory": os.getenv('RESULTS_DIRECTORY'),
        "train_data_file": os.getenv('TRAIN_DATA_FILE'),
        "test_data_file": os.getenv('TEST_DATA_FILE'),
        "seed": os.getenv('SEED')
    }
    check_environment_variables([env_variables[key] for key in env_variables])
    env_variables['seed'] = int(env_variables['seed'])

    return env_variables


@log_execution("Tworzenie katalogów")
def create_directories(env_variables):
    os.makedirs(env_variables['results_directory'], exist_ok=True)
    current_results_directory_path = os.path.join(env_variables['results_directory'],
                                                  generate_filename_with_timestamp(env_variables['results_directory']))
    os.makedirs(current_results_directory_path, exist_ok=True)
    data_directory_path = os.path.join(current_results_directory_path, env_variables['data_directory'])
    os.makedirs(data_directory_path, exist_ok=True)
    plots_directory_path = os.path.join(current_results_directory_path, env_variables['plots_directory'])
    os.makedirs(plots_directory_path, exist_ok=True)
    summaries_directory_path = os.path.join(current_results_directory_path, env_variables['summaries_directory'])
    os.makedirs(summaries_directory_path, exist_ok=True)

    return {
        'results': current_results_directory_path,
        'data': data_directory_path,
        'plots': plots_directory_path,
        'summaries': summaries_directory_path
    }


@log_execution("Wczytywanie zbiorów danych")
def load_data(input_data_directory_path, train_data_file_name, test_data_file_name):
    train_data_path = f"{input_data_directory_path}/{train_data_file_name}"
    test_data_path = f"{input_data_directory_path}/{test_data_file_name}"
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

    return [train_df, test_df]


def main():
    print()
    step = Stepper(1)

    print_header(step(), "Przygotowanie projektu")
    env_variables = load_environment_variables()
    directories_paths = create_directories(env_variables)

    print_header(step(), "Wczytywanie danych")
    train_df, test_df = load_data(env_variables['data_directory'],
                                  env_variables['train_data_file'],
                                  env_variables['test_data_file'])

    print_header(step(), "Analiza danych źródłowych")
    data_analysis_pipeline(train_df,
                           test_df,
                           f"{directories_paths['summaries']}/raw_data_analysis",
                           f"{directories_paths['plots']}/raw_data_analysis")

    print_header(step(), "Przetworzenie danych")
    preprocessed_train_df, preprocessed_test_df = data_preprocessing_pipeline(train_df,
                                                                              test_df,
                                                                              directories_paths['data'])
    print_header(step(), "Analiza danych przetworzonych")
    data_analysis_pipeline(preprocessed_train_df,
                           preprocessed_test_df,
                           f"{directories_paths['summaries']}/preprocessed_data_analysis",
                           f"{directories_paths['plots']}/preprocessed_data_analysis")

    print_header(step(), "Trenowanie modelu")
    model_training_pipeline(directories_paths['results'],
                            directories_paths['summaries'],
                            directories_paths['plots'],
                            preprocessed_train_df,
                            preprocessed_test_df,
                            env_variables['seed'])

    print("### Wszystkie zadania zakończone ###")


main()
