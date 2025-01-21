import os
import sys
import pandas as pd
from dotenv import load_dotenv

from helpers import Stepper, print_header, print_function_execution, generate_filename_with_timestamp
from data_analysis import data_analysis_pipeline
from data_preprocessing import data_preprocessing_pipeline
from model_training import model_training_pipeline


def check_environment_variables(variables):
    for variable in variables:
        if not variable:
            print(f"Błąd: Brak wartości dla zmiennej {variable} w pliku .env.")
            sys.exit(1)


def load_environment_variables():
    load_dotenv()
    env_variables = {
        "data_directory": os.getenv('DATA_DIRECTORY'),
        "plots_directory": os.getenv('PLOTS_DIRECTORY'),
        "summaries_directory": os.getenv('SUMMARIES_DIRECTORY'),
        "results_directory": os.getenv('RESULTS_DIRECTORY'),
        "train_data_file": os.getenv('TRAIN_DATA_FILE'),
        "test_data_file": os.getenv('TEST_DATA_FILE'),
        "random_forest_trees": os.getenv('RANDOM_FOREST_TREES'),
        "random_forest_seed": os.getenv('RANDOM_FOREST_SEED')
    }
    check_environment_variables([env_variables[key] for key in env_variables])
    env_variables['random_forest_trees'] = int(env_variables['random_forest_trees'])
    env_variables['random_forest_seed'] = int(env_variables['random_forest_seed'])

    return env_variables


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
    env_variables = print_function_execution("Wczytywanie zmiennych środowiskowych", load_environment_variables)
    directories_paths = print_function_execution("Tworzenie katalogów", create_directories, env_variables)

    print_header(step(), "Wczytywanie danych")
    train_df, test_df = print_function_execution("Tworzenie zbiorów danych", load_data,
                                                 env_variables['data_directory'],
                                                 env_variables['train_data_file'],
                                                 env_variables['test_data_file'])

    print_header(step(), "Analiza brakujących danych")
    data_analysis_pipeline(train_df, test_df, directories_paths['summaries'], directories_paths['plots'])

    print_header(step(), "Przetworzenie danych")
    data_preprocessing_pipeline(train_df, test_df, directories_paths['data'])

    print_header(step(), "Trenowanie modelu")
    model_training_pipeline(directories_paths['results'], directories_paths['summaries'], directories_paths['plots'],
                            train_df, test_df, env_variables['random_forest_trees'],
                            env_variables['random_forest_seed'])

    print("### Przetwarzanie zakończone ###")


main()
