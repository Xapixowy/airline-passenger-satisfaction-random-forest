import os
from enum import Enum
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

from helpers import print_file_saved, print_function_execution


class ColumnActionType(Enum):
    NONE = 'none'
    DELETE_UNNECESSARY = 'delete_unnecessary_columns'
    FILL_MISSING = 'fill_missing_values'
    CONVERT_CATEGORICAL = 'convert_categorical_columns'
    STANDARDIZE = 'standardize_columns'
    NORMALIZE = 'normalize_columns'


def get_list_of_columns_with_action(action_type, columns):
    return [col for col in columns if action_type in columns[col]]


def delete_unnecessary_columns(df, columns):
    df.drop(columns, axis=1, inplace=True)


def fill_missing_values(df, columns):
    columns_dict = {col: 0 for col in columns}
    df.fillna(columns_dict, inplace=True)


def convert_categorical_columns(df, columns):
    label_encoders = {col: LabelEncoder() for col in columns}
    for col in columns:
        df[col] = label_encoders[col].fit_transform(df[col])


def standardize_columns(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])


def normalize_columns(df, columns):
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])


def export_dataframe_to_csv(df, filename, output_directory_path):
    os.makedirs(output_directory_path, exist_ok=True)
    file_path = os.path.join(output_directory_path, f"{filename.lower()}_preprocessed.csv")
    df.to_csv(file_path, index=False)
    print_file_saved(file_path)


def data_preprocessing_pipeline(train_df, test_df, output_data_directory_path):
    df_cols_actions = {
        'Unnamed: 0': [ColumnActionType.DELETE_UNNECESSARY],
        'id': [ColumnActionType.DELETE_UNNECESSARY],
        'Gender': [ColumnActionType.CONVERT_CATEGORICAL],
        'Customer Type': [ColumnActionType.CONVERT_CATEGORICAL],
        'Age': [ColumnActionType.STANDARDIZE],
        'Type of Travel': [ColumnActionType.CONVERT_CATEGORICAL],
        'Class': [ColumnActionType.CONVERT_CATEGORICAL],
        'Flight Distance': [ColumnActionType.STANDARDIZE],
        'Inflight wifi service': [ColumnActionType.NORMALIZE],
        'Departure/Arrival time convenient': [ColumnActionType.NORMALIZE],
        'Ease of Online booking': [ColumnActionType.NORMALIZE],
        'Gate location': [ColumnActionType.NORMALIZE],
        'Food and drink': [ColumnActionType.NORMALIZE],
        'Online boarding': [ColumnActionType.NORMALIZE],
        'Seat comfort': [ColumnActionType.NORMALIZE],
        'Inflight entertainment': [ColumnActionType.NORMALIZE],
        'On-board service': [ColumnActionType.NORMALIZE],
        'Leg room service': [ColumnActionType.NORMALIZE],
        'Baggage handling': [ColumnActionType.NORMALIZE],
        'Checkin service': [ColumnActionType.NORMALIZE],
        'Inflight service': [ColumnActionType.NORMALIZE],
        'Cleanliness': [ColumnActionType.NORMALIZE],
        'Departure Delay in Minutes': [ColumnActionType.STANDARDIZE],
        'Arrival Delay in Minutes': [ColumnActionType.FILL_MISSING, ColumnActionType.STANDARDIZE],
        'satisfaction': [ColumnActionType.CONVERT_CATEGORICAL]
    }

    unnecessary_cols = get_list_of_columns_with_action(ColumnActionType.DELETE_UNNECESSARY, df_cols_actions)
    missing_cols = get_list_of_columns_with_action(ColumnActionType.FILL_MISSING, df_cols_actions)
    categorical_cols = get_list_of_columns_with_action(ColumnActionType.CONVERT_CATEGORICAL, df_cols_actions)
    standardize_cols = get_list_of_columns_with_action(ColumnActionType.STANDARDIZE, df_cols_actions)
    normalize_cols = get_list_of_columns_with_action(ColumnActionType.NORMALIZE, df_cols_actions)

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
        print_function_execution(f"Usuwanie zbędnych kolumn w zbiorze {dataset['name']}",
                                 delete_unnecessary_columns, dataset['dataframe'], unnecessary_cols)
        print_function_execution(f"Uzupełnianie brakujących danych w zbiorze {dataset['name']}",
                                 fill_missing_values, dataset['dataframe'], missing_cols)
        print_function_execution(f"Konwersja kategorycznych kolumn w zbiorze {dataset['name']}",
                                 convert_categorical_columns, dataset['dataframe'], categorical_cols)
        print_function_execution(f"Standardyzacja kolumn w zbiorze {dataset['name']}",
                                 standardize_columns, dataset['dataframe'], standardize_cols)
        print_function_execution(f"Normalizacja kolumn w zbiorze {dataset['name']}",
                                 normalize_columns, dataset['dataframe'], normalize_cols)
        dataset['dataframe'][:] = dataset['dataframe'].round(2)
        print_function_execution(f"Eksportowanie danych ze zbioru {dataset['name']}",
                                 export_dataframe_to_csv, dataset['dataframe'], dataset['name'],
                                 output_data_directory_path)

    return train_df, test_df
