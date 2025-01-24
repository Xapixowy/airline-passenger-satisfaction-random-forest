import os
from enum import Enum
from slugify import slugify
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

from helpers import log_execution, print_file_saved


class ColumnActionType(Enum):
    NONE = 'none'
    DELETE_UNNECESSARY = 'delete_unnecessary_columns'
    FILL_MISSING = 'fill_missing_values'
    CONVERT_CATEGORICAL = 'convert_categorical_columns'
    STANDARDIZE = 'standardize_columns'
    NORMALIZE = 'normalize_columns'


class DataPreprocessing:
    def __init__(self, name, dataframe, column_actions):
        self.name = name
        self.dataframe = dataframe
        self.column_actions = column_actions

    @log_execution("Usuwanie niepotrzebnych kolumn")
    def delete_unnecessary_columns(self):
        columns = self.__get_list_of_columns_with_action(ColumnActionType.DELETE_UNNECESSARY)
        self.dataframe = self.dataframe.drop(columns, axis=1)

    @log_execution("Uzupełnianie brakujących wartości")
    def fill_missing_values(self, fill_value=None):
        columns = self.__get_list_of_columns_with_action(ColumnActionType.FILL_MISSING)
        for col in columns:
            column_fill_value = self.dataframe[col].median() if fill_value is None else fill_value
            self.dataframe[col] = self.dataframe[col].fillna(column_fill_value)

    @log_execution("Konwertowanie kolumn kategorycznych")
    def convert_categorical_columns(self):
        columns = self.__get_list_of_columns_with_action(ColumnActionType.CONVERT_CATEGORICAL)
        label_encoders = {col: LabelEncoder() for col in columns}
        for col in columns:
            self.dataframe[col] = label_encoders[col].fit_transform(self.dataframe[col])

    @log_execution("Standardyzacja kolumn")
    def standardize_columns(self):
        columns = self.__get_list_of_columns_with_action(ColumnActionType.STANDARDIZE)
        scalers = {col: StandardScaler() for col in columns}
        for col in columns:
            self.dataframe[col] = scalers[col].fit_transform(self.dataframe[[col]])

    @log_execution("Normalizacja kolumn")
    def normalize_columns(self):
        columns = self.__get_list_of_columns_with_action(ColumnActionType.NORMALIZE)
        scalers = {col: MinMaxScaler() for col in columns}
        for col in columns:
            self.dataframe[col] = scalers[col].fit_transform(self.dataframe[[col]])

    @log_execution("Zaokrąglanie wartości")
    def round_values(self, precision=2):
        self.dataframe[:] = self.dataframe.round(precision)

    @log_execution("Eksportowanie zbioru danych do CSV")
    def export_dataframe_to_csv(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, f"{slugify(self.name)}_preprocessed.csv")
        self.dataframe.to_csv(file_path, index=False)
        print_file_saved(file_path)

    def __get_list_of_columns_with_action(self, action_type):
        return [col for col in self.column_actions if action_type in self.column_actions[col]]


def data_preprocessing_pipeline(train_df, test_df, data_directory_path):
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
        data_preprocessing = DataPreprocessing(dataset['name'], dataset['dataframe'], df_cols_actions)
        data_preprocessing.delete_unnecessary_columns()
        data_preprocessing.fill_missing_values()
        data_preprocessing.convert_categorical_columns()
        data_preprocessing.standardize_columns()
        data_preprocessing.normalize_columns()
        data_preprocessing.round_values()
        data_preprocessing.export_dataframe_to_csv(data_directory_path)
        dataset['dataframe'] = data_preprocessing.dataframe

    return map(lambda dataset: dataset['dataframe'], datasets)
