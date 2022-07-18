import pandas
from sklearn.model_selection import train_test_split

FILENAME = "deposit_account.csv"
DELI = '|'
TEST_SIZE = 0.3

def get_dataset(filename):
    dataframe = pandas.read_csv(filename, sep=DELI)
    dataframe.pop('Account Number')
    dataframe.pop('Account Name')
    dataframe.pop('Customer Name')
    dataframe.pop('Currency ')
    dataframe.pop('District')
    dataframe.pop('Employer ')
    dataframe.pop('Education ')
    dataframe.pop('Marital status')
    dataframe.pop('SEX')

    dataframe['Open_Date'] = pandas.to_datetime(dataframe['Open_Date'])
    dataframe['Open_Date'] = pandas.to_numeric(dataframe['Open_Date'])
    dataframe['P1'] = pandas.to_datetime(dataframe['P1'])
    dataframe['P1'] = pandas.to_numeric(dataframe['P1'])

    P3_mapping_table = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    dataframe['P3'] = dataframe['P3'].map(P3_mapping_table)

    return dataframe


def get_x_y(dataframe):
    y = dataframe.pop('Default ').to_frame()
    X = dataframe

    return X, y


def split_training_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
    return X_train, X_test, y_train, y_test
