import os
import requests
import sys
import pandas as pd
from sklearn.model_selection import train_test_split


# Task 1
# Importing data
if __name__ == '__main__':
    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Download data if it is unavailable.
    if 'house_class.csv' not in os.listdir('../Data'):
        sys.stderr.write("[INFO] Dataset is loading.\n")
        url = "https://www.dropbox.com/s/7vjkrlggmvr5bc1/house_class.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/house_class.csv', 'wb').write(r.content)
        sys.stderr.write("[INFO] Loaded.\n")

    # reading data from a .csv file
    df = pd.read_csv('../Data/house_class.csv')
    # printing some data from the dataset
    # print(f'''{df.shape[0]}
    # {df.shape[1]}
    # {df.isna().any().sum() != 0}
    # {df.Room.max()}
    # {df.Area.mean()}
    # {df.Zip_loc.nunique()}
    # ''')

    # Task 2
    # Spliting the dataset into train and test sets
    X, y = df.iloc[:, 1:7], df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1,
                                                        stratify=X['Zip_loc'].values)
    print(dict(X_train["Zip_loc"].value_counts()))
