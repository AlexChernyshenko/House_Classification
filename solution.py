import pandas as pd
import sys
import os
import requests
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import OrdinalEncoder
from category_encoders import TargetEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    if not os.path.exists('../Data'):
        os.mkdir('../Data')

if 'house_class.csv' not in os.listdir('../Data'):
    sys.stderr.write("[INFO] Dataset is loading.\n")
    url = "https://www.dropbox.com/s/7vjkrlggmvr5bc1/house_class.scv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/house_class.csv', 'wb').write(r.content)
    sys.stderr.write("[INFO] Loaded. \n")

pd.set_option('future.no_silent_downcasting', True)
df = pd.read_csv('../Data/house_class.csv')

X, y = df.iloc[:, 1:7], df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1, stratify=X['Zip_loc'].values)

# print(dict(X_train['Zip_loc'].value_counts()))

# enc = OneHotEncoder(drop='first')
# enc = OrdinalEncoder()
enc = TargetEncoder(cols=['Zip_area', 'Room', 'Zip_loc'])

enc.fit(X_train[[ 'Zip_area', 'Room', 'Zip_loc']], y_train)

# X_train_transformed = pd.DataFrame(enc.transform(X_train[['Zip_area', 'Zip_loc', 'Room']]).toarray(),
#                                    index=X_train.index)
# X_test_transformed = pd.DataFrame(enc.transform(X_test[['Zip_area', 'Zip_loc', 'Room']]).toarray(),
# index=X_test.index)
X_train_transformed = enc.transform(X_train[['Zip_area','Room', 'Zip_loc']])
X_test_transformed = enc.transform(X_test[['Zip_area','Room', 'Zip_loc']])


X_train_final = X_train[['Area', 'Lon', 'Lat']].join(X_train_transformed)
X_test_final = X_test[['Area', 'Lon', 'Lat']].join(X_test_transformed)
X_train_final.columns = X_train_final.columns.astype(str)
X_test_final.columns = X_test_final.columns.astype(str)

clf = DecisionTreeClassifier(criterion='entropy', max_features=3, splitter='best', max_depth=6, min_samples_split=4,
                             random_state=3)

clf = clf.fit(X_train_final, y_train)

prediction_train = clf.predict(X_train_final)
prediction_test = clf.predict(X_test_final)

train_score = accuracy_score(y_train, prediction_train)
test_score = accuracy_score(y_test, prediction_test)

print(test_score)
# print(train_score)
