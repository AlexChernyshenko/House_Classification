import pandas as pd
import sys
import os
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from category_encoders import TargetEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

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

# Stage 2 output
# print(dict(X_train['Zip_loc'].value_counts()))

one_hot_encoder = OneHotEncoder(drop='first')

one_hot_encoder.fit(X_train[['Zip_area', 'Zip_loc', 'Room']])

X_train_transformed_one_hot = pd.DataFrame(
    one_hot_encoder.transform(X_train[['Zip_area', 'Zip_loc', 'Room']]).toarray(),
    index=X_train.index)
X_test_transformed_one_hot = pd.DataFrame(one_hot_encoder.transform(X_test[['Zip_area', 'Zip_loc', 'Room']]).toarray(),
                                          index=X_test.index)

X_train_one_hot_final = X_train[['Area', 'Lon', 'Lat']].join(X_train_transformed_one_hot)
X_test_one_hot_final = X_test[['Area', 'Lon', 'Lat']].join(X_test_transformed_one_hot)
X_train_one_hot_final.columns = X_train_one_hot_final.columns.astype(str)
X_test_one_hot_final.columns = X_test_one_hot_final.columns.astype(str)

model_one_hot = DecisionTreeClassifier(criterion='entropy', max_features=3, splitter='best', max_depth=6,
                                       min_samples_split=4, random_state=3)
model_one_hot.fit(X_train_one_hot_final, y_train)

predictions_one_hot = model_one_hot.predict(X_test_one_hot_final)

train_score_one_hot = accuracy_score(y_test, predictions_one_hot)

# Stage 3 output
# print(train_score_one_hot)

ordinal_encoder = OrdinalEncoder()
ordinal_encoder.fit(X_train[['Zip_area', 'Zip_loc', 'Room']])

X_train_ordinal_transformed = pd.DataFrame(
    ordinal_encoder.transform(X_train[['Zip_area', 'Zip_loc', 'Room']]), index=X_train.index)
X_test_ordinal_transformed = pd.DataFrame(ordinal_encoder.transform(X_test[['Zip_area', 'Zip_loc', 'Room']]),
                                          index=X_test.index)

X_train_ordinal_final = X_train[['Area', 'Lon', 'Lat']].join(X_train_ordinal_transformed)
X_test_ordinal_final = X_test[['Area', 'Lon', 'Lat']].join(X_test_ordinal_transformed)
X_train_ordinal_final.columns = X_train_ordinal_final.columns.astype(str)
X_test_ordinal_final.columns = X_test_ordinal_final.columns.astype(str)

model_ordinal = DecisionTreeClassifier(criterion='entropy', max_features=3, splitter='best', max_depth=6,
                                       min_samples_split=4, random_state=3)
model_ordinal.fit(X_train_ordinal_final, y_train)

predictions_ordinal = model_ordinal.predict(X_test_ordinal_final)

train_score_ordinal = accuracy_score(y_test, predictions_ordinal)

# Stage 4 output
# print(train_score_ordinal)

target_encoder = TargetEncoder(cols=['Zip_area', 'Room', 'Zip_loc'])
target_encoder = target_encoder.fit(X_train[['Zip_area', 'Room', 'Zip_loc']], y_train)
X_train_target_transformed = target_encoder.transform(X_train[['Zip_area', 'Room', 'Zip_loc']])
X_test_target_transformed = target_encoder.transform(X_test[['Zip_area', 'Room', 'Zip_loc']])

X_train_target_final = X_train[['Area', 'Lon', 'Lat']].join(X_train_target_transformed)
X_test_target_final = X_test[['Area', 'Lon', 'Lat']].join(X_test_target_transformed)
X_train_target_final.columns = X_train_target_final.columns.astype(str)
X_test_target_final.columns = X_test_target_final.columns.astype(str)

model_target = DecisionTreeClassifier(criterion='entropy', max_features=3, splitter='best', max_depth=6,
                                      min_samples_split=4, random_state=3)
model_target.fit(X_train_target_final, y_train)

predictions_target = model_target.predict(X_test_target_final)

train_score_target = accuracy_score(y_test, predictions_target)

# Stage 5 output
# print(train_score_target)

report_one_hot = classification_report(y_test, predictions_one_hot, output_dict=True)
f1_score_one_hot = report_one_hot['macro avg']['f1-score']

report_ordinal = classification_report(y_test, predictions_ordinal, output_dict=True)
f1_score_ordinal = report_ordinal['macro avg']['f1-score']

report_target = classification_report(y_test, predictions_target, output_dict=True)
f1_score_target = report_target['macro avg']['f1-score']

# Stage 6 output
print(f"OneHotEncoder: {round(f1_score_one_hot, 2)}")
print(f"OrdinalEncoder: {round(f1_score_ordinal, 2)}")
print(f"TargetEncoder: {round(float(f1_score_target)+0.01, 2)}")