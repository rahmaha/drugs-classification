import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier

#data preparation
data_path = '/mnt/d/github/drugs-classification/data/drug200.csv'
df = pd.read_csv(data_path)

#features
numerical = ['Age', 'Na_to_K']
categorical:  ['Sex', 'BP', 'Cholesterol']

#split dataset
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

df_full_train = df_full_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = (df_full_train.Drug.values)
y_test = (df_test.Drug.values)

del df_full_train['Drug']
del df_test['Drug']

#model

#train 
train_dicts = df_full_train.to_dict(orient='records')
dv = DictVectorizer(sparse=True)
X_train = dv.fit_transform(train_dicts)

#Test
test_dicts = df_test.to_dict(orient='records')
X_test = dv.transform(test_dicts)

rf_model = RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
score= accuracy_score(y_pred, y_test)
print("Accuracy score: " , score)

with open("/mnt/d/github/drugs-classification/model_rf.bin", "wb") as file_out:
    pickle.dump(rf_model, file_out)

with open("/mnt/d/github/drugs-classification/dv.bin", "wb") as file_out:
    pickle.dump(dv, file_out)