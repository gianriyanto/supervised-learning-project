import pandas as pd
from sklearn import preprocessing
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# LOAD THE DATA
df_train = pd.read_csv(r'Assignment3-TrainingData.csv')
df_test = pd.read_csv(r'Assignment3-UnknownData.csv')

# PRE-PROCESSING
# save and drop row ID
train_id = df_train["row ID"]
df_train.drop(columns='row ID', inplace=True)
test_id = df_test["row ID"]
df_test.drop(columns='row ID', inplace=True)

# drop redundant columns, irrelevant to the dataset.
columns_to_drop =['HEAT_D', 'STYLE_D', 'GRADE_D', 'STRUCT_D', 'CNDTN_D', 'EXTWALL_D', 'ROOF_D', 'INTWALL_D']
for col in columns_to_drop:
    df_train.drop(columns=col, inplace=True)
    df_test.drop(columns=col, inplace=True)

# select object columns
obj_col = df_test.columns[df_test.dtypes == 'object'].values
# select non object columns
num_col = df_test.columns[df_test.dtypes != 'object'].values
# replace null value in obj columns with None
df_train[obj_col] = df_train[obj_col].fillna('None')
df_test[obj_col] = df_test[obj_col].fillna('None')
# replace null value in numeric columns with 0
df_train[num_col] = df_train[num_col].fillna(0)
df_test[num_col] = df_test[num_col].fillna(0)

# Encoding
# get all categorical features
df_cat = df_train.select_dtypes(include=[object])
# Encode with Label Encoder
labelEncoder = preprocessing.LabelEncoder()
for col in df_cat.columns:
    df_train[col] = labelEncoder.fit_transform(df_train[col].astype(str))
    df_test[col] = labelEncoder.fit_transform(df_test[col].astype(str))

# FIT THE MODEL
# strip of first and last column (Row ID and target)
X = df_train.drop(columns="QUALIFIED")
y = df_train['QUALIFIED']

# Split data into training and test set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0)

# fit model no training data
xgb_model = XGBClassifier(learning_rate=0.1,
                          colsample_bytree=0.9,
                          subsample=1,
                          n_estimators=1000,
                          min_child_weight=1,
                          max_depth=10,
                          gamma=0)
xgb_model.fit(X_train, y_train)

# make predictions for test data
y_pred = xgb_model.predict(X_val)

# EVALUATE PREDICTIONS
score = xgb_model.score(X, y)
accuracy = accuracy_score(y_val, y_pred)
f1_score = f1_score(y_val, y_pred)
print("Score: %.2f%%" % (score * 100.0))
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("f1 score: %.2f%%" % (f1_score * 100.0))


# PREDICT TEST
test_X = df_test
y_pred_test = xgb_model.predict(test_X)

print(len(test_id))
print(test_X.shape)

# SAVE AND SUBMIT PREDICTION
submission = pd.DataFrame({'Row ID': test_id, 'Predict-Qualified': y_pred_test})

submission.to_csv("submission.csv", index=False)
print("submission.csv successfully saved")


# TUNING HYPERPARAMETER
gsc = GridSearchCV(estimator=xgb_model,
                   param_grid={
                        'learning_rate': [0.1, 0.01, 0.001],
                        'n_estimators': [400, 800, 1200],
                        'subsample': [0.7, 0.8, 0.9],
                        'colsample_bytree': [0.8, 0.9, 1.0],
                        'min_child_weight': [1,3,5],
                        'max_depth': [3, 5, 7],
                        'gamma': [0, 0.9]
                   },
                   cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

gsc.fit(X, y)
print(gsc)
# summarize the results of the grid search
print(gsc.best_score_)
print(gsc.best_estimator_)
print(gsc.best_params_)

