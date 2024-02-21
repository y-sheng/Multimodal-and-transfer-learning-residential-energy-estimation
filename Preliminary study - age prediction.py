import pandas as pd
import numpy as np

print('now loading data...')
age = pd.read_csv('aggregated_age_sample.csv', index_col=0)

print('now specifying x and y...')
for_age = ['Total floor area','Perimeter','Relh2','Relhmax', 'NPI', 'Vxcount','Builtrate']
X=age.iloc[:,for_age]
y=age['agg_age']

# print autosklearn version
import autosklearn
print('autosklearn: %s' % autosklearn.__version__)

print('...now importing packages...')
# example of auto-sklearn for a classification dataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from autosklearn.classification import AutoSklearnClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt

'''
print('...now building transformation...')
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])


#building pipeline for data transformation
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

print('...now transforming...')
X_transformed = ct.fit_transform(X)
#y_transformed = ct.fit_transform(y)
'''

print('...now splitting data...')
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


print('...now defining model search...')
# define search
automl = AutoSklearnClassifier(
                            exclude = {
                'feature_preprocessor': ["no_preprocessing"]
                                        },
                            resampling_strategy='cv',
                            resampling_strategy_arguments={'folds': 5})

print('...now fitting...')
# perform the search
automl.fit(X_train, y_train)


print('...fitting finished...')
print('......')
print('...statistics...')
# summarize
print(automl.sprint_statistics())

print('...leaderboard...')
board = pd.DataFrame(automl.leaderboard(detailed = True))
board.to_csv(r'age-leaderboard.csv')
print(automl.leaderboard(detailed = True))

print('...now printing models found...')
ensemble_dict = automl.show_models()
print(ensemble_dict)

print('...now evaluating...')
# evaluate best model
prediction = automl.predict(X_test)
acc = accuracy_score(y_test, prediction)
f1_macro_score = autosklearn.metrics.f1_macro(y_test,prediction)
print("Accuracy: %.3f" % acc)
print("f1_macro: %.3f" % f1_macro_score)


print('...now trying performance over time plot...')
automl.performance_over_time_.plot(
        x='Timestamp',
        kind='line',
        legend=True,
        title='Auto-sklearn accuracy over time',
        grid=True,
    )
plt.savefig(r'plot_age.pdf')  
plt.show()

print('...for sankey...')
sankey_df = pd.DataFrame(y_test)
sankey_df['pred'] = prediction
print('...now saving df for sankey to results file...')
sankey_df.to_csv(r'sankey_age_automl.csv')

print('...now performing predictions for the rest of properties')

print('...now loading all epcs...')
o_epc = pd.read_csv(r'epc.csv', index_col=0)
to_predict = o_epc.iloc[:,for_age]

print('...now performing prediction...')
o_predict = automl.predict(to_predict)
result = pd.DataFrame(o_predict)
o_epc['predicted_agebandas']=result

print('...save results to drive...')
o_epc.to_csv(r'age_result_automl.csv')
