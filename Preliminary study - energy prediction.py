import pandas as pd
import numpy as np

print('...now loading data...')
epc = pd.read_csv('processed_epc.csv', index_col=0)

print('...now specifying x and y...')
inputs = ['Property type','Built form', 'Number habitable room', 'Number heated room', 'Roof description', 'Walls description', 'Floor description', 'Lighting description', 'Main heat', 'predicted age band']
X=epc.iloc[:,inputs]
y=epc['ENERGY_CONSUMPTION_CURRENT']


# print autosklearn version
import autosklearn
print('autosklearn: %s' % autosklearn.__version__)

print('...now importing packages...')
# example of auto-sklearn for a classification dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from autosklearn.regression import AutoSklearnRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
#from autosklearn.metrics import (accuracy,f1_macro,precision,recall,average_precision,log_loss)

''''
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

#from sklearn.compose import ColumnTransformer
#ct = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

print('...now transforming...')
X_transformed = ct.fit_transform(X)
y_transformed = ct.fit_transform(y)
'''

print('...now splitting data...')
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


print('...now defining model search...')
# define search
automl = AutoSklearnRegressor(
                            data_preprocessor = True,
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
board.to_csv('automl/energy-leaderboard.csv')
print(automl.leaderboard(detailed = True))

print('...now printing models found...')
ensemble_dict = automl.show_models()
print(ensemble_dict)


print('...now evaluating...')
# evaluate best model
prediction = automl.predict(X_test)
acc = accuracy_score(y_test, prediction)
print("Accuracy: %.3f" % acc)
train_predictions = automl.predict(X_train)
print("Train R2 score:", sklearn.metrics.r2_score(y_train, train_predictions))
test_predictions = automl.predict(X_test)
print("Test R2 score:", sklearn.metrics.r2_score(y_test, test_predictions))


print('...now trying performance over time plot...')
automl.performance_over_time_.plot(
        x='Timestamp',
        kind='line',
        legend=True,
        title='Auto-sklearn accuracy over time',
        grid=True,
    )
plt.savefig('automl/plot_energy.pdf')  
plt.show()

plt.scatter(train_predictions, y_train, label="Train samples", c='#d95f02')
plt.scatter(test_predictions, y_test, label="Test samples", c='#7570b3')
plt.xlabel("Predicted value")
plt.ylabel("True value")
plt.legend()
plt.plot([30, 400], [30, 400], c='k', zorder=0)
plt.xlim([30, 400])
plt.ylim([30, 400])
plt.tight_layout()
plt.savefig('automl/scatter_energy.pdf')
plt.show()


