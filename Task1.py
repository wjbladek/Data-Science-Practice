# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.stats import shapiro


# %%
df = pd.read_csv('Graduate - IRISES dataset (2019-06).csv', sep='|')

# %%
print(df.info())
print('\n')
print(df.groupby('Species').count())
print('\n')
print(df.describe())
print('\n')
print(df.isna().sum())
# %%

print('\n',df[df['Sepal.Length'] < 0])
print('', df[df['Sepal.Width'].isna()])

# %%
# do after splitting values to improve validity of the model 
# df.fillna(df.mean(), inplace=True)
df['Petal.Width'] = df['Petal.Width'].replace(',','.', regex=True).astype(float)
df.loc[25,'Sepal.Length'] = 4.8
print(df.loc[25], '\n')
print(df.loc[82])


# %%
# print('\n',df.to_string())


# %%

sns.set(style="ticks")
sns.pairplot(df,hue="Species")

# %%
clear_df = df.fillna(df.median())
sns.distplot(clear_df['Sepal.Length'])
sns.distplot(clear_df['Sepal.Width'])
sns.distplot(clear_df['Petal.Length'])
sns.distplot(clear_df['Petal.Width'])


# %%
print('Sepal.Length: ', shapiro(clear_df['Sepal.Length']))
print('Sepal.Width: ', shapiro(clear_df['Sepal.Width']))
print('Petal.Length: ', shapiro(clear_df['Petal.Length']))
print('Petal.Width: ', shapiro(clear_df['Petal.Width']))

# %%
sk = StratifiedKFold(n_splits=5, shuffle=True)
accuracy = []
f1score = []
cfm = []
svc = SVC()
lda = LDA()
classifier = lda
X = np.array(df.drop('Species', axis=1))
Y = np.array(df['Species'])
for train, test in sk.split(X, Y):
    ind = np.where(np.isnan(X))
    if ind in X[train]:
        col_median = np.nanmedian(X[train][-1], axis=0)
    else:
        col_median = np.nanmedian(X[test][-1], axis=0)
    X[ind] = col_median

    classifier.fit(X[train], Y[train])
    predicted = classifier.predict(X[test])
    accuracy.append(accuracy_score(Y[test], predicted) * 100)
    f1score.append(f1_score(Y[test], predicted, average='macro') * 100)
    cfm.append(confusion_matrix(Y[test], predicted, labels=["setosa", "versicolor", "virginica"]))

print("Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(accuracy), np.std(accuracy)))
print("F1 score: %.2f%% (+/- %.2f%%)" % (np.mean(f1score), np.std(f1score)))
print("\nConfusion Matrix:\n", np.sum(cfm, axis=0),'\nNumber of instances: ', np.sum(cfm))
