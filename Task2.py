# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
# from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

# %%
with open('Graduate - HEADLINES dataset (2019-06).json', 'r') as jsonFile:
    df = pd.read_json(jsonFile, lines=True)

# %%
print(df.info())

print(df.isna().sum())

# %%
print(df.groupby('is_sarcastic').count())

# %%
print(df.describe())

# %%
print(df.head(20))

# %%
pipe_sgdc = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
    alpha=1e-3, random_state=42, max_iter=5, tol=None)),
])
# para_sgdc = {
#     'vect__max_df': (0.5, 0.75, 1.0),
#     'vect__max_features': (None, 5000, 10000, 50000),
#     'vect__ngram_range': ((1, 1), (1, 2), (1,3)),  # unigrams, bigrams or trigrams
#     'vect__stop_words': ('english', None),
#     'tfidf__use_idf': (True, False),
#     'tfidf__norm': ('l1', 'l2'),
#     'clf__alpha': (0.00001, 0.000001),
#     'clf__penalty': ('l2', 'elasticnet'),
#     'clf__max_iter': (10, 50, 80),
# }

# %%
accuracy = []
f1score = []
cfm = []
sk = StratifiedKFold(n_splits=10, shuffle=True)
X = df['headline']
Y =  df['is_sarcastic']
for train, test in sk.split(X, Y):
    pipe_sgdc.fit(X[train], Y[train])
    predicted = pipe_sgdc.predict(X[test])
    # print(metrics.classification_report(Y[test], predicted))
    accuracy.append(accuracy_score(Y[test], predicted) * 100)
    f1score.append(f1_score(Y[test], predicted, average='macro') * 100)
    cfm.append(confusion_matrix(Y[test], predicted, labels=[0, 1]))

print("Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(accuracy), np.std(accuracy)))
print("F1 score: %.2f%% (+/- %.2f%%)" % (np.mean(f1score), np.std(f1score)))
print("\nConfusion Matrix:\n", np.sum(cfm, axis=0),'\nNumber of instances: ', np.sum(cfm))

# %%
# gs_clf = GridSearchCV(pipe_sgdc, para_sgdc, cv=5, n_jobs=-2, verbose=True)
# gs_clf = gs_clf.fit(df['headline'], df['is_sarcastic'])
# print(gs_clf.best_estimator_.get_params())
# print(gs_clf.best_score_)