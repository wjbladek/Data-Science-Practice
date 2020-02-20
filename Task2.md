# Task 2 &mdash; irony classifier

`This report shows my line of thought. To present my workflow better, I have attached excerpts of the source code.`

**Goal: classification of ironic statements** 

## Concerning irony
Irony is an intentional discrepancy between ostensible, semantic meaning, and pragmatic meaning. Such statements mean exactly the opposite of what their syntax suggests. What is even worse, irony is often contextual, requiring a general knowledge of certain cultural contexts. 

With that said, I do not expect any fancy results from the model I will present below. I simply lack resources and (currently) skills necessary to make a system that has a general understanding of words/notions, has and ability to associate ideas that words convey and, finally, is able to recognize contradiction in semantic and pragmatic meanings.

In this task I will focus on much simpler model, using Bag-of-Words model, implemented in Python, using *scikit-learn* library.

## Reviewing data

Let us begin with loading up the data. Quickly we find out that a "lines=" argument will be required. Visual inspection reveals that the dataset consists of multiple rows of single-line objects.
```Python
import pandas as pd
with open('Graduate - HEADLINES dataset (2019-06).json', 'r') as jsonFile:
    df = pd.read_json(jsonFile, lines=True)
```

Because it is a large dataset, checking its quality requires code. 

```Python
print(df.info())
print(df.isna().sum())
print(df.groupby('is_sarcastic').count())
print(df.describe())
print(df.head(20))
```

We have two, slightly imbalanced classes and no incorrect values. Hurray! Let us move on to the meat of the task.

## Pipelines 

NLP requires many steps, in order to process text into an algorithm-consumable form. Pipelines simplifies workflow and present parameters (which are of interest to a data scientist) in a clear and easy to play with way. 

The first element of the pipeline is CountVectorizer, which counts occurrences of words in the dataset. However, they are dependent on data length, making it suboptimal for machine learning. Therefore, they are converted into frequencies (TfidfTransformer). Finally, frequencies of terms can be outputted into our classifier of choice. 

```Python
pipe_sgdc = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
    alpha=1e-3, random_state=42, max_iter=5, tol=None)),
])
```

## Classification

similarly to *Task1*, I have decided to use a stratified cross-validation technique. I have increased the number of fold to 10, because of size of the dataset.

```Python
sk = StratifiedKFold(n_splits=10, shuffle=True)
X = df['headline']
Y =  df['is_sarcastic']
for train, test in sk.split(X, Y):
    pipe_sgdc.fit(X[train], Y[train])
    predicted = pipe_sgdc.predict(X[test])
```

## Results
    Accuracy: 80.36% (+/- 0.39%)\
    F1 score: 79.48% (+/- 0.43%)

Summed Confusion Matrix:
| true\predicted | Normal | Ironic |
|----------------|--------|--------|
| Normal         | 13492  | 1493   |
| Ironic         | 3753   | 7971   |
Number of instances:  26709


## Discussion
Obviously it is a very simple model. Still, 80% accuracy for such a simple solution is surprising. Experiments with grid search parameter optimisation shows that it might rise up to ~85% with proper parameters.