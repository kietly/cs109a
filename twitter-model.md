---
title: Models
nav_include: 4
---

#### Baseline and NLP features

From Milestone#3, we have determined only these baseline features are important, and similarly for NLP. Please refer
back to EDA and NLP pages for explanation of these selected features

|Baseline features|NLP features| 
|:----------------|:-----------|
|retweet_count|sentiment_negative|
|favorite_count|sentiment_neutral|
|num_urls|sentiment_positive|
|num_mentions|token_count|
|num_hashtags|url_token_ratio|
| |ratio_neg|
| |ant|
| |fear|
| |joy|
| |trust|
| |jaccard|

### Evaluate Baseline features accuracy

1. ##### kNN with Baseline features
We have a fairly large number of tweets (over 100K). kNN takes forever to run. We decided drop the kNN from further evaluation.  

```
all_tweets_df = all_tweets[['retweet_count', 'favorite_count', 'num_hashtags', 'num_urls', 'num_mentions','user_type']]
train_base_tweets_df, test_base_tweets_df = train_test_split(all_tweets_df, test_size=0.33, random_state=42,
                                                              stratify=all_tweets_df['user_type'])

X_train, y_train = train_base_tweets_df.drop('user_type',axis=1), train_base_tweets_df['user_type']
X_test, y_test = test_base_tweets_df.drop('user_type',axis=1), test_base_tweets_df['user_type']
Xs_train, Xs_test = scale(X_train), scale(X_test)

neighbors, train_scores, cvmeans, cvstds, cv_scores = [], [], [], [], []
for n in range(1,11):
    neighbors.append(n)
    knn = KNeighborsClassifier(n_neighbors = n)
    train_scores.append(knn.fit(X_train, y_train).score(X_train, y_train))
    scores = cross_val_score(estimator=knn,X=Xs_train, y=y_train, cv=5)
    cvmeans.append(scores.mean())
    cvstds.append(scores.std())
```

2. ##### LDA/QDA with Baseline features
LDA and QDA performed poorly. They are in the range of 71-75% accuracy. Also note that LDA/QDA perform well with 
low number of observations according to class lecture. Note we have over 100K observations.

```
X_train, y_train = train_base_tweets_df.drop('user_type',axis=1), train_base_tweets_df['user_type']
X_test, y_test = test_base_tweets_df.drop('user_type',axis=1), test_base_tweets_df['user_type']

lda = LinearDiscriminantAnalysis().fit(X_train, y_train)
qda = QuadraticDiscriminantAnalysis().fit(X_train, y_train)
```

3. ##### Decision Tree/RandomForest with Baseline features
The Decision Tree performed well here. After tree depth = 7, the accuracy is not improving.
We pick 7 as the optimal tree depth for RandomForest. Given the tree accuracy is fairly
constant, the accuracy is ~ 79% for both models.

```
depths, train_scores, cvmeans, cvstds, cv_scores = [], [], [], [], []
for depth in range(1,21):
    depths.append(depth)
    dt = DecisionTreeClassifier(max_depth=depth)
    train_scores.append(dt.fit(X_train, y_train).score(X_train, y_train))
    scores = cross_val_score(estimator=dt, X=X_train, y=y_train, cv=5)
    cvmeans.append(scores.mean())
    cvstds.append(scores.std())
```
![Decision Tree with Baseline features](image/dt_baseline.PNG)

```
fitted_rf = RandomForestClassifier(n_estimators=7, max_depth=7).fit(X_train,y_train)
random_forest_train_score = fitted_rf.score(X_train, y_train)
random_forest_test_score = fitted_rf.score(X_test, y_test)
```

### Evaluate NLP features accuracy
Now we rerun the same models again with the addition of NLP features the model, the  
accuracy should improve.

4. ##### LDA/QDA with NLP features
Even with NLP features added, the accuracy of the LDA is 81% and 75% for QDA. We decided to drop
these models from further evaluation and concentrated on the models that are doing really well.

5. ##### Decision Tree/RandomForest with NLP features

With the addition of NLP features, the DecisionTreeClassifier accuracy improved dramatically.
The graph shows over fitting starting at depth = 13. We will select depth = 12 as the optimal
depth for RandomForest. The RandomForestClassifier accuracy is lower than decision tree because it
is an average result over multiple estimators (DecisionTree).  

```
# Base + NLP features
all_tweets_df = all_tweets[['retweet_count', 'favorite_count', 'num_hashtags', 'num_urls',
'num_mentions','user_type', 'sentiment_negative', 'sentiment_neutral', 'sentiment_positive',
'token_count', 'url_token_ratio', 'ratio_neg', 'ant', 'fear', 'joy', 'trust','jaccard']]

#Choosing the best depth
idx = depths.index(12)
print("Accuracy: Mean={:.3f}, +/- 2 SD: [{:.3f} -- {:.3f}]".format(
    cvmeans[idx], cvmeans[idx] - 2*cvstds[idx], cvmeans[idx] + 2*cvstds[idx]))

Accuracy: Mean=0.961, +/- 2 SD: [0.958 -- 0.964]   
The Random Forest scored 0.928 on the training set.
The Random Forest scored 0.923 on the test set.
```                            

![DT with NLP](image/dt_nlp.PNG)
