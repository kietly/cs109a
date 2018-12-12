---
title: Conclusions
nav_include: 5
---

### Results

#### Model Accuracy

We ran 8 different models against both the baseline set of features and extended features. Models trained against the baseline features did not achieve high accuracy, with the range in accuracy across all the models falling between 76% and 79%.

When trained and tested against the extended data, however, the accuracy significantly improved for all models and ranged between 89% and 99%.
Even the Logistic Regression Classifier, though the worst performer, had a 6% improvement in accuracy just by adding the extended fields.
We chose the Random Forest Classifier as our best performing model because it provided the highest accuracy with the least complexity.
The table summarize the progressive improvement in accuracy among all the dataset.

|Models|Base features|NLP features|NLP features+Lexical Diversity|
|:-----|:------------|:-----------|:-----------------------------|
|Logistics Regression|76.4|82.4|82.7|
|LDA|71.6|81.0|81.1|
|QDA|75.4|76.3|73.5|
|DecisionTree|79.5|96.1|99.1|
|RandomForest|79.4|92.3|99.3|
|AdaBoost|79.5|98.9|99.4|

Table 1. Models prediction accuracy

#### NLP Features Improve Bot Detection

Based on the results of our testing, we assess that including NLP features engineered from the tweet text improved Bot Detection. These Online Social Platforms are built for users to communicate ideas, so capturing deeper language patterns in the tweet text itself, in combination with metrics about the reaction to the tweets and users (likes, retweets, etc), and other activity can improve bot detection.

#### Future Work
 There are still many unexplored ideas that we have yet to explore. For example, we read research indicating that analyzing the temporal nature of tweet activity can uncover bot behaviors. It would be interesting to explore these temporal patterns in user behavior and again extend these with NLP to see if Tweet Topics have temporal patterns.
