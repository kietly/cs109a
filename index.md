### Group #4 CS109a Data Science Final Project
Kiet Ly, Mary Monroe, and Shaswati Mukherjee

![Evil Twitter](image/social-media-free-speech-weapon.png)

#### Motivation
The role of social media bots, automated accounts capable of posting content or interacting
with other users with no direct human involvement, has been the subject of much scrutiny and
attention in recent years. These bots have been used to attempt to alter perceptions
of political discourse on social media, spread misinformation, and manipulate online rating and
review systems[1]. In the 2016 United States Presidential election, a recent study estimated that Twitter Bots may have boosted President Trump's votes by 3.23%. During the United Kingdom Brexit vote in 2016,
the bots may have added 1.76% points to "pro-leave" votes[6].

#### The Questions
Accurately identifying these social bots tweets can provide an effective weapon to curb propaganda, disinformation, and provocation [7]. We planned to determine whether automated social bot detection was achievable by applying machine learning techniques to tweet data. In addition, we tested whether including NLP based features such as those built from sentimental and emotional analysis would enhance prediction accuracy.


#### Investigatory Approach

Given the above questions, we intended to find the answers by using the following approach:

1. Obtain baseline features by performing EDA and using feature selection to identify the most important features
 of the tweet data.
2. Run pre-selected classification models on the baseline features to obtain a baseline
accuracy.
3. Generate extended features based on the tweet text by using NLP techniques such as topic modeling, linguistic, sentimental and emotional analysis.
4. Perform feature selection to choose the most important features from the extended features.
5. Combine the baseline and extended features, retrain and test the same classification models from step #2.
6. Note any changes in accuracy
7. Pick the highest accuracy model and tune it further if needed.
