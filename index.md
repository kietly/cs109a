## Group #4 CS109a Data Science Final Projects
Kiet Ly, Mary Monroe, and Shaswati Mukherjee

![Evil Twitter](image/social-media-free-speech-weapon.png)

#### Motivation:
The role of so-called social media "bots" automated accounts capable of posting content or interacting
with other users with no direct human involvement has been the subject of much scrutiny and
attention in recent years. These bosts can also be used to attempt to alter perceptions
of political discourse on social media, spread misinformation, or manipulate online rating and
review systems. In the 2016 election, Twitter bots were shaping the outcome of the election.
A recent study by Times, Twitter Bots may have boosted Trump's votes by 3.23%. During the Brexit 2016,
the bots may have added 1.76% point to "pro-leave" votes[6].

#### The Questions
We want to automate the identification of these social bots through machine learning using just tweet
data. In addition, we want to know by using NLP to generate additional features such as sentimental and emotional
features will enhance prediction accuracy. By accurately identify these social bots tweets, it will provide an effective weapon to curb
propaganda, disinformation, and provocation [7].


#### Investigation approaches:

Given the above questions, here is how we find the answers to the questions above.
1. Apply EDA and using feature selection to identify the most important features from
the tweets data collection. We call this the baseline features.
2. Run pre-selected classification models on the baseline features. This give us the baseline
accuracy.
3. Using NLP to generate additional features for topic modeling, sentimental and emotional features.
4. Again, we use features selection to select the most important features from NLP. We call this extended features
5. Combining the baseline and extended features and rerun with the same classification models from step #2.
6. Pick the highest accuracy model and tune it further if needed.
