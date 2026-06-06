# What Makes a Spotify Track Explicit?

By Dariy Marshaev


## Introduction

Music streaming platforms like Spotify collect many types of information about songs, including both metadata and measurable audio features. In this project, I analyze the Spotify Music Tracks dataset, where each row represents one track and each column describes either information about the song or an audio characteristic of that song.

The main question I explore is: **Can Spotify audio features help explain or predict whether a track is explicit?**

This question is interesting because explicitness is not just a random label; it may be connected to certain musical patterns, genres, and audio characteristics. For example, explicit tracks might differ from non-explicit tracks in energy, speechiness, loudness, or genre distribution.

This question is also relevant in the real world because streaming platforms use similar data to organize music, recommend songs, filter content, and understand listening behavior. By studying this dataset, I can investigate how measurable song features relate to a label that matters for both users and platforms.

The dataset contains 114,000 tracks and includes columns such as `track_name`, `artists`, `album_name`, `popularity`, `duration_ms`, `explicit`, `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `time_signature`, and `track_genre`. The column `explicit` is especially important because it tells whether a song is marked as explicit or non-explicit.

## Data Cleaning and Exploratory Data Analysis

To clean the data, I removed the `Unnamed: 0` column because it was just an unnecessary index column from the original CSV file. I also created an `explicit_label` column to make the values of `explicit` easier to interpret in visualizations, changing `True` to `"Explicit"` and `False` to `"Non-explicit"`. Finally, I created a `duration_min` column by converting `duration_ms` from milliseconds to minutes, since minutes are easier to interpret when describing song length.

In my exploratory data analysis, I compared explicit and non-explicit tracks across several audio features. One important feature was `energy`, which measures how intense or active a track sounds. The box plot comparing energy scores showed that explicit tracks generally had higher energy scores than non-explicit tracks.

![Energy of Explicit vs Non-Explicit Spotify Tracks](images/energy_boxplot.jpeg)

I also examined `speechiness`, which measures the presence of spoken words in a track. This feature is relevant because explicit songs may be more common in genres with more lyrical or spoken-word-heavy content, such as rap or hip-hop. In addition, I looked at the proportion of explicit tracks by genre, since genre is likely connected to whether a song is explicit.

Overall, the exploratory analysis suggested that explicitness is related to both audio features and genre-level patterns.

## Assessment of Missingness

For the missingness analysis, I focused on the missingness of `track_name`. I created a missingness indicator that recorded whether each track had a missing track name. Then, I tested whether the missingness of `track_name` depended on `popularity`.

The null hypothesis was that the missingness of `track_name` does not depend on `popularity`; in other words, the popularity distribution is the same for rows with missing and non-missing track names. The alternative hypothesis was that the missingness of `track_name` does depend on `popularity`.

The test statistic was the absolute difference in mean popularity between tracks with missing `track_name` values and tracks with non-missing `track_name` values. This helped determine whether missing track names were associated with systematically different popularity scores.

The missingness of `track_name` could potentially be NMAR if the reason a track name is missing is related to the track name itself. For example, some track names may contain unusual characters, formatting issues, or metadata problems that make them harder to store or process correctly. However, the missingness could also be MAR if it is explained by another observed column, such as `popularity`, `track_genre`, or `artists`.

## Hypothesis Testing

For my hypothesis test, I tested whether explicit tracks have higher energy scores than non-explicit tracks.

**Null Hypothesis:** Explicit and non-explicit tracks come from the same distribution of energy scores. Any observed difference in average energy is due to random chance.

**Alternative Hypothesis:** Explicit tracks have higher energy scores on average than non-explicit tracks.

**Test Statistic:** Mean energy of explicit tracks minus mean energy of non-explicit tracks.

The observed test statistic was **0.0872**, meaning that explicit tracks had an average energy score about 0.0872 higher than non-explicit tracks. The p-value was approximately **0.0**.

![Permutation Test for Difference in Mean Energy](images/hypothesis_test.jpeg)

Since the p-value is less than 0.05, I reject the null hypothesis. This means there is statistically significant evidence that explicit tracks have higher energy scores on average than non-explicit tracks.

## Framing a Prediction Problem

For my prediction problem, I predict the `explicit` column. This column indicates whether a Spotify track is marked as explicit or non-explicit. Since there are only two possible outcomes, this is a binary classification problem.

The response variable is `explicit`. A value of `True` means the track is explicit, while a value of `False` means the track is non-explicit.

This prediction problem is useful because explicitness is an important label for music streaming platforms. Platforms may use this kind of information for content filtering, recommendations, playlist organization, and user preferences. I am interested in whether audio features such as `energy`, `danceability`, `speechiness`, `loudness`, `acousticness`, `valence`, and `tempo` can help predict whether a track is explicit.

I evaluate my models using accuracy and F1-score. Accuracy measures the overall proportion of correct predictions, while F1-score is useful because the dataset is imbalanced, with many more non-explicit tracks than explicit tracks. Since predicting explicit tracks correctly is important, F1-score gives a better sense of performance on the positive class than accuracy alone.

## Baseline Model

For my baseline model, I used a logistic regression classifier to predict whether a track is explicit. Since `explicit` is a binary variable, logistic regression is a reasonable simple baseline model.

The baseline model used four quantitative audio features: `energy`, `danceability`, `speechiness`, and `loudness`. These features are all numeric, so they were standardized before being passed into the model. I chose these features because they describe basic audio characteristics of a song and are relatively easy to interpret.

The baseline logistic regression model achieved an accuracy of **0.914** and an F1-score of **0.112** for the explicit class.

Although the accuracy is high, the classification report showed that the model performed poorly at identifying explicit tracks. The model had high recall for non-explicit tracks, but for explicit tracks it only had a recall of **0.06**, meaning it missed most explicit songs. This happened because the dataset is imbalanced: most tracks are non-explicit, so the model can get high accuracy by mostly predicting the majority class.

The confusion matrix supported this. Out of 1,949 explicit tracks in the test set, the model correctly predicted only 124 of them as explicit and incorrectly predicted 1,825 of them as non-explicit. This means the baseline model was not strong enough for the main prediction task.

## Final Model

For my final model, I improved on the baseline model by adding more features and tuning the model. The baseline model only used four quantitative audio features, so it missed important information such as genre and other audio characteristics.

For the final model, I used both quantitative and categorical features. I added `track_genre` as a categorical feature because explicitness likely depends strongly on genre. I also added more numeric audio features, including `acousticness`, `instrumentalness`, `valence`, `tempo`, and `duration_min`. In addition, I created an engineered feature called `energy_speechiness`, which multiplies `energy` and `speechiness`. This feature may be useful because explicit tracks may be both energetic and lyric-heavy.

I used logistic regression again, but tuned the regularization strength `C` and the `class_weight` parameter using GridSearchCV. The best parameters selected by GridSearchCV were `C = 0.1` and `class_weight = 'balanced'`. The balanced class weight helped because the dataset contains many more non-explicit tracks than explicit tracks.

The final model achieved an accuracy of **0.772** and an F1-score of **0.388** for the explicit class. Compared to the baseline model, which had an accuracy of 0.914 and an F1-score of 0.112, the final model had lower overall accuracy but much better performance on the explicit class.

![Baseline Model vs Final Model Performance](images/model_comparison.jpeg)

The main improvement was recall for explicit tracks. In the baseline model, the recall for explicit tracks was only 0.06, meaning the model missed most explicit songs. In the final model, recall for explicit tracks increased to **0.81**, meaning the model correctly identified most of the explicit tracks in the test set. The confusion matrix showed that the final model correctly predicted 1,325 explicit tracks and missed only 315 explicit tracks.

Overall, the final model is better for my prediction task because it is much more successful at identifying explicit tracks, even though it sacrifices some overall accuracy.

## Fairness Analysis

For my fairness analysis, I compared the final model's performance on high-popularity tracks and low-popularity tracks. I defined high-popularity tracks as tracks with popularity greater than or equal to the median popularity in the test set, and low-popularity tracks as tracks below the median.

I used recall for the explicit class as my evaluation metric. Recall measures the proportion of actually explicit tracks that the model correctly identifies as explicit. This metric is important because my baseline model had high accuracy but missed most explicit tracks.

**Null Hypothesis:** The final model has the same recall for explicit tracks in the high-popularity and low-popularity groups. Any observed difference is due to random chance.

**Alternative Hypothesis:** The final model has different recall for explicit tracks in the high-popularity and low-popularity groups.

**Test Statistic:** The absolute difference in recall between the high-popularity and low-popularity groups.

The final model had similar recall for explicit tracks across the two popularity groups. The recall was **0.805** for high-popularity tracks and **0.812** for low-popularity tracks. The observed absolute difference in recall was **0.0069**, and the p-value was **0.743**.

![Fairness Analysis: Difference in Explicit Recall by Popularity Group](images/Fairness_analysis.jpeg)

Since the p-value is greater than 0.05, I fail to reject the null hypothesis. This means there is not statistically significant evidence that the final model has different explicit-track recall for high-popularity and low-popularity songs.

In this fairness analysis, the model appears to perform similarly across the two popularity groups with respect to recall for explicit tracks.

