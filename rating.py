"""
Course Rating Analysis

This script performs a comprehensive analysis of course ratings using:
- Simple Average Rating
- Time-Based Weighted Average
- User Progress-Based Weighted Average
- A final Weighted Rating combining both time and progress weights

Dataset: course_reviews.csv
"""

###################################################
# Rating Products
###################################################

# - Average
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating


############################################
# Application: User and Time Weighted Course Rating Calculation
############################################

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# (50+ Hours) Python A-Z™: Data Science and Machine Learning
# Rating: 4.8 (4.764925)
# Total Ratings: 4611
# Rating Percentages: 75, 20, 4, 1, <1
# Approximate Numerical Equivalents: 3458, 922, 184, 46, 6

df = pd.read_csv("course_reviews.csv")
df.head()
df.shape

# Distribution of Ratings
df["Rating"].value_counts()

df["Questions Asked"].value_counts()

# Average rating based on number of questions asked
df.groupby("Questions Asked").agg({"Questions Asked": "count",
                                   "Rating": "mean"})


df.head()

####################
# Average
####################

# Average Rating
df["Rating"].mean()

####################
# Time-Based Weighted Average
####################
# Weighted Average Based on Rating Times

df.head()
df.info()

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Calculate how many days ago each review was made
# We have information about when the reviews and ratings were made
# Determine today's date and subtract the review date
current_date = pd.to_datetime('2021-02-10 0:0:0')

df["days"] = (current_date - df["Timestamp"]).dt.days

# Average rating for reviews in the last 30 days
# Select only the rating from this subset
df.loc[df["days"] <= 30, "Rating"].mean()

# Second range: greater than 30 and less or equal to 90 days
df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean()

df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean()

df.loc[(df["days"] > 180), "Rating"].mean()

# Assign weights to each time segment
df.loc[df["days"] <= 30, "Rating"].mean() * 28/100 + \
    df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() * 26/100 + \
    df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() * 24/100 + \
    df.loc[(df["days"] > 180), "Rating"].mean() * 22/100

def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[dataframe["days"] <= 30, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["days"] > 30) & (dataframe["days"] <= 90), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["days"] > 90) & (dataframe["days"] <= 180), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["days"] > 180), "Rating"].mean() * w4 / 100

time_based_weighted_average(df)

time_based_weighted_average(df, 30, 26, 22, 22)



####################
# User-Based Weighted Average   ( Everyone's rating weight should be the same! )
####################
# Should different progress levels have different weight on rating?
# Like the difference between watching 5% and 75% of the course

df.head()

df.groupby("Progress").agg({"Rating": "mean"})

df.loc[df["Progress"] <= 10, "Rating"].mean() * 22 / 100 + \
    df.loc[(df["Progress"] > 10) & (df["Progress"] <= 45), "Rating"].mean() * 24 / 100 + \
    df.loc[(df["Progress"] > 45) & (df["Progress"] <= 75), "Rating"].mean() * 26 / 100 + \
    df.loc[(df["Progress"] > 75), "Rating"].mean() * 28 / 100



def user_based_weighted_average(dataframe, w1=22, w2=24, w3=26, w4=28):
    return dataframe.loc[dataframe["Progress"] <= 10, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 10) & (dataframe["Progress"] <= 45), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 45) & (dataframe["Progress"] <= 75), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 75), "Rating"].mean() * w4 / 100


user_based_weighted_average(df, 20, 24, 26, 30)


####################
# Final Weighted Rating (Combining Time & User-Based Averages)
####################

def final_weighted_rating(dataframe, time_w=50, user_w=50):
    return time_based_weighted_average(dataframe) * time_w/100 + user_based_weighted_average(dataframe)*user_w/100

final_weighted_rating(df)

final_weighted_rating(df, time_w=40, user_w=60)




############################################
# SORTING REVIEWS (Wilson Lower Bound)
############################################

def score_up_down_diff(up, down):
    return up - down

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence level (default: 0.95)

    Returns
    -------
    wilson score: float
    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

# Example case study
up = [15, 70, 14, 4, 2, 5, 8, 37, 21, 52, 28, 147, 61, 30, 23, 40, 37, 61, 54, 18, 12, 68]
down = [0, 2, 2, 2, 15, 2, 6, 5, 23, 8, 12, 2, 1, 1, 5, 1, 2, 6, 2, 0, 2, 2]
comments = pd.DataFrame({"up": up, "down": down})

comments["score_pos_neg_diff"] = comments.apply(lambda x: score_up_down_diff(x["up"], x["down"]), axis=1)
comments["score_average_rating"] = comments.apply(lambda x: score_average_rating(x["up"], x["down"]), axis=1)
comments["wilson_lower_bound"] = comments.apply(lambda x: wilson_lower_bound(x["up"], x["down"]), axis=1)

comments = comments.sort_values("wilson_lower_bound", ascending=False)
print(comments.head())
