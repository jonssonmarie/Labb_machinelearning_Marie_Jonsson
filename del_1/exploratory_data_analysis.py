import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

"""
Movielens - EDA
"""

# paths to data
movies = "Data/ml-latest/movies.csv"
ratings = "Data/ml-latest/ratings.csv"

# read csv files
df_movies = pd.read_csv(movies, usecols=['movieId', 'title'], dtype={'movieId': 'int32', 'title': 'str', 'genres': str})
df_ratings = pd.read_csv(ratings, usecols=['userId', 'movieId', 'rating'],
                         dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})


def check_result(df1, df2, df1_col, df2_col):
    """
    A function to doublecheck filtering by column name, usually not run
    :param df1: DataFrame
    :param df2: DataFrame
    :param df1_col: column name
    :param df2_col: column name
    :return: DataFrame with all rows after filtering
    """
    df1 = df1.assign(result=df1[df1_col].isin(df2[df2_col]))
    exist = df1[df1['result'] == 1]
    not_exist = df1[df1['result'] == 0]
    print("\ndf_1 shape: ", df1.shape)
    print("df2 shape:", df2.shape)
    print("exist shape:", exist.shape)
    print("not_exist shape:", not_exist.shape, "\n")
    return df1


def check_result_index(df1, df2, df1_col):
    """
     A function to doublecheck filtering by index and column name, usually not run
    :param df1: DataFrame
    :param df2: DataFrame
    :param df1_col: str
    :return: DataFrame with all rows after filtering
    """
    df1 = df1.assign(result=df1[df1_col].isin(df2.index))
    exist = df1[df1['result'] == 1]
    not_exist = df1[df1['result'] == 0]
    print("\ndf_1 shape: ", df1.shape)
    print("df2 shape:", df2.shape)
    print("exist shape:", exist.shape)
    print("not_exist shape:", not_exist.shape, "\n")
    return df1


# Check memory usage, calculate by 2^8 or 2^16 etc
print("df_ratings.info(verbose=False, memory_usage=deep: ", df_ratings.info(verbose=False, memory_usage="deep"))
print("df_movies.info(verbose=False, memory_usage=deep: ", df_movies.info(verbose=False, memory_usage="deep"))


# 1.1 a) begins
def initial_analyse(df):
    """
    :param df: DataFrame
    :return: print
    """
    print("info():\n", df.info(), "\n")
    print("describe():\n", df.describe(), "\n")
    print("value_counts():\n", df.value_counts(), "\n")
    print("head():\n", df.head(), "\n")
    print("tail():\n", df.tail(), "\n")
    print("columns:\n", df.columns, "\n")
    print("index:\n", df.index, "\n")


# Initial analyse
initial_analyse(df_movies)
initial_analyse(df_ratings)


def get_unique(df):
    """
    Collect a files unique items in one DataFrame for easy check
    :param df: DataFrame
    :return: DataFrame
    """
    df_col = df.columns
    unique_col = pd.DataFrame(df.columns)
    for column in df_col:
        unique_col = pd.concat([unique_col, pd.DataFrame(df[column].unique())], axis=1)
        print(f"{column}:\n {df[column].unique()}")
    return unique_col


uni_df_movies = get_unique(df_movies)
uni_ratings = get_unique(df_ratings)

# check statistics by stats
mp_movies = np.array(df_movies)
np_ratings = np.array(df_ratings)
print(stats.describe(np_ratings))


def check_min_max(df, df_col):
    """
    Check columns min and max value
    :param df: DataFrame
    :param df_col: str
    :return: None
    """
    print(f"min: {df[df_col].min()} max: {df[df_col].max()}")


check_min_max(df_ratings, 'userId')
check_min_max(df_ratings, 'movieId')


def plot_num_per_rating(df):
    """
    Plot Number of ratings per rating
    :return: None
    """
    sns.countplot(df.rating)
    plt.title("Number of ratings per rating")
    plt.show()


plot_num_per_rating(df_ratings)


def plot_bar(df):
    """
    Plot Mean rating per top 10 movies
    :param df: DataFrame
    :return: None
    """
    plt.figure(figsize=(18, 15))
    plt.title("Mean rating per top 10 movies")
    sns.barplot(y='title', x='rating', data=df)
    plt.xticks(rotation=0)
    plt.yticks(rotation=60)
    plt.xlabel("Rating", fontsize=14, labelpad=10)
    plt.ylabel("Title", fontsize=14)
    plt.show()


def scatter_plot(df):
    """
    :param df: DataFrame
    :return: None
    """
    plt.figure(figsize=(15, 8))
    plt.title("Rating per User ID")
    sns.scatterplot(data=df, x="userId_rating", y="userId")
    plt.xlabel("Rating", fontsize=16, labelpad=10)
    plt.ylabel("User ID", fontsize=16)
    plt.tight_layout()
    plt.show()


df_ratings['userId_rating'] = df_ratings.groupby(['userId'])['rating'].transform('sum')
sum_user_ratings = df_ratings.reset_index().drop_duplicates(subset=['userId', 'userId_rating'], keep='first')\
                    .drop(columns=["index", "movieId", "rating"]).sort_values(by='userId_rating', ascending=False)

scatter_plot(sum_user_ratings)

df_ratings = df_ratings.drop(columns=["userId_rating"])


def plot_statistics(df):
    """
    :param df: DataFrame
    :return: None
    """
    plt.figure(figsize=(15, 8))
    plot_ = sns.barplot(x='year', y='count', data=df)
    plt.title("Number of movies per year")
    plt.xlabel("Year", fontsize=14, labelpad=10)
    plt.ylabel("Amount", fontsize=14)
    # https://stackoverflow.com/questions/20337664/cleanest-way-to-hide-every-nth-tick-label-in-matplotlib-colorbar
    # answered Dec 3, 2013 at 1:31
    for label in plot_.get_xticklabels()[::2]:
        label.set_visible(False)
    plt.xticks(rotation=90)
    plt.show()


def plot_rating(df, y_value, titles, ylab):
    """
    :param df: DataFrame
    :param y_value: str
    :param titles: str
    :param ylab: str
    :return: None
    """

    plt.figure(figsize=(16, 10))
    plt.title(titles)
    plot_ = sns.scatterplot(x='movieId', y=y_value, data=df)
    # https://stackoverflow.com/questions/20337664/cleanest-way-to-hide-every-nth-tick-label-in-matplotlib-colorbar
    # answered Dec 3, 2013 at 1:31
    for label in plot_.get_xticklabels()[::2]:
        label.set_visible(False)
        plt.xticks(rotation=90)
    plt.xlabel("Movie ID", fontsize=14, labelpad=10)
    plt.ylabel(ylab, fontsize=14)
    plt.show()


# 1.1 b)
ten_movies_sorted = df_ratings['movieId'].value_counts().sort_values(ascending=False).head(10)
ten_movie_titles = df_movies[df_movies['movieId'].isin(ten_movies_sorted.index)]
print(f"Movies with the 10 highest amount of ratings:\n {ten_movie_titles['title'].to_string(index=False)}\n")
#check_result_index(df_movies, ten_movies_sorted, "movieId")


# 1.1 c)
ten_movies_tot_rating_mean = df_ratings[df_ratings['movieId'].isin(ten_movie_titles['movieId'])].mean()\
                             .drop(labels=['movieId', 'userId']).round(2)
print(f"Mean rating for 10 top movies: {ten_movies_tot_rating_mean[0]}\n")
#check_result(df_ratings, ten_movie_titles, 'movieId', 'movieId')


# 1.1 d)
def extract_year(df_mov, col_str):
    """ Extract year to a separate column
    :param df_mov: DataFrame
    :param col_str: str
    :return: DataFrame with column 'year' added df_mov
    """
    df_year = df_mov[col_str].str.extract(r"(\(\d{4}\))", expand=True).dropna(how='any')
    df_year_clean_0 = df_year[0].str.split('(', 1, expand=True)
    df_year_clean_1 = df_year_clean_0[1].str.split(')', 1, expand=True)
    df_year_clean_2 = df_year_clean_1.drop(labels=1, axis=1).dropna(how='any')
    df_year_strip = df_movies.join(df_year_clean_2, how='left').rename(columns={0: 'year'})
    return df_year_strip


df_year_striped = extract_year(df_movies, 'title')


def filter_year(df):
    """
    :param df: DataFrame
    :return: DataFrame filtered on year
    """
    df_year_filtered = df[(df['year'] < '2019') & (df['year'] > '1870')]
    return df_year_filtered


df_movies_year = filter_year(df_year_striped)


check_min_max(df_movies_year, 'year')
uni_df_movies_year = get_unique(df_movies_year)
#check_result(df_movies_year, df_movies, 'movieId', 'movieId')
#check_result(df_movies_year, df_year_striped, 'movieId', 'movieId')
#check_result(df_ratings, df_year_striped, 'movieId', 'movieId')


# and finally a plot for 1.1 d)
movies_per_year = pd.DataFrame(df_movies_year['year'].value_counts()).reset_index()\
                    .rename(columns={'year': 'count', 'index': 'year'})\
                    .sort_values(by="year", ascending=False)
plot_statistics(movies_per_year)


# 1.1 e)
df_ratings['amountRating'] = df_ratings.groupby(['movieId'])['rating'].transform('count')
amount_movies_rating = (df_ratings.merge(df_year_striped, on='movieId', how='inner', suffixes='_caller', copy=False))\
                                  .drop(columns=['userId', 'rating']).sort_values(by="amountRating", ascending=False)\
                                  .drop_duplicates(keep='first')


# for filtering purpose: summary of all ratings for alla movies
df_ratings['sum_rating_movieId'] = df_ratings.groupby(['movieId'])['rating'].transform('sum')
sum_movie_ratings = df_ratings.reset_index().drop(['index', 'userId', 'rating'], axis=1)\
                              .drop_duplicates(keep='first')

# plot for 1.1 e)
plot_rating(amount_movies_rating, 'amountRating', "Amount of rating per movie", ' Amount')
plot_rating(sum_movie_ratings, 'sum_rating_movieId', "Summary of rating per movie", 'Summary')


def filter_on_sum_ratings(df):
    """
    Filtering
    :param df: DataFrame
    :return: DataFrame after filtering
    """
    df['sum_rating_movieId'] = df.groupby(['movieId'])['rating'].transform('sum')
    sum_movie_rating = df.reset_index().drop(['index', 'userId', 'rating'], axis=1).drop_duplicates(keep='first')
    sum_movie_ratings_reduced = sum_movie_rating[(sum_movie_rating['sum_rating_movieId'] > 5000.0)].reset_index()
    df_rating_reduce_pd = df[df['movieId'].isin(sum_movie_ratings_reduced['movieId'])]
    return df_rating_reduce_pd


df_rating_reduced_pd = filter_on_sum_ratings(df_ratings)


def plot_rating_before_after():
    """
    plot data before and after filtering
    :return: None
    """
    plt.figure(figsize=(16, 10))
    plt.title("Rating as sum / amount per movie ")
    plt.scatter(data=df_rating_reduced_pd, x='movieId', y="amountRating", c='b', marker='+', label="amountRating")
    plt.scatter(data=df_rating_reduced_pd, x='movieId', y="sum_rating_movieId", c='r', marker='.',
                label="sum_rating_movieId")
    plt.xlabel("Movie ID", fontsize=14, labelpad=10)
    plt.ylabel("sum/amount", fontsize=14)
    plt.legend()
    plt.show()


plot_rating_before_after()


df_rating_reduced_pd = df_rating_reduced_pd.drop(columns=['sum_rating_movieId', "amountRating"])

# drop columns not to be used anymore
df_ratings = df_ratings.drop(columns=['amountRating', 'sum_rating_movieId'])


# 1.1 f)
movie_rating_mean = df_ratings[df_ratings['movieId'].isin(ten_movies_sorted.index)].groupby('movieId').mean()\
                            .sort_values(by='rating', ascending=False)
check_result_index(df_ratings, ten_movies_sorted, 'movieId')

movie_rating_mean_title = pd.concat([movie_rating_mean, ten_movie_titles.set_index('movieId')], axis=1)\
                            .drop(labels='userId', axis=1)

plot_bar(movie_rating_mean_title)

# 1.1 a) ends
