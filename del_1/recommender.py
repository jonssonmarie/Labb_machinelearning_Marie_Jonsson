import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process
from scipy.sparse import coo_matrix
import time

# paths to data
movies = "Data/ml-latest/movies.csv"
ratings = "Data/ml-latest/ratings.csv"

# read csv files
df_movies = pd.read_csv(movies, usecols=['movieId', 'title'], dtype={'movieId': 'int32', 'title': 'str'})
df_ratings = pd.read_csv(ratings, usecols=['userId', 'movieId', 'rating'],
                         dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})


# filtering
def filter_on_sum_ratings(df):
    df['sum_rating_movieId'] = df.groupby(['movieId'])['rating'].transform('sum')
    sum_movie_ratings = df.reset_index().drop(['index','userId', 'rating'], axis=1).drop_duplicates(keep='first')
    sum_movie_ratings_reduced = sum_movie_ratings[(sum_movie_ratings['sum_rating_movieId'] > 5000.0)].reset_index()
    df_rating_reduce_pd = df_ratings[df_ratings['movieId'].isin(sum_movie_ratings_reduced['movieId'])].drop(columns=['sum_rating_movieId'])
    return df_rating_reduce_pd


df_rating_reduced_pd = filter_on_sum_ratings(df_ratings)


# 1.2
def pandas_pivot_table(df, idx, col, val):
    """
    # Pandas pivot_table
    :param df: DataFrame
    :param idx: str
    :param col: str
    :param val: str
    :return: DataFrame
    """
    start = time.time()                                 # start - measure time for loop
    movies_users = pd.pivot_table(data=df, index=[idx], columns=[col], values=val).fillna(0)
    mat_movies_users = csr_matrix(movies_users.values)
    end = time.time()                                    # end
    print("\nThe time of execution of pandas_pivot_table is :", end - start)
    return mat_movies_users


mat_movies_users = pandas_pivot_table(df_rating_reduced_pd, "movieId", "userId", "rating")


def scipy_coo_matrix(df, row_str, col_str, data_str):
    """
    :param df: DataFrame
    :param row_str: str
    :param col_str: str
    :param data_str: str
    :return: coo_matrix
    """
    start = time.time()                     # start - measure time for loop
    row = np.array(df[row_str])
    col = np.array(df[col_str])
    data = np.array(df[data_str])
    mat = coo_matrix((data, (row, col)))
    end = time.time()                       # end
    print("\nThe time of execution of scipy_coo_matrix is :", end - start)
    return mat


#mat = scipy_coo_matrix(df_ratings, "movieId", "userId", "rating")


# 1.3 Rekommenderarsystemet
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6)


def recommender_dataframe(data, model, n_recommendations):
    """
    :param data: DataFrame
    :param model: NearestNeighbors
    :param n_recommendations: int
    :return: str of recommendations
    """
    while True:
        try:
            print("Enter a movie title you like and get recommendations")
            movie_name = str(input("Enter a title: "))

            if len(movie_name) > 0:
                model.fit(data)
                idx = process.extractOne(movie_name, df_movies['title'])[2]
                print('Movie selected: ', df_movies['title'][idx])  # 'Index: ', idx used for testing
                print('Searching for recommendations.....')
                distance, indices = model.kneighbors(data[idx], n_neighbors=n_recommendations)
                for i in indices[0]:
                    print(df_movies['title'][i])
                print("\nWant to stop search? Push Enter! If not, just wait\n")
            else:
                print("Invalid input")
                break
        except ValueError:
            print("Invalid input")
            break


recommender_dataframe(mat_movies_users, model_knn, 6)


def recommender(data, model, n_recommendations):
    """
    :param data: coo_matrix
    :param model: NearestNeighbors
    :param n_recommendations: int
    :return: str of recommendations
    """
    while True:
        try:
            print("Enter a movie title you like and get recommendations")
            movie_name = str(input("Enter a title: "))
            if len(movie_name) > 0:
                idx = process.extractOne(movie_name, df_movies['title'])[2]
                # {tuple3}('Toy Story (1995)', 90, 0) sista är index
                data_index = data.tocsr()[idx:]
                # https://stackoverflow.com/questions/50898924/typeerror-with-accessing-to-coo-matrix-by-index
                # see comment under asked Jun 17, 2018 at 17:27, but [idx,:] is wrong it should be [idx:]
                # so it gets all values  Jun 17, 2018 at 18:02
                model.fit(data_index)
                print('Movie selected: ', df_movies['title'][idx])  # , 'Index: ', idx used for testing
                print('Searching for recommendations.....')
                distance, indices = model.kneighbors(data_index[idx], n_neighbors=n_recommendations)
                for i in indices[0]:
                    title = df_movies[(df_movies.index == i)].values
                    title_clean = (str(title).replace(' [', '').replace('[', '').replace(']', '').replace('\'','')
                                             .replace('\'',''))
                    print(f"Title: {title_clean[5:]}")
                    # Index {i} Movie Id:{test[:5]} Title: {test[5:-7]} used for testing
                print("\nWant to stop search? Push Enter! If not, just wait!\n")
            else:
                print("Invalid input")
                break
        except ValueError:
            print("Invalid input")
            break


#recommender(mat, model_knn, 6)
