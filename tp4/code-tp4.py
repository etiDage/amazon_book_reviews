import pandas as pd
import numpy as np
import constantes
import pymongo
import matplotlib.pyplot as plt
from random import sample, randint

def stratified_selection():
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client['amazonReviews']
    features_col = db.features
    # Nombre de livres apprecier: 358568
    # Nombre de livres neutre: 3326
    # Nombre de livres bad: 6088
    nb_good = 3585
    nb_neutral = 33
    nb_bad = 60
    good_reviews_df = pd.DataFrame(list(features_col.find({"appreciation": "good"}).limit(nb_good)))
    good_reviews_df = good_reviews_df.filter(axis=1, items=['asin', 'med', 'mean', 'rev_nb', 'std', 'good_rev',
                                                            'neutral_rev', 'bad_rev']).set_index('asin')

    neutral_reviews_df = pd.DataFrame(list(features_col.find({"appreciation": "neutral"}).limit(nb_neutral)))
    neutral_reviews_df = neutral_reviews_df.filter(axis=1, items=['asin', 'med', 'mean', 'rev_nb', 'std', 'good_rev',
                                                                  'neutral_rev', 'bad_rev']).set_index('asin')

    bad_reviews_df = pd.DataFrame(list(features_col.find({"appreciation": "bad"}).limit(nb_bad)))

    bad_reviews_df = bad_reviews_df.filter(axis=1, items=['asin', 'med', 'mean', 'rev_nb', 'std', 'good_rev',
                                                          'neutral_rev', 'bad_rev']).set_index('asin')

    frames = [good_reviews_df, neutral_reviews_df, bad_reviews_df]
    features_df = pd.concat(frames)
    print(features_df)
    return features_df

def build_temp_collection():
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client['amazonReviews']
    goodBooksCol = db.goodBooksOrdered
    neutralBooksCol = db.neutralBooksOrdered
    badBooksCol = db.badBooksOrdered
    tempCol = db["booksObserved"]
    nb_good = 1000
    nb_neutral = 10
    nb_bad = 20
    good_books = list(goodBooksCol.find().limit(nb_good))
    neutral_books = list(neutralBooksCol.find().limit(nb_neutral))
    bad_books = list(badBooksCol.find().limit(nb_bad))
    tempCol.insert_many(good_books)
    tempCol.insert_many(neutral_books)
    tempCol.insert_many(bad_books)

def build_reviewers_collection():
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client['amazonReviews']
    books_col = db["books2"]
    temp_col = db.tempCol
    asins = temp_col.distinct("asin")
    asins_np = np.array(asins)
    splitted_asins = np.array_split(asins_np, 10)
    for asins in splitted_asins:
        books_col.aggregate([{"$match": {"asin": {"$in": list(asins)}}}, {"$out": "reviewsFromSelectedBooks"}])
    print("DONE")

def build_dataframe():
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client['amazonReviews']
    rev_n_book_col = db["reviewersAndBooks"]
    book_selected_col = db["booksObserved"]
    rev_n_books_df = pd.DataFrame(list(rev_n_book_col.find({}).limit(2000)))
    print(rev_n_books_df["_id"])
    book_list = list(book_selected_col.distinct("asin"))
    reviewer_list = rev_n_books_df["_id"]
    fr1 = pd.DataFrame([], columns=book_list, index=reviewer_list)
    rev_n_books_df.apply(insert_user_rating, axis=1, args=[fr1])

    return [fr1, rev_n_books_df, book_list]

def insert_user_rating(row, heat_map_df):
    for book in row["books_reviewed"]:
        heat_map_df.loc[row["_id"], book["asin"]] = book["overall"]

def data_selection(base_df, book_list):
    base_df_modified = base_df.apply(hide_some_reviews, axis=1)
    reviewer_list = base_df["_id"]
    fr1_hide = pd.DataFrame([], columns=book_list, index=reviewer_list).fillna(0)
    base_df_modified.apply(insert_user_rating, axis=1, args=[fr1_hide])
    return fr1_hide

def hide_some_reviews(row):
    nb_to_hide = randint(0, int(row["nb_review"]/10)) # 10% des reviews
    row["books_reviewed"] = remove_n_random_items(row["books_reviewed"], nb_to_hide)
    row["nb_review"] = len(row["books_reviewed"])
    return row


# From: https://bobbyhadz.com/blog/python-remove-random-element-from-list
def remove_n_random_items(lst, n):
    if n == 0:
        return lst
    to_delete = set(sample(range(len(lst)), n))
    return [
        item for index, item in enumerate(lst)
        if not index in to_delete
    ]

def show_heat_map(heat_map_df):
    plt.imshow(heat_map_df, cmap='hot', interpolation='nearest')
    plt.show()

def find_sim(id, sim_matrix, k_nearest):
    sim_col = sim_matrix.loc[:, id]
    sim_array = sim_col.fillna(-1).to_numpy()
    index_of_k_nearest = np.argsort(sim_array)[::-1][:k_nearest + 1]
    print("Indices:", index_of_k_nearest[1:])
    print("Values: ", sim_array[index_of_k_nearest[1:]])
    print("Ids: ", list(sim_matrix.columns[index_of_k_nearest[1:]]))
    return list(sim_matrix.columns[index_of_k_nearest[1:]])


def eval_function(df, reviewers_sim_matrix, books_sim_matrix, omega):
    acc = 0
    for reviewer in df.index:
        similar_reviewers = find_sim(reviewer, reviewers_sim_matrix, 8)
        for book in df.columns:
            similar_books = find_sim(book, books_sim_matrix, 3)
            r_i_j = df.loc[reviewer, book]
            R_result = rating(similar_books, similar_reviewers, omega, df)
            acc += abs(R_result - r_i_j)
    acc/(len(df.index)*len(df.columns))

def rating(similar_books, similar_reviewers, omega, df):
    Z = build_Z_matrix(similar_books, similar_reviewers, df)
    return np.linalg.norm(np.dot(omega, Z))

def build_Z_matrix(similar_books, similar_reviewers, df):
    base = np.zeros((5,8))
    for book in similar_books:
        for i in range(0, 8):
            rating = df.loc[similar_reviewers[i], book]
            if rating > 0:
                base[rating - 1][i] += rating
    return base


def main():

    # (a)
    fr1, rev_n_books_df, book_list = build_dataframe()
    # fr1_without_nan = fr1.copy().fillna(0)
    # show_heat_map(fr1_without_nan)

    # (b)
    fr2_1 = data_selection(rev_n_books_df, book_list)
    fr2_2 = data_selection(rev_n_books_df, book_list)
    fr2_3 = data_selection(rev_n_books_df, book_list)

    # (c)
    sim_books_fr2_1 = fr2_1.corr()
    sim_books_fr2_2 = fr2_2.corr()
    sim_books_fr2_3 = fr2_3.corr()

    sim_reviewers_fr2_1 = fr2_1.transpose().corr()
    # sim_reviewers_fr2_2 = fr2_2.transpose().corr()
    # sim_reviewers_fr2_3 = fr2_3.transpose().corr()

    show_heat_map(sim_books_fr2_1)
    show_heat_map(sim_books_fr2_2)
    show_heat_map(sim_books_fr2_3)
    show_heat_map(sim_reviewers_fr2_1)
    # show_heat_map(sim_reviewers_fr2_2)
    # show_heat_map(sim_reviewers_fr2_3)

    print("Les 3 livres les plus similaires au livre 030758836X")
    find_sim("030758836X", sim_books_fr2_1, 3)
    print("Les reviewers les plus similaires au reviewer A19DWIC1T7127Y")
    find_sim("A19DWIC1T7127Y", sim_reviewers_fr2_1, 8)

    omega = [0.3, 1, 0.2, 0.6, 0.1]



if __name__ == "__main__":
    main()