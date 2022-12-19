import pandas as pd
import numpy as np
import constantes
import pymongo

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
    tempCol = db["tempCol"]
    nb_good = 3585
    nb_neutral = 33
    nb_bad = 60
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

def main():
    build_reviewers_collection()

if __name__ == "__main__":
    main()