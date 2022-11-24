import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import constantes


def get_data_frame():
    _dataJSON = pd.read_json(constantes.ABR_PATH, lines=True, orient='records',
                             chunksize=28, nrows=20000, typ='frame')
    return pd.DataFrame(data=_dataJSON.read())


def build_scores(row, scores):
    score = np.zeros(5)

    def build_score(note):
        score[note - 1] += 1

    row['overall'].apply(build_score)

    scores.insert(len(scores.columns), row['asin'].iloc[0], score)


def build_features_from_scores(scores):
    scores_t = scores.transpose()
    score_mean_p(scores_t)
    score_nb_reviews(scores_t)
    score_med(scores.transpose(), scores_t)
    print(scores.median())


# Ajout de la moyenne ponderer a la matrice
def score_mean_p(scores_t):
    moy_p = 0
    for i in range(1, 6):
        moy_p += scores_t[i] * i
    moy_p = moy_p / scores_t.sum(axis=1)
    scores_t.insert(len(scores_t.columns), 'moy_p', moy_p)


# Ajout du nombre de review par livre
def score_nb_reviews(scores_t):
    scores_t.insert(len(scores_t.columns), 'nb_review', scores_t.sum(axis=1))


# original_scores_t c'est la mat de scores original sans les autres ajout
def score_med(original_scores_t, cumulated_scores_t):
    med =

def main():
    data_f = get_data_frame()
    data_f.info()
    reviews_filter = data_f.filter(axis=1, items=['asin', 'overall', 'reviewTime'])
    scores = pd.DataFrame([], columns=[], index=[1, 2, 3, 4, 5])
    reviews_filter.groupby(by=['asin']).apply(build_scores, scores)
    scores = scores.copy();
    print(scores)
    build_features_from_scores(scores)


if __name__ == "__main__":
    main()
