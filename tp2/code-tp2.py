import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import constantes
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics.pairwise import pairwise_distances as pwdists
from sklearn.metrics import silhouette_score as silh_sc
from sklearn.metrics import mutual_info_score as mut_info_sc


def get_data_frame():
    _dataJSON = pd.read_json(constantes.ABR_PATH, lines=True, orient='records',
                             chunksize=28, nrows=20000, typ='frame')
    return pd.DataFrame(data=_dataJSON.read())


def build_scores(row, scores, median_vector, mean_vector, rev_nb_vector, std_vector):
    score = np.zeros(5)

    def build_score(note):
        score[note - 1] += 1

    row['overall'].apply(build_score)
    median_vector.append(row['overall'].median())
    mean_vector.append(row['overall'].mean())
    rev_nb_vector.append(len(row['overall']))
    if len(row['overall']) == 1:
        std_vector.append(0)
    else:
        std_vector.append(row['overall'].std())  # On va peut-etre avoir a check if NaN (quand nb_rev == 1)
    scores.insert(len(scores.columns), row['asin'].iloc[0], score)


def add_median_to_scores(scores, median_vector):
    scores_t = scores.transpose()
    median_vector_np = np.array(median_vector)
    scores_t.insert(len(scores_t.columns), 'med', median_vector_np.transpose())
    return scores_t


def add_features_to_scores(scores, median_vector, mean_vector, rev_nb_vector, std_vector):
    scores_t = scores.transpose()
    scores_t.insert(len(scores_t.columns), "med", median_vector);
    scores_t.insert(len(scores_t.columns), "mean", mean_vector);
    scores_t.insert(len(scores_t.columns), "rev_nb", rev_nb_vector);
    scores_t.insert(len(scores_t.columns), "std", std_vector);
    return scores_t


def add_rev_info_to_scores(scores_t):
    good_rev = scores_t[4] + scores_t[5]
    neutral_rev = scores_t[3]
    bad_rev = scores_t[1] + scores_t[2]
    scores_t.insert(len(scores_t.columns), "good_rev", good_rev)
    scores_t.insert(len(scores_t.columns), "neutral_rev", neutral_rev)
    scores_t.insert(len(scores_t.columns), "bad_rev", bad_rev)


# Ajout de la moyenne ponderer a la matrice
def score_mean_p(scores_t):
    moy_p = 0
    for i in range(1, 6):
        moy_p += scores_t[i] * i
    moy_p = moy_p / scores_t['nb_review']
    scores_t.insert(len(scores_t.columns), 'moy_p', moy_p)


# Ajout du nombre de review par livre
def score_nb_reviews(scores_t):
    review_cols = [1, 2, 3, 4, 5]
    sum = scores_t[review_cols].sum(axis=1)
    scores_t.insert(len(scores_t.columns), 'nb_review', sum)


def calculate_kmean_euclidean_dist(features_df):
    features_matrix = features_df.to_numpy()
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(features_matrix)
    return kmeans


def calculate_kmean_cosin_dist(features_df):
    features_matrix = features_df.to_numpy()
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(preprocessing.normalize(features_matrix))
    return kmeans


def kmean_elbow(features_df):
    features_matrix = features_df.to_numpy()
    inertias = []

    for i in range(1,11):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(features_matrix)
        inertias.append(kmeans.inertia_)
    plt.plot(range(1,11), inertias, marker='o')
    plt.show()


def principal_component_projection(features_df):
    features_matrix = features_df.transpose().to_numpy()
    features_cov = np.cov(features_matrix)

    eig_values, eig_vectors = np.linalg.eig(features_cov)
    sorted_eig_values = np.sort(eig_values)

    index_of_vp1, = np.where(np.isclose(eig_values, sorted_eig_values[-1]))[0]
    index_of_vp2, = np.where(np.isclose(eig_values, sorted_eig_values[-2]))[0]

    vp1 = eig_vectors[index_of_vp1] / np.linalg.norm(eig_vectors[index_of_vp1])
    vp2 = eig_vectors[index_of_vp2] / np.linalg.norm(eig_vectors[index_of_vp2])

    projection = pd.DataFrame([], columns=[], index=['x', 'y'])

    mean_v = features_df.mean().to_numpy()

    def build_projection_df(col):
        col_np = col.to_numpy() - mean_v
        x = np.dot(vp1, col_np)
        y = np.dot(vp2, col_np)
        projection.insert(len(projection.columns), col.name, [x, y])

    features_df.apply(build_projection_df, axis=1)
    projection_t = projection.transpose()
    return projection_t

def build_similarity_matrix(projection_t:pd.DataFrame, distmode='c'):
    if distmode == 'c':
        M = pwdists(projection_t, metric='cosine', n_jobs=5, force_all_finite=True)
    else:
        M = pwdists(projection_t, metric='euclidean', n_jobs=5, force_all_finite=True)
    return M

def decompose_similarity_matrix(sim_matrix:np.ndarray):
    P, D, PDT = np.linalg.svd(sim_matrix)
    # return np.transpose(PDT)
    return P*D

def do_kmeans_on_decomp_matrix(decomp_matrix:np.ndarray):
    # features_matrix = features_df.to_numpy()
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(preprocessing.normalize(decomp_matrix))
    return kmeans

def do_2d_segm(projection_t, useCosine:bool=True):
    sim_mat = build_similarity_matrix(projection_t, 'c' if useCosine else 'e')
    return sim_mat, do_kmeans_on_decomp_matrix(decompose_similarity_matrix(sim_mat))

def main():
    data_f = get_data_frame()
    data_f.info()
    reviews_filter = data_f.filter(axis=1, items=['asin', 'overall', 'reviewTime'])
    scores = pd.DataFrame([], columns=[], index=[1, 2, 3, 4, 5])
    median_vector = []
    mean_vector = []
    rev_nb_vector = []
    std_vector = []
    reviews_filter.groupby(by=['asin']).apply(build_scores, scores, median_vector, mean_vector, rev_nb_vector,
                                              std_vector)
    scores = scores.copy()

    print("MEAN_V")
    print(scores.mean(axis=1).to_numpy())

    scores_t = add_features_to_scores(scores, median_vector, mean_vector, rev_nb_vector, std_vector)
    add_rev_info_to_scores(scores_t)
    features_df = scores_t.filter(axis=1, items=['med', 'mean', 'rev_nb', 'std', 'good_rev', 'neutral_rev', 'bad_rev'])
    kmean_euclidean = calculate_kmean_euclidean_dist(features_df)
    kmean_cosin = calculate_kmean_cosin_dist(features_df)
    # kmean_elbow(features_df)
    print(features_df.mean().to_numpy())
    projection_t = principal_component_projection(features_df)
    print(projection_t)
    
    M_euclid, sim_eucl_kmeans = do_2d_segm(projection_t, False)
    M_cos, sim_cos_kmeans = do_2d_segm(projection_t, True)

    print(M_cos)
    print(M_euclid)

    plt.scatter(projection_t['x'], projection_t['y'], c=kmean_euclidean.labels_)
    plt.show()
    plt.scatter(projection_t['x'], projection_t['y'], c=kmean_cosin.labels_)
    plt.show()

    plt.scatter(M_euclid[:,0], M_euclid[:,1], c=sim_eucl_kmeans.labels_)
    plt.show()
    plt.scatter(M_cos[:,0], M_cos[:,1], c=sim_cos_kmeans.labels_)
    plt.show()

    km_eucl_sil = silh_sc(projection_t, kmean_euclidean.labels_)
    km_eucl_mut = mut_info_sc(sim_cos_kmeans.labels_, kmean_euclidean.labels_)
    km_cos_sil = silh_sc(projection_t, kmean_cosin.labels_)
    km_cos_mut = mut_info_sc(sim_cos_kmeans.labels_, kmean_cosin.labels_)

    spec_eucl_sil = silh_sc(M_euclid[:,0:1], sim_eucl_kmeans.labels_)
    spec_eucl_mut = mut_info_sc(sim_cos_kmeans.labels_, sim_eucl_kmeans.labels_)
    spec_cos_sil = silh_sc(M_cos[:,0:1], sim_cos_kmeans.labels_)
    spec_cos_mut = mut_info_sc( sim_cos_kmeans.labels_, sim_cos_kmeans.labels_)

    print(f"DIST EUCLIDIENNE:\n  Silhouet.   Info Mut.\
\n╔═══════════╦══════════╗\n║ {km_eucl_sil} ║ {km_eucl_mut} ║ \
╠═══════════╬══════════╣\n║ {spec_eucl_sil} ║ {spec_eucl_mut} ║\
\n╚═══════════╩══════════╝\n\n\
COSINUS:\n  Silhouet.   Info Mut.\n\
╔═══════════╦══════════╗\n║ {km_cos_sil} ║ {km_cos_mut} ║\
\n╠═══════════╬══════════╣\n║ {spec_cos_sil} ║ {spec_cos_mut} ║\
\n╚═══════════╩══════════╝")

if __name__ == "__main__":
    main()
