from collections import Counter
import re
from tabnanny import process_tokens
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import constantes

_dataJSON = pd.read_json(constantes.ABR_PATH, lines=True, orient='records',\
      chunksize=28, nrows=10000, typ='frame')

dataF = pd.DataFrame(data=_dataJSON.read())
dataF.info()

cnt = Counter(dataF['asin'])
livres = cnt.keys()
frequences = cnt.values()
freqList = np.array(list(cnt.values()))

moyennes = dict().fromkeys(livres)

reviewsFiltres = dataF.filter(axis=1, items=['asin', 'overall', 'reviewTime'])
moyennes = reviewsFiltres.groupby(by=['asin']).mean()
test = reviewsFiltres.groupby(by=['asin']).value_counts()

# b) 
# 1.------------------------------------------------------------------

# Construction de la matrice des scores 
# On veut:
#     | Livre 1 | Livre 2 | Livre 3 | Livre 4 |
# 1         3           4
# 2         4           3
# 3         2           2
# 4         1           8
# 5         0           12

scores = pd.DataFrame([], columns=[], index=[1, 2, 3, 4, 5] )
# avg = pd.DataFrame([], columns=[], index=[1, 2, 3, 4, 5] )

# def buildAvgs(row):
#       score = np.zeros(5)
#       def buildAvg(note): 
#             avg[note - 1] += 1
#       row['overall'].apply(buildAvg)

#       scores.insert(len(scores.columns), row['asin'].iloc[0], score)

def buildScores(row):
      score = np.zeros(5)
      def buildScore(note): 
            score[note - 1] += 1
      row['overall'].apply(buildScore)

      scores.insert(len(scores.columns), row['asin'].iloc[0], score)

reviewsFiltres.groupby(by=['asin']).apply(buildScores)


 # a) 6.##############################
scoresBoxPlt = scores.transpose().plot(kind='box', 
        sharex=True,
        title='Étendue des scores',
        ylim=(-2,75),
        rot=0,
        # grid=True,
        figsize=(12,8),
        fontsize=15,
        color='red'
        # xlabel='4)'
        )

plt.show()
#######################################

scores_np = scores.to_numpy()
scores_cov_np = np.cov(scores_np)

eig_values, eig_vectors = np.linalg.eig(scores_cov_np)
sorted_eig_values = np.sort(eig_values)

index_of_vp1, = np.where(np.isclose(eig_values, sorted_eig_values[-1]))[0]
index_of_vp2, = np.where(np.isclose(eig_values, sorted_eig_values[-2]))[0]

vp1 = eig_vectors[index_of_vp1] / np.linalg.norm(eig_vectors[index_of_vp1])
vp2 = eig_vectors[index_of_vp2] / np.linalg.norm(eig_vectors[index_of_vp2])

projection = pd.DataFrame([], columns=[], index=['x', 'y'])

mean_v = scores.mean(axis=1).to_numpy()

def buildProjectionDataFrame(col):
      col_np = col.to_numpy() - mean_v
      x = np.dot(vp1, col_np)
      y = np.dot(vp2, col_np)
      projection.insert(len(projection.columns), col.name, [x, y])

scores.apply(buildProjectionDataFrame)


projection_t = projection.transpose()

projection_t.plot.scatter('x', 'y')


# 2. --------------------------------------------------------------------
# Moyenne pondere pour chaque livres
scores_t = scores.transpose()
moy_p = 0

for i in range(1, 6):  
      moy_p += scores_t[i] * i

moy_p = moy_p / scores.sum()

scores_t.insert(len(scores_t.columns), 'moy_p', moy_p)

### Algo from https://math.stackexchange.com/questions/942738/algorithm-to-calculate-rating-based-on-multiple-reviews-using-both-review-score
# Adapted score
nbReviewsForEachBook = scores.sum().to_numpy()
moyNbReviews = np.median(nbReviewsForEachBook)
Q = -moyNbReviews/np.log(1.0/2.0)
ponderation = 2.5 * (np.ones_like(nbReviewsForEachBook) - np.exp(-nbReviewsForEachBook/Q))
adapted_scores = (moy_p * 0.5) + ponderation 

scores_t.insert(len(scores_t.columns), 'a_score', adapted_scores)
print("--------------------------SCORES_T---------------------------")
print(scores_t)

def chooseColor(a_score):
      if a_score < 2.5:
            return "red"
      elif a_score > 3.5:
            return "green"
      else:
            return "blue"

color_scheme = scores_t['a_score'].apply(chooseColor)

projection_t.insert(len(projection_t.columns), 'color', color_scheme)

moy_color = projection_t.groupby(by=['color']).mean()

projection_t.insert(len(projection_t.columns), 'size', np.ones(len(projection_t)) * 20)

projection_t.loc[len(projection_t)] = [moy_color.loc['blue']['x'], moy_color.loc['blue']['y'], 'yellow', 50]
projection_t.loc[len(projection_t)] = [moy_color.loc['red']['x'], moy_color.loc['red']['y'], 'yellow', 50]
projection_t.loc[len(projection_t)] = [moy_color.loc['green']['x'], moy_color.loc['green']['y'], 'yellow', 50]


projection_t.plot.scatter('x', 'y', c='color', s='size')

# Histogramme --------------------------------------------------------
print(frequences)
moyNbReviews = np.mean(freqList)
Q = -moyNbReviews/np.log(1.0/2.0)
ponderation = 2.5 * (np.ones_like(freqList) - np.exp(-freqList/Q))

scores = (moyennes['overall'] * 0.5 + ponderation)

print(scores)

# BEEP! Display is ready!
frequency = 397  # Set Frequency To 2500 Hertz
duration = 250  # Set Duration To 1000 ms == 1 second

cntBas = len(scores[scores < 2.5])
cntMil = len(scores[(scores > 2.5) & (scores < 3.5)])
cntHaut = len(scores[scores > 3.5])

xlabels = ['non-appréciés', 'plus-ou-moins', 'appréciés']
ylabels = [cntBas, cntMil, cntHaut]
pd.DataFrame({'xaxis':xlabels,'fréquence':ylabels}).plot.bar(x='xaxis',y='fréquence', rot=0, color='green')

plt.xlabel('')
plt.ylabel("Fréquence")
plt.Color = (0.8,0.2,0.6)

# pd.DataFrame.hist(data=pd.DataFrame(data=scores), column='overall')
plt.show()

print(max(scores))