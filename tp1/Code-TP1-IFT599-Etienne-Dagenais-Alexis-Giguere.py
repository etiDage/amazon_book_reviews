#Author
# Etienne Dagenais dage1001
# Alexis Giguere giga0601

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import random as rand
import constantes

# import winsound

_dataJSON = pd.read_json(constantes.ABR_PATH, lines=True, orient='records',\
       chunksize=28, nrows=20000, typ='frame')
    #  chunksize=28, typ='frame')

dataF = pd.DataFrame(data=_dataJSON.read())

dataF.info()

print("\
a) \n\t1) Quelle est la moyenne de score de chaque livre?" )
#    (ou est-ce que le livre est en general apprecie ou pas) ?

cnt = Counter(dataF['asin'])
livres = cnt.keys()
frequences = cnt.values()
moyennes = dict().fromkeys(livres)

reviewsFiltres = dataF.filter(axis=1, items=['asin', 'overall'])
moyennes = reviewsFiltres.groupby(by=['asin']).mean()

print(moyennes, end="\n")

moyClassee = moyennes.sort_values(by='overall')

print("\n \
    2) Quels sont le(s) livre(s) le(s) mieux apprecie(s) et le(s) moins apprecie(s)?")

minOverall = moyennes.min()[0]
maxOverall = moyennes.max()[0]

minLivres = moyennes.loc[moyennes['overall'] == minOverall]
maxLivres = moyennes.loc[moyennes['overall'] == maxOverall]

print("min:\n{0}\n\n max:\n{1}".format(minLivres, maxLivres), end="\n")

print("\n \
3) Quels sont le 1er quart des livres les plus apprecies?" )
plusAppreciesLimite = moyClassee.quantile(q=0.75, axis=0, interpolation='lower')[0]

"""Ce sera ceux dont l'évaluation moyenne sera supérieur à plusAppreciesLimite,
    ce qui qui correspond au premier quartile."""

plusApprecies = moyennes.loc[moyennes['overall'] > plusAppreciesLimite]


print(plusApprecies, end="\n")

print("\n \
    4) Entre deux livres, lequel est mieux apprecie?")
"""Ce sera celui qui, en moyenne, aura le score d'appréciation le plus haut."""
elt1 = rand.choice(list(livres))
elt2 = rand.choice(list(livres))

print(elt1)

elt1moy = reviewsFiltres.loc[reviewsFiltres['asin'] == elt1]['overall'].mean()
elt2moy = reviewsFiltres.loc[reviewsFiltres['asin'] == elt2]['overall'].mean()

print("Element 1: {0}\tMoyenne: {1}\nElement 2: {2}\tMoyenne: {3}".format(elt1, elt1moy, elt2, elt2moy))
print("Le mieux apprécié: {}".format(elt1 if elt1moy > elt2moy else elt2))

# 5) Est-ce que l’utilisation des comparaisons de scores moyennes est toujours une bonne
#    facon de faire pour repondre a ces questions? Sinon, quels sont les alternatives?
"""
Non, puique cela ne prend pas compte du nombre de revues effectuées, ni du biais lié au fait que l'auteur ait pu
influencer sur le résultat de ces commentaires afin de les affecter positivement. Une alternative peut être d'utiliser
une moyenne pondérée par le nombre de revues, ou bien par le score d'utilité donné à ces commentaires (si possible).

La meilleure option serait cependant d'utiliser une moyenne pondérée avec le nombre de reviews effectués sur le livre.
Cela vient palier au fait que les livres avec peu de reviews puissent facilement dépasser ceux qui en ont davantage
"""

# 6) Faire un diagramme en moustaches (box plot) affichant les tendances (les etendus)
#    statistiques pour chacune des scores 1, 2, 3, 4 et 5 et interpreter la figure.

## Dans Qb.py pour utiliser la matrice des scores
       
############################

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
scores = pd.DataFrame([], columns=[], index=[1, 2, 3, 4, 5] )

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

cntBas = len(scores_t[scores_t['a_score'] < 2.5])
cntMil = len(scores_t[(scores_t['a_score'] >= 2.5) & (scores_t['a_score'] <= 3.5)])
cntHaut = len(scores_t[scores_t['a_score'] > 3.5])

xlabels = ['non-appréciés', 'plus-ou-moins', 'appréciés']
ylabels = [cntBas, cntMil, cntHaut]
boxPlot = pd.DataFrame({' ':xlabels,'fréquence':ylabels}).plot.bar(x=' ',y='fréquence', rot=0, color='green')


boxPlot.Color = (0.8,0.2,0.6)

plt.show()
###############################################
#-------------Section-C------------------------
_dataJSON = pd.read_json(constantes.ABR_PATH, lines=True, orient='records',\
      chunksize=28, nrows=10000, typ='frame')

dataF = pd.DataFrame(data=_dataJSON.read())

reviews = dataF.filter(axis=1, items=['asin', 'overall', 'unixReviewTime', 'reviewTime'])

scores = pd.DataFrame([], columns=[], index=[1, 2, 3, 4, 5] )

monthsReviews = list(pd.DataFrame([], columns=[], index=[1, 2, 3, 4, 5]))

year = "2012"

for m in range(1, 13):
      str_month = str(m)
      if m < 10 : str_month = "0" + str_month
      regx = str_month + " .., " + year
      monthsReviews.append(reviews[reviews['reviewTime'].str.match(regx)])

monthlyScores = []
monthScores = pd.DataFrame([], columns=[], index=[1, 2, 3, 4, 5] )

def buildScores(row):
      score = np.zeros(5)
      def buildScore(note): 
            score[note - 1] += 1
      row['overall'].apply(buildScore)
      monthScores.insert(len(scores.columns), row['asin'].iloc[0], score)


for monthReview in monthsReviews:
      monthScores = pd.DataFrame([], columns=[], index=[1, 2, 3, 4, 5] )
      monthReview.groupby(by=['asin']).apply(buildScores)
      monthlyScores.append(monthScores.transpose())


print("________________SCORE_FOR_MONTH____________________")
print(monthlyScores[0])
months_counter = 1

monthlyProjections = []


for monthScores in monthlyScores:
      ### PROJECTION
      mean_v = monthScores.mean().to_numpy()
      month_scores_cov_np = monthScores.cov().to_numpy()

      eig_values, eig_vectors = np.linalg.eig(month_scores_cov_np)
      sorted_eig_values = np.sort(eig_values)

      index_of_vp1, = np.where(np.isclose(eig_values, sorted_eig_values[-1]))[0]
      index_of_vp2, = np.where(np.isclose(eig_values, sorted_eig_values[-2]))[0]

      vp1 = eig_vectors[index_of_vp1] / np.linalg.norm(eig_vectors[index_of_vp1])
      vp2 = eig_vectors[index_of_vp2] / np.linalg.norm(eig_vectors[index_of_vp2])

      month_projection = pd.DataFrame([], columns=[], index=['x', 'y'])

      def buildProjectionDataFrame(col): 
            col_np = col.to_numpy() - mean_v
            x = np.dot(vp1, col_np)
            y = np.dot(vp2, col_np)
            month_projection.insert(len(month_projection.columns), col.name, [x, y])
      
      monthScores.apply(buildProjectionDataFrame, axis=1)

      monthlyProjections.append(month_projection.transpose())
      ### Calcul moyenne ponderer par mois
      moy_p = 0
      for i in range(1, 6):
            moy_p += monthScores[i] * i
      moy_p = moy_p / monthScores.sum(axis=1)
      monthScores.insert(len(monthScores.columns), 'moy_p', moy_p)
      print("_________________Score_for_month_" + str(months_counter) + "_______________________")
      print(monthScores) 
      ### Livre le(s) plus apprecier du mois
      minOverall = moy_p.min()
      maxOverall = moy_p.max()

      print("_____Best_books_of_month_" + str(months_counter) + "__________")
      print(monthScores.loc[monthScores['moy_p'] == maxOverall])

      print("_____Worst_books_of_month_" + str(months_counter) + "__________")
      print(monthScores.loc[monthScores['moy_p'] == minOverall])

      print("_____Best_first_quarter_of_books_of_month_" + str(months_counter) + "_________")
      firstQuarterLimit = monthScores['moy_p'].quantile(q=0.75, interpolation="lower")
      print(monthScores.loc[monthScores['moy_p'] >= firstQuarterLimit])
      
      print("____Projection_on_principal_component_for_month_" + str(months_counter) + "______________")
      print(month_projection.transpose())

      months_counter += 1