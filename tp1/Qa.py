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
