import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

import winsound


print("1) \n a) Quelle est la moyenne de score de chaque livre?" )
#    (ou est-ce que le livre est en general apprecie ou pas) ?

_dataJSON = pd.read_json('C:/Users/giga0601/Desktop/IFT599/ABR.json', lines=True, orient='records',\
       chunksize=28, nrows=10000, typ='frame')
    #  chunksize=28, typ='frame')

dataF = pd.DataFrame(data=_dataJSON.read())

cnt = Counter(dataF['asin'])
livres = cnt.keys()
frequences = cnt.values()
moyennes = dict().fromkeys(livres)

reviewsFiltres = dataF.filter(axis=1, items=['asin', 'overall'])
moyennes = reviewsFiltres.groupby(by=['asin']).mean()

print(moyennes, end="\n")
# for l in livres:
#     print(row['asin'])
#     moyennes[row['asin'].item()] += row['overall'].item()

# moyennes[:] /= frequences[:]

# dataF['overall'].plot(kind='hist',
#     alpha=0.8,
#     bins=5,
#     title='Impression des livres',
#     rot=45,
#     grid=True,
#     figsize=(12,8),
#     fontsize=15, 
#     color=['#A0E8AF'])

# plt.xlabel('Score')
# plt.ylabel("Fréquence")

moyClassee = moyennes.sort_values(by='overall')

print("2) Quels sont le(s) livre(s) le(s) mieux apprecie(s) et le(s) moins apprecie(s)?")
print(moyClassee.iloc[[0]], end="\n") # JUSTE LE PREMIER --> Trouver dernier
# moyennes.sort_values(by=['overall'])

print("3) Quels sont le 1er quart des livres les plus apprecies?")
print(moyClassee.quantile(q=0.25, axis='columns'), end="\n")

# 4) Entre deux livres, lequel est mieux apprecie?


# 5) Est-ce que l’utilisation des comparaisons de scores moyennes est toujours une bonne
#    facon de faire pour repondre a ces questions? Sinon, quels sont les alternatives?
"""
Non, puique cela ne prend pas compte du nombre de revues effectuées, ni du biais lié au fait que l'auteur ait pu
influencer sur le résultat de ces commentaires afin de les influencer positivement. Une alternative peut être d'utiliser
une moyenne pondérée par le nombre de revues, ou bien par le score d'utilité donné à ces commentaires (si possible).

La meilleure option serait cependant d'utiliser une moyenne pondérée avec le nombre de reviews effectués sur .
"""

# 6) Faire un diagramme en moustaches (box plot) affichant les tendances (les etendus)
#    statistiques pour chacune des scores 1, 2, 3, 4 et 5 et interpreter la figure.

# BEEP! Display is ready!
frequency = 397  # Set Frequency To 2500 Hertz
duration = 250  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)

plt.show()