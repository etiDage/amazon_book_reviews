from typing import overload
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import random as rand

import winsound

_dataJSON = pd.read_json('C:/Users/giga0601/Desktop/ABR.json', lines=True, orient='records',\
       chunksize=28, nrows=10000, typ='frame')
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

print("\n \
    2) Quels sont le(s) livre(s) le(s) mieux apprecie(s) et le(s) moins apprecie(s)?")

minOverall = moyennes.min()[0]
maxOverall = moyennes.max()[0]

minLivres = moyennes.loc[moyennes['overall'] == minOverall]
maxLivres = moyennes.loc[moyennes['overall'] == maxOverall]

print("min:\n{0}\n\n max:\n{1}".format(minLivres, maxLivres), end="\n") # JUSTE LE PREMIER --> Trouver dernier
# moyennes.sort_values(by=['overall'])

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

# elt1 = dict.values(livres)[ind1]
# elt2 = dict.values(livres)[ind2]
# elt1moy = moyennes[ind1]
# elt2moy = moyennes[ind2]

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


# plt1 = moyClassee.plot.box(rot=0, color='orange')
# plt2 = pd.DataFrame(maxLivres.add(minLivres)).plot.box(rot=0, color='green')


# ticks = []
# labels =[]

# for item in :
#     ticks.append(x[item])
#     labels.append(month(item+1))

# plt.xticks(ticks=ticks,labels=labels, rotation=25,fontsize=8)
       
############################

labelsVals = ['1)', '2)', '3)', '4)']
ticksVals = np.array([1,2,3,4])
# # plt.xticks(ticks=ticksVals, labels=labelsVals)

# plt1 = moyClassee.plot(kind="box", xlabel='1)', xticks=ticksVals, sharex=True)
# plt1.set_xticks(ticksVals-1)
# plt.xticks(ticks=[], labels=[])

# plt2 = maxLivres.add(minLivres).plot(kind="box", xlabel='2)', xticks=ticksVals, ax = plt1, sharex=True)
# plt2.set_xticks(ticksVals)
# plt.xticks(ticks=[], labels=[])
# # plt.xticks(ticks=ticksVals, labels=labelsVals)
# # plt2.set_xticks(['2)'])
# # plt2.set(xticks=ticks, xticklabels=labels)

# plt3 = pd.DataFrame({'overall':[elt1moy, elt2moy]}).plot(kind="box", ax = plt2, xlabel='3)', xticks=ticksVals, sharex=True)
# plt3.set_xticks(ticksVals+1)
# plt.xticks(ticks=[], labels=[])
# # plt3.set(xticks=ticks, xticklabels=labels)
# # plusApprecies.plot.box(rot=0, color='purple')

#########################################################

# dataNo2 = pd.concat(minLivres['overall'], maxLivres['overall'])
# concatMinMax = np.array(list(minLivres['overall'].values) + list(maxLivres['overall'].values))
# concatMinMaxAsin = np.array(list(minLivres['overall'].values) + list(maxLivres['overall'].values))
# print(concatMinMax)
concatMinMax = minLivres['overall'] + maxLivres['overall']
print(concatMinMax)

for li in range(len(minLivres['overall'])):
    concatMinMax.iloc[li] = minLivres['overall'].iloc[li]

for li in range(len(minLivres['overall']), len(maxLivres['overall'])):
    concatMinMax.iloc[len(minLivres['overall']) + li] = maxLivres.iloc[li]

elemRand = pd.Series([elt1moy, elt2moy], [elt1,elt2])
# elemRand.iloc[0] = elt1moy
# elemRand.iloc[1] = elt2moy
print(elemRand)

# plotData = pd.DataFrame(ydata, columns=xdata)
plotData = pd.DataFrame()
plotData['1)'] = moyennes['overall']
# plotData['2'] = pd.concat(pd.Series(minLivres['overall']), pd.Series(maxLivres['overall']))
plotData['2)'] = concatMinMax
plotData['3)'] = plusApprecies['overall']
plotData['4)'] = elemRand

plotData.info()
# plotData.plot.box()

plt4 = plotData.plot(kind='box', 
        sharex=True,
        title='Étendue des scores',
        rot=0,
        # grid=True,
        figsize=(12,8),
        fontsize=15,
        xticks=ticksVals,
        # xlabel='4)'
        )

plt4.set_xticks(ticksVals)
# plt4.set(xticks=ticks, xticklabels=labels)
        # color=['#E098AF', '#54F632', '#3285F9'])

# plusApprecies.plot(kind='box',
#         title='Étendue des livres',
#         rot=45,
#         grid=True,
#         figsize=(12,8),
#         fontsize=15,
#         ax=plt2,
#         color=['#E098AF', '#54F632', '#3285F9'])

# plt.xlabel('')
plt.ylabel("Valeur")
plt.Color = (0.2,0.8,0.6)

# plt.xticks(ticks=[0,1,2,3], labels=labelsVals)

# BEEP! Display is ready!
# frequency = 397  # Set Frequency To 2500 Hertz
# duration = 250  # Set Duration To 1000 ms == 1 second
# winsound.Beep(frequency, duration)

plt.show()