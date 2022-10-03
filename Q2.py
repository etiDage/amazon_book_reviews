from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import constantes

# _data = pd.read_json('P:/Cours/IFT599 - IFT799/Travaux pratiques/Données/ABR.json.gz', lines=True, chunksize=100, compression='gzip')


pd.DataFrame()

_dataJSON = pd.read_json(constantes.ABR_PATH, lines=True, orient='records',\
      chunksize=28, nrows=10000, typ='frame')
#      chunksize=28, typ='frame')

# print(_dataJSON)

# dataF = pd.DataFrame(data=_dataJSON.read(), columns=[], dtype=[str,int,str,[],str,np.float32,str,int,str])
dataF = pd.DataFrame(data=_dataJSON.read())
# print(dataF)
dataF.info()

# dataF.hist(column='overall', bins=5)


# dataF['overall'].plot(kind='hist',
#         alpha=0.8,
#         bins=5,
#         title='Impression des livres',
#         rot=45,
#         grid=True,
#         figsize=(12,8),
#         fontsize=15, 
#         color=['#A0E8AF'])

# plt.xlabel('Score')
# plt.ylabel("Fréquence")

cnt = Counter(dataF['asin'])
livres = cnt.keys()
frequences = cnt.values()
freqList = np.array(list(cnt.values()))

moyennes = dict().fromkeys(livres)

reviewsFiltres = dataF.filter(axis=1, items=['asin', 'overall'])
moyennes = reviewsFiltres.groupby(by=['asin']).mean()
test = reviewsFiltres.groupby(by=['asin']).value_counts()

scores = pd.DataFrame([], columns=[], index=[1, 2, 3, 4, 5] )

def buildScores(row):
      score = np.zeros(5)
      def buildScore(note): 
            score[note - 1] += 1
      row['overall'].apply(buildScore)

      scores.insert(len(scores.columns), row['asin'].iloc[0], score)

reviewsFiltres.groupby(by=['asin']).apply(buildScores)
print("_____________________SCORES_________________________")
print(scores)
# [ 43 20 16 56 ... ]
# q[]
# mu_score = np.zeros(5)
# def calculateMuScore(row):
#       mu_score[row.name - 1] = row.mean()
# scores.apply(calculateMuScore, axis=1)
# print("MU SCORE:")
# print(mu_score)

scores_cov = scores.cov()
print("SCORES_COV")
print(scores_cov)
# b) 
# 1.

# Construction de la matrice des scores 
# On veut:
#     | Livre 1 | Livre 2 | Livre 3 | Livre 4 |
# 1         3           4
# 2         4           3
# 3         2           2
# 4         1           8
# 5         0           12


# 2.
print(frequences)
moyNbReviews = np.mean(freqList)
Q = -moyNbReviews/np.log(1.0/2.0)
ponderation = 2.5 * (np.ones_like(freqList) - np.exp(-freqList/Q))

scores = (moyennes['overall'] * 0.5 + ponderation)

print(scores)

# BEEP! Display is ready!
frequency = 397  # Set Frequency To 2500 Hertz
duration = 250  # Set Duration To 1000 ms == 1 second

# pd.DataFrame(data=scores).plot(kind='hist',
#         alpha=0.8,
#         bins=5,
#         title='Impression des livres',
#         rot=45,
#         grid=True,
#         figsize=(12,8),
#         fontsize=15, 
#         color=['#A0E8AF'])

# plt.xlabel('Score')
# plt.ylabel("Fréquence")

cntBas = len(scores[scores < 2.5])
cntMil = len(scores[(scores > 2.5) & (scores < 3.5)])
cntHaut = len(scores[scores > 3.5])

xlabels = ['non-apprécié', 'plus-ou-moins', 'appréciés']
ylabels = [cntBas, cntMil, cntHaut]
pd.DataFrame({'xaxis':xlabels,'fréquence':ylabels}).plot.bar(x='xaxis',y='fréquence', rot=0, color='green')

plt.xlabel('')
plt.ylabel("Fréquence")
plt.Color = (0.8,0.2,0.6)

# pd.DataFrame.hist(data=pd.DataFrame(data=scores), column='overall')
plt.show()

print(max(scores))