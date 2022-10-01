from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import winsound

# _data = pd.read_json('P:/Cours/IFT599 - IFT799/Travaux pratiques/Données/ABR.json.gz', lines=True, chunksize=100, compression='gzip')


pd.DataFrame()

_dataJSON = pd.read_json('C:/Users/giga0601/Desktop/ABR.json', lines=True, orient='records',\
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
# [ 43 20 16 56 ... ]
# q[]


# b) 
# 1.

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
winsound.Beep(frequency, duration)

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