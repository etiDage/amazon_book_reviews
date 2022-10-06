from collections import Counter
from tabnanny import process_tokens
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

reviews = dataF.filter(axis=1, items=['asin', 'overall', 'unixReviewTime', 'reviewTime'])

reviewsFiltres = reviews.filter(axis=1, items=['asin', 'overall'])
moyennes = reviewsFiltres.groupby(by=['asin']).mean()
test = reviewsFiltres.groupby(by=['asin']).value_counts()

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

scores = pd.DataFrame([], columns=[], index=[1, 2, 3, 4, 5] )
# avg = pd.DataFrame([], columns=[], index=[1, 2, 3, 4, 5] )

# def buildAvgs(row):
#       score = np.zeros(5)
#       def buildAvg(note): 
#             avg[note - 1] += 1
#       row['overall'].apply(buildAvg)

#       scores.insert(len(scores.columns), row['asin'].iloc[0], score)

monthlyRevs = pd.DataFrame([], columns=[], index=[1, 2, 3, 4, 5])

def buildScoresTime(row, month, year):
      monthlyRevs = pd.DataFrame([], columns=[], index=[1, 2, 3, 4, 5])
      score = np.zeros(5)
      def buildScore(note): 
            score[note - 1] += 1

      date = str(row['reviewTime'])
      if date.startswith(month) and date.endswith(year):
            row['overall'].apply(buildScore)
            monthlyRevs.insert(len(scores.columns), row['asin'].iloc[0], score)

monthsReviews = list(pd.DataFrame([], columns=[], index=[1, 2, 3, 4, 5]))

year = "2005"

for m in range(0,12):
      reviews.groupby(by=['asin']).apply(buildScoresTime, m, year)
      monthsReviews.append(monthlyRevs)

print("_____________________SCORES_________________________")
print(monthsReviews)