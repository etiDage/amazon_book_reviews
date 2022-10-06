from calendar import month
from collections import Counter
from os import RWF_APPEND
from tabnanny import process_tokens
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import constantes

pd.DataFrame()

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

for monthScores in monthlyScores:
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
      months_counter += 1