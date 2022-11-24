import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import winsound

# _data = pd.read_json('P:/Cours/IFT599 - IFT799/Travaux pratiques/Données/ABR.json.gz', lines=True, chunksize=100, compression='gzip')


pd.DataFrame()

_dataJSON = pd.read_json('C:/Users/giga0601/Desktop/IFT599/ABR.json', lines=True, orient='records',\
     chunksize=28, typ='frame')
    #  chunksize=28, nrows=100, typ='frame')

# print(_dataJSON)

# dataF = pd.DataFrame(data=_dataJSON.read(), columns=[], dtype=[str,int,str,[],str,np.float32,str,int,str])
dataF = pd.DataFrame(data=_dataJSON.read())
# print(dataF)
dataF.info()

# dataF.hist(column='overall', bins=5)


dataF['overall'].plot(kind='hist',
        alpha=0.8,
        bins=5,
        title='Impression des livres',
        rot=45,
        grid=True,
        figsize=(12,8),
        fontsize=15, 
        color=['#A0E8AF'])

plt.xlabel('Score')
plt.ylabel("Fréquence")

# BEEP! Display is ready!
frequency = 397  # Set Frequency To 2500 Hertz
duration = 250  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)

plt.show()
# _data.plot(kind='hist',
#         alpha=0.8,
#         bins=30,
#         title='Impression des livres',
#         rot=45,
#         grid=True,
#         figsize=(12,8),
#         fontsize=15, 
#         color=['#A0E8AF'])

# plt.xlabel('Score')
# plt.ylabel("Fréquence")

# pd.DataFrame.hist(data=_data, column='overall')
