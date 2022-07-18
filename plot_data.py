from deposit_account import dataset
import matplotlib.pyplot as plt
from sklearn import preprocessing

dataframe = dataset.get_dataset(dataset.FILENAME)
dataframe_scaled = preprocessing.scale(dataframe)
plt.plot(dataframe_scaled)
plt.savefig('normalized_dataset.png')
plt.show()