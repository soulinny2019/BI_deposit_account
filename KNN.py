from sklearn.neighbors import KNeighborsClassifier
from deposit_account import dataset
from deposit_account import accuracy


dataframe = dataset.get_dataset(dataset.FILENAME)
X, y =  dataset.get_x_y(dataframe)
X_train, X_test, y_train, y_test = dataset.split_training_data(X, y)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

accuracy.accuracy(y_test, y_pred)