from deposit_account import dataset
from deposit_account import accuracy
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping


dataframe = dataset.get_dataset(dataset.FILENAME)
X, y =  dataset.get_x_y(dataframe)
X_train, X_test, y_train, y_test = dataset.split_training_data(X, y)

x_train_scaled = preprocessing.scale(X_train)
scaler = preprocessing.StandardScaler().fit(X_train)
x_test_scaled = scaler.transform(X_test)

model = Sequential()
model.add(Dense(64, kernel_initializer = 'normal', activation = 'relu',input_shape = (21,)))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(1))

model.compile(
   loss = 'mse',
   optimizer = RMSprop(),
   metrics = ['mean_absolute_error']
)

history = model.fit(
   x_train_scaled, y_train,
   batch_size=128,
   epochs = 500,
   verbose = 1,
   validation_split = 0.2,
   callbacks = [EarlyStopping(monitor = 'val_loss', patience = 20)]
)

score = model.evaluate(x_test_scaled, y_test, verbose = 0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

y_pred = model.predict(x_test_scaled)

