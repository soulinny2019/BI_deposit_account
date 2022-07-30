
from deposit_account import dataset
from deposit_account import accuracy
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg


dataframe = dataset.get_dataset(dataset.FILENAME)
X, y =  dataset.get_x_y(dataframe)
X_train, X_test, y_train, y_test = dataset.split_training_data(X, y)

feature = ['Open_Date', 'D1', 'C1', 'Pay First', 'open Balance', 'O1', 'B1',
       'End Balance', 'E1', 'Sex_V', 'Interest Rate', 'Account type / Months',
       'Value District', 'Dis_v', 'Edu_V', 'Emp_V', 'Mar_V', 'Age ',
       'P1', 'P2', 'P3']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X_train, y_train)
data = tree.export_graphviz(dtree, out_file=None, feature_names=feature)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('mydecisiontree.png')


img=pltimg.imread('mydecisiontree.png')
imgplot = plt.imshow(img)
# plt.show()

y_pred = dtree.predict(X_test)
accuracy.accuracy(y_test, y_pred)