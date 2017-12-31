#################
### Logistic Regression with Sklearn
#################
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, model_selection, linear_model, metrics


### define input data
# Load the iris dataset
iris = datasets.load_iris()

# iris has two attributes: data, target
print(iris.data.shape)
print(iris.target.shape)

# in order to plot data, we select first two features
iris.data = iris.data[:,:2]
# for binary classification, set label 2 as true class while others are false class
iris.target[ iris.target != 2 ] = -1
iris.target[ iris.target == 2 ] = 1
iris.target[ iris.target == -1 ] = 0


# # Split the data into training/testing sets
d_train, d_test, t_train, t_test = model_selection.train_test_split(
  iris.data, iris.target, test_size=0.3, random_state=0)


# make model
C = 1
lr_model = linear_model.LogisticRegression(C=C)

# training model
lr_model.fit(d_train, t_train)

# calculation for plotting grid
h = 0.01
x_min, x_max = d_train[:, 0].min() - .5, d_train[:, 0].max() + .5
y_min, y_max = d_train[:, 1].min() - .5, d_train[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = lr_model.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)


# plotting
fig = plt.figure()

# plot training set
subplot = fig.add_subplot(3, 1, 1)
subplot.set_title("Training data Fitting")
subplot.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

subplot.scatter(d_train[:, 0], d_train[:, 1], c=t_train, edgecolors='k', cmap=plt.cm.Paired)
subplot.set_xlabel('Sepal length')
subplot.set_ylabel('Sepal width')

subplot.set_xlim(xx.min(), xx.max())
subplot.set_ylim(yy.min(), yy.max())
subplot.set_xticks(())
subplot.set_yticks(())


# plot test set
subplot = fig.add_subplot(3, 1, 2)
subplot.set_title("Test data Fitting")
subplot.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

subplot.scatter(d_test[:, 0], d_test[:, 1], c=t_test, edgecolors='k', cmap=plt.cm.Paired)
subplot.set_xlabel('Sepal length')
subplot.set_ylabel('Sepal width')

subplot.set_xlim(xx.min(), xx.max())
subplot.set_ylim(yy.min(), yy.max())
subplot.set_xticks(())
subplot.set_yticks(())


# plot roc curve for test set
pred_test = lr_model.predict(d_test)

fpr, tpr, _ = metrics.roc_curve(y_true=t_test, y_score=pred_test)
roc_auc = metrics.auc(fpr, tpr)


subplot = fig.add_subplot(3, 1, 3)
subplot.set_title("ROC curve")
lw = 2
subplot.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
subplot.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
subplot.set_xlim([0.0, 1.0])
subplot.set_ylim([0.0, 1.05])
subplot.set_xlabel('False Positive Rate')
subplot.set_ylabel('True Positive Rate')
subplot.legend(loc="lower right")


plt.tight_layout()
plt.show()