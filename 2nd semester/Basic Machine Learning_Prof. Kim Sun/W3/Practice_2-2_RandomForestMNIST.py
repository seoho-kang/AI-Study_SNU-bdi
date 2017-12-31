import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import csv


#Load MNIST training data
train = pd.read_csv("train.csv")

#Pre-process MNIST data and split it into train, test data sets
features = train.columns[1:]
X = train[features]
y = train['label']
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X/255.,y,test_size=0.1,random_state=0)

#Train the Random Forest classifier. Fit the train data
clf_rf = RandomForestClassifier()
clf_rf.fit(X_train, y_train)

#Predict the handwritten digits in the test data
y_pred_rf = clf_rf.predict(X_test)

#Measure accuracy of the prediction
acc_rf = accuracy_score(y_test, y_pred_rf)
print "Random Forest accuracy: ",acc_rf

# Now use the whole train set to predict the test set
clf_rf = RandomForestClassifier()
clf_rf.fit(X, y)

#Predict the test data in 'test.csv' file
TestFile="test.csv"
test=pd.read_csv(TestFile)
y_pred_rf = clf_rf.predict(test)
print y_pred_rf

np.set_printoptions(linewidth=200)
with open(TestFile, 'r') as csv_file:

	for didx, data in enumerate(csv.reader(csv_file)):
		if "pixel" in data[0]: continue

		#label = data[0]
		label=y_pred_rf[didx-1]

		# The rest of columns are pixels
		pixels = data

		# Make those columns into a array of 8-bits pixels
		# This array will be of 1D with length 784
		# The pixel intensity values are integers from 0 to 255
		pixels = np.array(pixels, dtype='uint8')

		# Reshape the array into 28 x 28 array (2-dimensional array)
		pixels = pixels.reshape((28, 28))
		print pixels

		# Plot
		plt.title('Predicted as {label}'.format(label=label))
		plt.imshow(pixels, cmap='gray')
		plt.show()





