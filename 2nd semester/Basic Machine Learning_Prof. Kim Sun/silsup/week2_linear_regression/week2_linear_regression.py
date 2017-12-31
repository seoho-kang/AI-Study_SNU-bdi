#################
### LR
#################
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# diabetes has two attributes: data, target
print(diabetes.data.shape)
print(diabetes.target.shape)

# diabetes consists of 442 samples 
#with 10 attributes and 1 real target value.

# Use only one feature
diabetes_X = diabetes.data[:, 2:3]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# The coefficients
print('Coefficients: \n', regr.coef_)

print("Mean squared error: %.2f"
      % np.mean((regr.predict(diabetes_X_train) 
          - diabetes_y_train) ** 2))

# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(diabetes_X_test) 
          - diabetes_y_test) ** 2))


# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), 
        color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()