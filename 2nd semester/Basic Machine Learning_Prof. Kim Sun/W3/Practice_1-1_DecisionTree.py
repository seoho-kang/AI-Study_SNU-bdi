from sklearn.datasets import load_iris
from sklearn import tree
import graphviz

#Data load
iris = load_iris()
print(iris.data)
print(iris.target)


#Build Decision Tree Classifier Model
clf = tree.DecisionTreeClassifier()

#Fit Model to iris data
clf = clf.fit(iris.data, iris.target)

#Draw decision tree
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("iris") 
#graph.view("iris") 

#A characterized decision tree
#dot_data = tree.export_graphviz(clf, out_file=None,\
#						feature_names=iris.feature_names,\
#						class_names=iris.target_names,\
#						filled=True, rounded=True,\
#						special_characters=True)  
#graph = graphviz.Source(dot_data)  
#graph.render("iris_color") 
#graph.view("iris_color") 

#Identify prediction of the model
predict = clf.predict(iris.data)
print(predict)


#Compute correct prediction rate
numOfCorrectPrediction = (predict == iris.target).sum()  # It should be 150
numOfDataSamples = iris.target.shape[0]  # Total number of sampels. It also should be 150.

print numOfCorrectPrediction / float(numOfDataSamples)
