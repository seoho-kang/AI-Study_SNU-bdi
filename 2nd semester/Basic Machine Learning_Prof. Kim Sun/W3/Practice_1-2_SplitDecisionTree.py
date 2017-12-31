import numpy as np
from sklearn.datasets import load_iris

def gini(arr):
	classCounts = np.unique(arr, return_counts=True)[1]
	sampleNum = arr.shape[0]
	p_arr = classCounts / float(sampleNum)
	ret = 1. - (p_arr**2).sum()
	return ret

def giniSplit(arr1, arr2):
	sampleNum1 = float(arr1.shape[0])
	sampleNum2 = float(arr2.shape[0])
	totalSampleNum = sampleNum1 + sampleNum2
	return gini(arr1) * sampleNum1 / totalSampleNum + \
		   gini(arr2) * sampleNum2 / totalSampleNum

iris = load_iris()
feature = iris.data

minGiniSplit = np.inf
minSplitFeature = None
minSplitBoundary = None


for fIdx in range(feature.shape[1]):
	featureArr = feature[:,fIdx]
	uniqueSorted = np.unique(np.sort(featureArr))

	for i in range(len(uniqueSorted)-1):
		splitBoundary = (uniqueSorted[i] + uniqueSorted[i+1]) / 2.
		split1 = iris.target[featureArr <= splitBoundary]
		split2 = iris.target[featureArr > splitBoundary]
		giniSplitValue = giniSplit(split1, split2)

		#print fIdx, i, giniSplitValue

		if giniSplitValue < minGiniSplit:
			minGiniSplit = giniSplitValue
			minSplitFeature = fIdx
			minSplitBoundary = splitBoundary


print 'minSplitValue:     {}'.format(minGiniSplit)
print 'minSplitFeature:   {}'.format(iris.feature_names[minSplitFeature])
print 'minSplitBoundary:  {}'.format(minSplitBoundary)
