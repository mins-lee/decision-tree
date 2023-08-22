import numpy as np
import sys

# decision tree with a depth level of one
class DecisionStump(object):
    def __init__(self, splitIndex):
        """
        splitIndex (int): the index of the feature column to train data
        """
        self.splitIndex = int(splitIndex)
        self.results = {}
        self.counts = {}

    def __repr__(self):
        return f'DecisionStump(split index = {self.splitIndex})'

    def train(self, X, y):
        """
        train the model on the column selected by splitIndex
        
        X (numpy.ndarray): feature data
        y (numpy.ndarray): label data
        """
        splitCol = X[:, self.splitIndex]
        featureVals = set(splitCol)
        labelVals = set(y)

        # create an empty dict and count distinct labels for each feature value
        for val in featureVals:
            self.counts[val] = {}
            for label in labelVals:
                self.counts[val][label] = 0

        for x, y in zip(splitCol, y):
            self.counts[x][y] += 1

        # get majority vote for each feature value
        for x in featureVals:
            self.results[x] = max(self.counts[x], key=self.counts[x].get)
            
    
    def predict(self, X):
        """
        predict labels

        X (numpy.ndarray): feature data
        """
        splitCol = X[:, self.splitIndex]

        output = [self.results[x] for x in splitCol]

        return output


def writeFile(filename, output):
    """
    create a file using the output data

    output (list): list of values
    """
    with open(filename, 'wt') as f:
        f.write('\n'.join(output))


def main():
    trainIn, testIn, splitIndex, trainOut, testOut = sys.argv[1:]
    
    trainData = np.genfromtxt(trainIn, dtype=str, skip_header=1, delimiter='\t')
    testData = np.genfromtxt(testIn, dtype=str, skip_header=1, delimiter='\t')
    # create label and feature columns
    trainX = trainData[:, :-1]
    trainY = trainData[:, -1]
    testX = testData[:, :-1]
    testY = testData[:, -1]
    
    ds = DecisionStump(splitIndex)
    ds.train(trainX, trainY)
    trainOutput = ds.predict(trainX)
    testOutput = ds.predict(testX)
    writeFile(trainOut, trainOutput)
    writeFile(testOut, testOutput)


if __name__ == '__main__': 
    main()