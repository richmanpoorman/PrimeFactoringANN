from typing import Tuple, List
from random import randrange
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier

def readPrimeFile(filePath : str) -> List[int]:
    """
        Name       | readPrimeFile
        Parameters | filePath(str)  : The name of the file which contains a list of prime numbers
        Return     | (List[int...]) : Contains all of the prime numbers as integers that was in the file
        Purpose    | The purpose is to read the file for data that is used for training the ANN
        Use        | The program will assume that all of the numbers are on a separate line, and that they are numbers
    """
    numberList = []

    file = open(filePath)

    for number in file:
        numberList.append(int(number))
    file.close()

    return numberList

def predictMLPList(xData : np.ndarray, classifierList : List[MLPClassifier]) -> np.ndarray:
    """
        Name       | predictMLPList
        Parameters | xData(np.ndarray)                   : The input data to predict
                   | classifierList(List[MLPClassifier]) : The list of the trained classifiers for each digit
        Return     | (np.ndarray[str])                   : The digits predicted (note that they are still)
        Purpose    | Run predict on all of the different classifiers
        Use        | Expects already trained list of classifiers for each digit
    """
    answers = np.array([
        cls.predict(xData) for cls in classifierList
    ]).T
    # digitLists = answers.sum(axis = 1)
    return answers # digitLists

def scoreMLPList(xData : np.ndarray, yData : np.ndarray, classifierList : List[MLPClassifier]) -> List[float]:
    """
        Name       | scoreMLPList
        Parameters | xData(np.ndarray)                   : The input data to score
                   | yData(np.ndarray)                   : The output labels
                   | classifierList(List[MLPClassifier]) : The list of the trained classifiers for each digit
        Return     | (List[float])                       : The list of scores for each digit
        Purpose    | Get the scores of the digit
        Use        | Uses the .score for each digit
    """
    size = len(classifierList)
    scoreList = [classifierList[i].score(xData, yData[:, i]) for i in range(size)]
    return scoreList

def trainMLPList(xData : np.ndarray, yData : np.ndarray, layerSizes : Tuple = (100,), activation : str = 'relu', alpha : int = 0.001) -> List[MLPClassifier]:
    """
        Name       | trainMLPList
        Parameters | xData(np.ndarray)          : The input data to train on
                   | yData(np.ndarray)          : The output labels
                   | layerSizes(Tuple[int,...]) : (optional) The sizes of each hidden layer
                   | activation(str)            : (optional) The activation method of the perceptrons
                   | alpha(int)                 : (optional) The normalization power of L2
        Return     | (List[MLPClassifer])       : The classifier for each digit
        Purpose    | Train an ANN on the training data
        Use        | Uses the sklearn MLPClassifier 
    """

    size = yData.shape[1]
    classifiers = [
        MLPClassifier(
            hidden_layer_sizes = layerSizes, 
            solver = 'lbfgs',
            activation = activation, 
            alpha = alpha
        ).fit(xData, yData[:, i])
        for i in range(size)
    ]

    return classifiers

def trainMLP(xData : np.ndarray, yData : np.ndarray, layerSizes : Tuple = (100,), activation : str = 'relu', alpha : int = 0.001) -> MLPClassifier:
    """
        Name       | trainMLP
        Parameters | xData(np.ndarray)          : The input data to train on
                   | yData(np.ndarray)          : The output labels
                   | layerSizes(Tuple[int,...]) : (optional) The sizes of each hidden layer
                   | activation(str)            : (optional) The activation method of the perceptrons
                   | alpha(int)                 : (optional) The normalization power of L2
        Return     | (MLPClassifer)             : The trained classifier
        Purpose    | Train an ANN on the training data
        Use        | Uses the sklearn MLPClassifier [DEPRECIATED]
    """
    
    clf = MLPClassifier(
        hidden_layer_sizes = layerSizes, 
        solver = 'lbfgs',
        activation = activation, 
        alpha = alpha
    )

    clf.fit(xData, yData)
    return clf

def makePrimeDataSet(primeList : List[int], numDataPoints : int = 10000, digitCount : int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
        Name       | makeDataSet
        Parameters | primeList(List[int...]) : The list of prime numbers to choose from and multiply
                   | numDataPooints(int)     : (optional) The number of data points to generate
                   | digitCount(int)         : (optional) The number of digits to standardize to, including when they are multiplied together
        Return     | (np.ndarray)            : List of the digits of the product
                   | (np.ndarray)            : List of the digits of the two primes appended together
        Purpose    | The purpose is to choose primes to multiply in order to feed into the ANN as a single data point
        Use        | Will create a matrix of digits of a products of primes, as well as those primes appened toegher for the ANN to train/test on
    """
    dataPoints = [choosePrimesIntList(primeList, digitCount) for i in range(numDataPoints)]
    # Note the features are the digits of the product, which is put in the last spot, whereas the concatination is the class
    featureList = np.array([x[1] for x in dataPoints])
    labelList   = np.array([x[0] for x in dataPoints])
    return featureList, labelList

def choosePrimesIntList(primeList : List[int], digitCount : int = 20) -> Tuple[List[int], List[int]]:
    """
        Name       | choosePrimesIntList
        Parameters | primeList(List[int...]) : The list of prime numbers to choose from and multiply
                   | digitCount(int)         : (optional) The number of digits to standardize to, including when they are multiplied together
        Return     | (List[int])             : A list of the digits of two primes appended together
                   | (List[int])             : A list of the digits of the product of the two primes
        Purpose    | The purpose is to choose primes and format in a way such that every digit is a feature
        Use        | Will randomly pick from prime list and multiply (assuming all primes in the prime list are unique) [DEPRECIATED]
    """
    a, b, c = choosePrimes(primeList, digitCount)
    abValues = list(map(int, list(a + b)))
    cValues = list(map(int, list(c)))
    return abValues, cValues

def choosePrimes(primeList : List[int], digitCount : int = 20) -> Tuple[str, str, str]:
    """
        Name       | choosePrimes
        Parameters | primeList(List[int...]) : The list of prime numbers to choose from and multiply
                   | digitCount(int)         : (optional) The number of digits to standardize to, including when they are multiplied together
        Return     | (str)                   : The first prime as a string
                   | (str)                   : The second prime as a string
                   | (str)                   : The product as a string
        Purpose    | The purpose is to choose primes to multiply in order to feed into the ANN as a single data point
        Use        | Will randomly pick from prime list and multiply (assuming all primes in the prime list are unique), also a < b
    """
    listSize = len(primeList)
    a, b = randrange(listSize), randrange(listSize)
    while (a == b):
        b = randrange(listSize)
    c = a * b
    # Choose the lower one as a
    if (a > b):
        temp = a 
        a = b
        b = temp

    values = (intToString(a, digitCount), intToString(b, digitCount), intToString(c, digitCount))
    return values

def intToString(value : int, digitCount : int = 20) -> str:
    """
        Name       | choosePrimes
        Parameters | value(int)      : The value to convert into a string with size digitCount
                   | digitCount(int) : (optional) The number of digits to standardize to, including when they are multiplied together
        Return     | (str)           : The number with digitCount amount of digits as a string
        Purpose    | The purpose is to convert a number to a corresponding string
        Use        | Help to convert when picking out digits
    """
    valueString = str(value)
    numZeros = digitCount - len(valueString)
    if (numZeros < 0):
        raise OverflowError("More Digits than Allowed")
    valueString = '0' * numZeros + valueString
    return valueString