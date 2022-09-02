import random
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os
from datetime import datetime
import time
from sklearn.metrics import confusion_matrix
import string
import re


def getDataDict(labelsFileName, imagesFileName):
    '''
    Given a txt file of labels and a txt file of the corresponding images, return a dictionary 
    allData = {rowNum: {'label': intLabel, 'image': 'string of image values'}}
    '''
    allData = {}

    rowNum = 0
    lblFile = open(labelsFileName, 'r')
    labels = {}
    for row in lblFile:
        label = int(row.split()[0])
        labels[rowNum] = label
        rowNum += 1
    lblFile.close()

    rowNum = 0
    imgsFile = open(imagesFileName, 'r')
    for row in imgsFile.readlines():
        allData[rowNum] = {'label': labels[rowNum], 'image': row}
        rowNum += 1
    imgsFile.close()

    return allData


def filterData(allData, labelsNeeded):
    filteredData = {}
    for rowNum in allData.keys():
        if allData[rowNum]['label'] in labelsNeeded:
            filteredData[rowNum] = allData[rowNum]
    return filteredData


def writeImagesByLabel(dataDict, filesDict):
    for rowNum in dataDict.keys():
        label = dataDict[rowNum]['label']
        image = dataDict[rowNum]['image']
        try:
            filesDict[label].write(image)
        except KeyError:
            print('Label has no corresponding file')
            continue
    

def storeSet(dataSet, labelsInSet, numOfEachLabel, fileName, randomize=True):
    '''
    takes in a dataSet, filters it by label, writes out results to a new file
    
    '''
    newSet = {} # a dictionary of {lineNum: {'label': int, 'image': 'string of img values'}}
    lineCount = 0
    labelCounts = {}
    for lbl in labelsInSet:
        labelCounts[lbl] = 0

    # possibleLines is the range of total number of lines in the set
    possibleLines = list(range(numOfEachLabel * len(labelsInSet)))
    if randomize:
        # shuffle the line numbers so they're in random order
        random.shuffle(possibleLines)

    for item in dataSet.items():
        label = item[1]['label']
        # save item in newSet if it's a needed label
        if label in labelsInSet:
            # only save the item if we don't have too many of that item already
            if labelCounts[label] < numOfEachLabel:
                # store item's value (a dictionary of label and image) at the possibleLine at index lineCount
                newSet[possibleLines[lineCount]] = item[1]
                lineCount += 1
                # update how many times we've found an image with this label
                labelCounts[label] += 1
    
    # write out the images in newSet to the given fileName
    f = open(fileName, 'w+')
    for rowNum in sorted(newSet.keys()):
        f.write(newSet[rowNum]['image'])
    f.close()
    return newSet


def displayImage(img, numPixels, show=False, save=False, fileName=''):
    sideLength = math.floor(math.sqrt(numPixels))
    if type(img) == str:
        imgValues = np.mat([float(value) for value in img.split()])
    else:
        imgValues = np.mat(img)
    imgValues = np.transpose(imgValues[0,:pow(sideLength, 2)].reshape((sideLength, sideLength)))
    fig, ax = plt.subplots()
    ax.imshow(imgValues, cmap='gray', interpolation='nearest')
    if show: plt.show()
    if save:
        if not os.path.isdir('weights'): os.mkdir('weights')
        plt.savefig(f'{fileName}')

    plt.close()
            
    return imgValues


def trainAndTestPerceptron(trainSet, testSet, weights, numEpochs = 40, eta = 0.01):
    trainFracs = []
    testFracs = []

    # test the training set to get initial error fraction (before any training)
    initTrainOutputs = test(trainSet, weights)
    trainFracs.append(calcErrorFraction(initTrainOutputs['results'], initTrainOutputs['desired']))

    # begin epochs
    for epochCount in range(numEpochs):
        trainOutputs = {'desired': [], 'results': []}

        # get test results
        testOutputs = test(testSet, weights) # testOutputs = {'desired': [...], 'results': [...]}
        
        # calculate initial stats of test set only on first epoch (before training)
        if epochCount == 0:
            initStats = calcStats(testOutputs)
            initPrecision = calcPrecision(initStats)
            initRecall = calcRecall(initStats)
            initF1 = calcF1(initPrecision, initRecall)

        # store error fraction for test set
        testFracs.append(calcErrorFraction(testOutputs['desired'], testOutputs['results']))

        # begin training -- iterate through each image in dataSet
        for t in range(len(trainSet)): 
            image = trainSet[t]['image'].split()

            # store desired outputs for each epoch (for safety)
            trainOutputs['desired'].append(trainSet[t]['label'])

            # netInput starts as the bias weight * 1 (the bias input)
            netInput = weights[-1] # bias weight is weights[-1]

            # iterate through each pixel to calculate net input for the image
            for j in range(784):
                x = float(image[j])
                netInput += x * weights[j]

            # determine and store output
            if netInput > 0: output = 1 
            else: output = 0
            trainOutputs['results'].append(output)

            # now adjust first 784 weights
            for j in range(784):
                x = float(image[j])
                weights[j] = weights[j] + (eta * x * (trainOutputs['desired'][t] - output))

            # then adjust bias weight
            weights[-1] = weights[-1] + (eta * (trainOutputs['desired'][t] - output))

        # calculate and store error fraction for training set    
        trainFracs.append(calcErrorFraction(trainOutputs['results'], trainOutputs['desired']))
    
    # calculate final stats for test set after training
    testOutputs = test(testSet, weights) 
    testFracs.append(calcErrorFraction(testOutputs['desired'], testOutputs['results']))

    finalStats = calcStats(testOutputs)
    finalPrecision = calcPrecision(finalStats)
    finalRecall = calcRecall(finalStats)
    finalF1 = calcF1(finalPrecision, finalRecall)

    plotErrorFractions(trainFracs, testFracs, numEpochs, show=True)
    plotStats([initPrecision, initRecall, initF1], [finalPrecision, finalRecall, finalF1], show=True)


    return weights 
                

def calcErrorFraction(numWrong, numInputsTested):
    return float(numWrong) / float(numInputsTested)


def plotErrorFractions(fracs1, fracs2, title='', show=False, save=True, filename=''):
    fig, ax = plt.subplots()
    ax.plot([x * 10 for x in range(len(fracs1))], fracs1, label='Case I (LMS)')
    ax.plot([x * 10 for x in range(len(fracs1))], fracs2, label='Case II (Backpropagation)')
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error Fraction')
    ax.set_xticks([x * 10 for x in range(len(fracs1))])
    ax.set_yticks([y * 0.1 for y in range(11)])
    fig.suptitle(title)
    if show: fig.show()
    if save:
        if not os.path.isdir('figures'):
            os.makedirs('figures')
        fig.savefig(f'figures\\{filename}')
    

def plotLossTimeSeries(trainLosses, show=False):
    fig, ax = plt.subplots()
    ax.plot([x * 5 for x in range(len(trainLosses))], trainLosses)
    ax.legend()
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Mean Squared Error', fontsize=14)
    # ax.set_xticks([x * 5 for x in range(len(trainLosses))])
    fig.suptitle('Figure 3: Loss of Training Set vs Epoch', fontsize=18)
    if show: fig.show()
    if not os.path.isdir('figures'):
        os.makedirs('figures')
    fig.savefig('figures\\Figure_3.png')
    

def plotFinalLosses(trainLoss, testLoss, title='', show=False, save=True, filename=''):
    fig, ax = plt.subplots()
    labels = ['Training Loss', 'Test Loss']
    ax.bar([0.0, 0.25], [trainLoss, testLoss], width=0.15, tick_label=labels)
    fig.suptitle(title, fontsize=16)
    ax.set_xlabel('Dataset', fontsize=14)
    ax.set_ylabel('Mean Squared Error', fontsize=14)
    plt.tight_layout()
    if show: plt.show()
    if save:
        if not os.path.isdir('figures'):
            os.makedirs('figures')
        fig.savefig(f'figures\\{filename}')


def plotLossesByDigit(trainLosses, testLosses, title='', show=False, save=True, filename=''):
    fig, ax = plt.subplots()
    ax.bar(np.arange(10) - 0.125, trainLosses.values(), width=0.25, label='Training Set')
    ax.bar(np.arange(10) + 0.125, testLosses.values(), width=0.25, label='Test Set')
    fig.suptitle(title, fontsize=16)
    ax.set_xlabel('Digit', fontsize=14)
    ax.set_xticks(range(10))
    ax.set_ylabel('Mean Squared Error', fontsize=14)
    ax.legend()
    plt.tight_layout()
    if show: plt.show()
    if save:
        if not os.path.isdir('figures'):
            os.makedirs('figures')
        fig.savefig(f'figures\\{filename}')


def plotHiddenNeurons(hiddenNeurons, show=False):
    fig, axes = plt.subplots(4, 5)
    count = 0
    for i in range(4):
        for j in range(5):
            imgValues = np.mat(hiddenNeurons[count])
            imgValues = np.transpose(imgValues[0,:pow(28, 2)].reshape((28, 28)))
            axes[i][j].imshow(imgValues, cmap='gray', interpolation='none')
            axes[i][j].xaxis.set_visible(False)
            axes[i][j].yaxis.set_visible(False)
            axes[i][j].set_title(f'({LETTERS[count]})')
            count += 1
    
    fig.suptitle('Figure 4: Weights of 20 Random Hidden Neurons', fontsize=18)
    plt.tight_layout()
    if show: plt.show()
    if not os.path.isdir('figures'): 
        os.mkdir('figures')
    plt.savefig('figures\\Figure_4')


def plotInputsAndOutputs(inputImages, outputImages, show=False):
    fig, axes = plt.subplots(2, 8)
    for i in range(8):
        inputValues = np.mat(inputImages[i])
        inputValues = np.transpose(inputValues[0,:pow(28, 2)].reshape((28, 28)))
        axes[0][i].imshow(inputValues, cmap='gray', interpolation='none')
        axes[0][i].xaxis.set_visible(False)
        axes[0][i].yaxis.set_visible(False)
        axes[0][i].set_title(f'({LETTERS[i]}1)')

        outputValues = np.mat(outputImages[i])
        outputValues = np.transpose(outputValues[0,:pow(28, 2)].reshape((28, 28)))
        axes[1][i].imshow(outputValues, cmap='gray', interpolation='none')
        axes[1][i].xaxis.set_visible(False)
        axes[1][i].yaxis.set_visible(False)
        axes[1][i].set_title(f'({LETTERS[i]}2)')

    fig.suptitle('Figure 6: MNIST Data and Network Output of 8 Random Images', fontsize=14)
    plt.tight_layout()
    if show: plt.show()
    if not os.path.isdir('figures'): 
        os.mkdir('figures')
    plt.savefig('figures\\Figure_6')


def test(dataSet, weights):
    outputs = {'desired': [], 'results': []}
    for t in range(len(dataSet)):
        image = dataSet[t]['image'].split()
        # netInput starts with just the bias
        netInput = weights[-1]
        for j in range(len(image)):
            x = float(image[j])
            netInput += x * weights[j]
        if netInput > 0:
            output = 1
        else:
            output = 0
        outputs['desired'].append(dataSet[t]['label'])
        outputs['results'].append(output)

    return outputs
    

def calcStats(outputs):
    stats = {'true pos': 0, 'true neg': 0, 'false pos': 0, 'false neg': 0}
    for i in range(len(outputs['results'])):
        if outputs['results'][i] == 1:
            if outputs['desired'][i] == outputs['results'][i]:
                # true positive
                stats['true pos'] += 1
            else:
                # false positive
                stats['false pos'] += 1
        if outputs['results'][i] == 0:
            if outputs['desired'][i] == outputs['results'][i]:
                # true negative
                stats['true neg'] += 1
            else:
                # false negative
                stats['false neg'] += 1

    return stats


def calcPrecision(stats):
    return stats['true pos'] / (stats['true pos'] + stats['false pos'])


def calcRecall(stats):
    return stats['true pos'] / (stats['true pos'] + stats['false neg'])


def calcF1(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


def plotStats(initStats, finalStats, show=False):
    x = np.arange(3)
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x-0.2, initStats, width)
    ax.bar(x+0.2, finalStats, width)
    ax.set_xlabel('Statistic')
    ax.set_ylabel('Proportional Result')
    plt.xticks(x, ['Precision', 'Recall', 'F1 Score'], fontsize=10)
    ax.legend(['Before', 'After'], loc='upper left')
    fig.suptitle('Figure 1.2: Precision, Recall, and F1 Score Before and After Training')
    if show: fig.show()
    fig.savefig('figures\\Figure1_2.png')


def challengeStats(outputs):
    possibleValues = list(set(outputs['desired']))
    val1Pos = str(possibleValues[0]) + ' pos'
    val1Neg = str(possibleValues[0]) + ' neg'
    val2Pos = str(possibleValues[1]) + ' pos'
    val2Neg = str(possibleValues[1]) + ' neg'

    stats = {val1Pos: 0, val1Neg: 0, val2Pos: 0, val2Neg: 0}
    for i in range(len(outputs['results'])):
        if outputs['desired'][i] == possibleValues[0]:
            if outputs['results'][i] == 1:
                stats[val1Pos] += 1
            else:
                stats[val1Neg] += 1
        if outputs['desired'][i] == possibleValues[1]:
            if outputs['results'][i] == 1:
                stats[val2Pos] += 1
            else:
                stats[val2Neg] += 1
    return stats


def calculateTotalLoss(dataSet, hiddenLayers, outputWeights):

    totalLoss = 0
    
    for q in dataSet.keys():
        start = time.time()
        hiddenNetInputs = {}
        actualOutputs = []
        outputNetInputs = []

        # get input vector for input pattern q
        inputs = [float(x) for x in dataSet[q]['image'].split()]
        desiredOutputs = inputs # autoencoder specific -- trying to replicate inputs with the outputs

        # calculate outputs of hidden neurons
        hiddenOutputs = {}
        for h in range(len(hiddenLayers)):
            hiddenNetInputs[h] = []
            for j in range(len(hiddenLayers[h])): # for each hidden neuron
                if h == 0:
                    hiddenNetInputs[h].append(calcNetInput(inputs, hiddenLayers[h][j])) # pass in row of weight matrix (so, all weights for one hidden neuron) and all inputs from image q
                else:
                    hiddenNetInputs[h].append(calcNetInput(hiddenOutputs[h-1], hiddenLayers[h][j])) # pass in row of weight matrix (so, all weights for one hidden neuron) and all inputs from image q
            hiddenOutputs[h] = list(map(hiddenActivationFunc, hiddenNetInputs[h])) # store hidden neuron outputs

        # calculate actual outputs
        for i in range(len(outputWeights)):
            outputNetInputs.append(calcNetInput(hiddenOutputs[len(hiddenOutputs) - 1], outputWeights[i])) # get each output net input
        actualOutputs = list(map(outputActivationFunc, outputNetInputs)) # apply the activation function to each net input

        loss = 0.5 * sum(np.square(np.subtract(desiredOutputs, actualOutputs)))
        totalLoss += loss

        stop = time.time()
        print(f'test input {q}: {stop - start} seconds')

    return totalLoss
     

def hiddenActivationFunc(s_kq):
    return 1 / (1 + math.exp(-1 * s_kq)) # using sigmoid activation function


def hiddenActivationFuncMatr(netInputs):
    return np.divide(1, np.add(1, np.exp(np.negative(netInputs)))) # using sigmoid activation function


def outputActivationFunc(s_jq):
    return 1 / (1 + math.exp(-1 * s_jq)) # using sigmoid activation function


def outputActivationFuncMatr(netInputs):
    return np.divide(1, np.add(1, np.exp(np.negative(netInputs))))  # using sigmoid activation function


def calcNetInput(inputs, weights): 
    weightedInputs = np.dot(np.matrix(inputs), np.delete(weights, -1, axis=0))[0,0] 
    return weightedInputs + weights[-1]


def calcNetInputMatr(inputs, weights):
    biases = weights[:,-1]
    weightedInputs = np.dot(np.matrix(inputs), np.delete(np.transpose(weights), -1, axis=0))
    return np.add(weightedInputs, biases)


def calcOutputDelta(actualOutput, desiredOutput): 
    '''
    Takes actual and desired outputs, and calculates the weighted error (delta) for the current neuron.
    For an output neuron, error = actual * (1-actual) * (desired-actual)
    '''
    return actualOutput * (1 - actualOutput) * (desiredOutput - actualOutput)


def calcOutputDeltaMatr(actualOutputs, desiredOutputs):
    return np.multiply(np.multiply(actualOutputs, np.subtract(1, actualOutputs)), np.subtract(desiredOutputs, actualOutputs))


def calcHiddenDelta(hiddenOutput, weights, errors):
    '''
    Takes actual and desired outputs, a vector of weights for the current hidden neuron, and a vector of errors for the neurons in the above layer.
    For a hidden neuron, error = actual * (1-actual) * (desired-actual) 
    '''
    return (hiddenOutput * (1 - hiddenOutput) * np.dot(np.matrix(errors), weights))[0,0]


def calcHiddenDeltaMatr(hiddenOutputs, weights, errors):
    return np.multiply(np.multiply(hiddenOutputs, np.subtract(1, hiddenOutputs)), np.dot(errors, np.delete(weights, -1, axis=1)))


def calcWeightChanges(errors, inputs, weights, prevWeightChanges, eta, alpha):
    '''
    Takes in a vector of errors, a vector of inputs to the synapse, a matrix of weights, 
    a matrix of previous weight changes, a learning rate eta, and a momentum constant alpha.
    
    Calculates the weight changes and updates the given weights based on these changes.

    Returns a tuple of (updated weights, weight changes). 
    '''

    ### calculate all non-bias weight changes ###
    momentumMatrix = alpha * np.delete(prevWeightChanges, -1, axis=1) # calculate entire momentum matrix for non-bias weight changes
    # changeMatrix =  eta * np.dot(np.transpose(np.matrix(inputs)), errors) # calculate entire change matrix for non-bias weight changes
    changeMatrix =  eta * np.dot(np.transpose(errors), np.matrix(inputs)) # calculate entire change matrix for non-bias weight changes
    weightChanges = np.asarray(np.add(momentumMatrix, changeMatrix)) # add momentumMatrix and changeMatrix (much faster than adding each element individually)


    ### calculate all bias weight changes ###
    momentumBiasVector = np.transpose(np.matrix(alpha * prevWeightChanges[:,-1])) # calculate momentum vector for all bias weights
    biasChangeVector = eta * np.transpose(np.matrix(errors)) # calculate change vector for all bias weights
    biasWeightChanges = np.add(momentumBiasVector, biasChangeVector) # add vectors to get resulting weight changes for all bias weights

    # append bias weight change column vector as last column in weight change matrix
    allWeightChanges = np.asarray(np.append(weightChanges, biasWeightChanges, axis=1)) 

    # calculate the new weights by adding the weight changes to the old weights (including bias weights)
    weights = np.asarray(np.add(weights, allWeightChanges)) # TESTME
    
    return (weights, allWeightChanges)


def trainLMSClassifier(dataSet, hiddenLayers, outputWeights, eta=0.01, alpha=0.2):
    '''
    Takes in a dictionary, dataSet, with randomly ordered integer keys, and values {'label': 0-9, 'image': 'xxxx...'}, where each x is a float
    Also takes in a dictionary of weight matrices for all hiddenLayers, a matrix of outputWeights, a learning rate, eta, and a momentum rate, alpha. 

    Trains only the output weights, leaves the hidden weights alone (they are initialized from the autoencoder) 

    Returns the updated hiddenLayers, outputWeights, errorFrac, and totalLoss calculated for this epoch of input patterns.
    '''
    prevOutputWeightChanges = np.zeros(outputWeights.shape)
    inputsDict = {q: [float(x) for x in dataSet[q]['image'].split()] for q in range(len(dataSet.keys()))}

    numWrong = 0
    totalLoss = 0

    for q in range(len(inputsDict.keys())): # for each input pattern
        start = time.time()

        hiddenNetInputs = {}
        hiddenOutputs = {}

        outputNetInputs = []
        actualOutputs = []
        desiredOutputs = [1 if (int(dataSet[q]['label']) == r) else 0 for r in range(10)]

        outputErrors = []

        #### carrying input signal forwards ####

        # carry input signal through the hidden layers
        for h in range(len(hiddenLayers)):
            hiddenNetInputs[h] = []
            if h == 0:
                hiddenNetInputs[h] = calcNetInputMatr(inputsDict[q], hiddenLayers[h])
            else:
                hiddenNetInputs[h] = calcNetInputMatr(hiddenOutputs[h-1], hiddenLayers[h])
            
            hiddenOutputs[h] = hiddenActivationFuncMatr(hiddenNetInputs[h])

        # feed the input signal to the output layer
        outputNetInputs = calcNetInputMatr(hiddenOutputs[max(hiddenOutputs.keys())], outputWeights) # get each output net input
        
        # calculate actual outputs
        actualOutputs = outputActivationFuncMatr(outputNetInputs) # apply the activation function to each net input
        actualOutputs = actualOutputs.tolist()[0] # actual outputs is returned as a matrix, need it as a regular list

        #### calculating errors ####

        # calculate the error for each output neuron
        loss = 0.5 * sum(np.square(np.subtract(actualOutputs, desiredOutputs)))
        totalLoss += loss

        outputErrors = calcOutputDeltaMatr(actualOutputs, desiredOutputs)

        # if any of the output neuron's actualOutput was within the correct threshold, change the error to 0 and don't change those weights
        for i in range(len(outputErrors)): # for each output neuron 
            if desiredOutputs[i] == 1 and actualOutputs[i] > 0.75:
                outputErrors[i] = 0.0
                prevOutputWeightChanges[i] = np.zeros(prevOutputWeightChanges[i].shape)
            
            elif desiredOutputs[i] == 0 and actualOutputs[i] < 0.25:
                outputErrors[i] = 0.0
                prevOutputWeightChanges[i] = np.zeros(prevOutputWeightChanges[i].shape)

        # change output weights, store the weight changes for future use (because of momentum)
        outputWeights, prevOutputWeightChanges = calcWeightChanges(np.asmatrix(outputErrors), hiddenOutputs[len(hiddenOutputs) - 1], outputWeights, prevOutputWeightChanges, eta, alpha)
            
        # record how many training inputs are classified wrong
        if dataSet[q]['label'] != actualOutputs.index(max(actualOutputs)):
            numWrong += 1

        stop = time.time()
        print(f"input {q}: {stop - start} seconds | predicted: {actualOutputs.index(max(actualOutputs))} | actual: {dataSet[q]['label']}")
    
    errorFrac = calcErrorFraction(numWrong, len(dataSet))
    averageLoss = totalLoss / len(dataSet)
            
    return (hiddenLayers, outputWeights, errorFrac, averageLoss)


def trainBackpropAutoencoder(dataSet, hiddenLayers, outputWeights, eta=0.01, alpha=0.2): 
    '''
    Takes in a dictionary, dataSet, with randomly ordered integer keys, and values {'label': 0-9, 'image': 'xxxx...'}, where each x is a float
    Also takes in a dictionary of weight matrices for all hiddenLayers, a matrix of outputWeights, a learning rate, eta, and a momentum rate, alpha. 
    
    Returns the updated hiddenLayers, outputWeights, errorFrac, and averageLoss calculated for this epoch of input patterns.
    '''
    prevOutputWeightChanges = np.zeros(outputWeights.shape)
    prevHiddenWeightChanges = {h: np.zeros(hiddenLayers[h].shape) for h in range(len(hiddenLayers))}
    numWrong = 0
    totalLoss = 0

    inputsDict = {q: [float(x) for x in dataSet[q]['image'].split()] for q in range(len(dataSet.keys()))}

    for q in range(len(dataSet.keys())): # for each input pattern
        start = time.time()

        hiddenNetInputs = {}
        hiddenOutputs = {}

        outputNetInputs = []
        actualOutputs = []

        outputErrors = []

        hiddenErrors = {}

        # carrying input signal forwards
        for h in range(len(hiddenLayers)):
            hiddenNetInputs[h] = []
            if h == 0:
                hiddenNetInputs[h] = calcNetInputMatr(inputsDict[q], hiddenLayers[h])
            else:
                hiddenNetInputs[h] = calcNetInputMatr(hiddenOutputs[h-1], hiddenLayers[h])
            
            hiddenOutputs[h] = hiddenActivationFuncMatr(hiddenNetInputs[h])

        outputNetInputs = calcNetInputMatr(hiddenOutputs[max(hiddenOutputs.keys())], outputWeights) # get each output net input
                
        # calculate actual outputs
        actualOutputs = outputActivationFuncMatr(outputNetInputs) # apply the activation function to each net input

        loss = 0.5 * sum(np.asarray(np.square(np.subtract(actualOutputs, inputsDict[q]))).tolist()[0])
        totalLoss += loss

        outputErrors = calcOutputDeltaMatr(actualOutputs, inputsDict[q])

        # change output weights, store the weight changes for future use (because of momentum)
        outputWeights, prevOutputWeightChanges = calcWeightChanges(outputErrors, hiddenOutputs[len(hiddenOutputs) - 1], outputWeights, prevOutputWeightChanges, eta, alpha)
        
        for h in range(len(hiddenLayers) - 1, -1, -1):
            hiddenErrors[h] = calcHiddenDeltaMatr(hiddenOutputs[h], outputWeights, outputErrors)                 

            # change hidden weights, store the weight changes for future use (because of momentum) 
            if h == 0: 
                hiddenLayers[h], prevHiddenWeightChanges[h] = calcWeightChanges(hiddenErrors[h], inputsDict[q], hiddenLayers[h], prevHiddenWeightChanges[h], eta, alpha)
            else:
                hiddenLayers[h], prevHiddenWeightChanges[h] = calcWeightChanges(hiddenErrors[h], hiddenOutputs[h-1], hiddenLayers[h], prevHiddenWeightChanges[h], eta, alpha)
                
        # record how many training inputs are classified wrong
        actualOutputs = actualOutputs.tolist()[0]
        if dataSet[q]['label'] != actualOutputs.index(max(actualOutputs)):
            numWrong += 1

        stop = time.time()
        print(f'input {q}: {stop - start} seconds')
    
    errorFrac = calcErrorFraction(numWrong, len(dataSet))
    averageLoss = totalLoss / len(dataSet)
    
    return (hiddenLayers, outputWeights, errorFrac, averageLoss)


def trainBackpropClassifier(dataSet, hiddenLayers, outputWeights, eta=0.01, alpha=0.2):
    '''
    Takes in a dictionary, dataSet, with randomly ordered integer keys, and values {'label': 0-9, 'image': 'xxxx...'}, where each x is a float
    Also takes in a dictionary of weight matrices for all hiddenLayers, a matrix of outputWeights, a learning rate, eta, and a momentum rate, alpha. 

    Trains both the output and the hidden weights. 

    Returns the updated hiddenLayers, outputWeights, errorFrac, and totalLoss calculated for this epoch of input patterns.
    '''
    prevOutputWeightChanges = np.zeros(outputWeights.shape)
    prevHiddenWeightChanges = {h: np.zeros(hiddenLayers[h].shape) for h in range(len(hiddenLayers))}
    inputsDict = {q: [float(x) for x in dataSet[q]['image'].split()] for q in range(len(dataSet.keys()))}

    numWrong = 0
    totalLoss = 0

    for q in range(len(inputsDict.keys())): # for each input pattern
        start = time.time()

        hiddenNetInputs = {}
        hiddenOutputs = {}

        outputNetInputs = []
        actualOutputs = []
        desiredOutputs = [1 if (int(dataSet[q]['label']) == r) else 0 for r in range(10)]

        outputErrors = []

        hiddenErrors = {}

        #### carrying input signal forwards ####

        # carry input signal through the hidden layers
        for h in range(len(hiddenLayers)):
            hiddenNetInputs[h] = []
            if h == 0:
                hiddenNetInputs[h] = calcNetInputMatr(inputsDict[q], hiddenLayers[h])
            else:
                hiddenNetInputs[h] = calcNetInputMatr(hiddenOutputs[h-1], hiddenLayers[h])
            
            hiddenOutputs[h] = hiddenActivationFuncMatr(hiddenNetInputs[h])

        # feed the input signal to the output layer
        outputNetInputs = calcNetInputMatr(hiddenOutputs[max(hiddenOutputs.keys())], outputWeights) # get each output net input
        
        # calculate actual outputs
        actualOutputs = outputActivationFuncMatr(outputNetInputs) # apply the activation function to each net input
        actualOutputs = actualOutputs.tolist()[0] # actual outputs is returned as a matrix, need it as a regular list

        #### calculating and backpropagating errors ####

        # calculate the error for each output neuron
        loss = 0.5 * sum(np.square(np.subtract(actualOutputs, desiredOutputs)))
        totalLoss += loss

        outputErrors = calcOutputDeltaMatr(actualOutputs, desiredOutputs)

        # if any of the output neuron's actualOutput was within the correct threshold, change the error to 0 and don't change those weights
        for i in range(len(outputErrors)): # for each output neuron 
            if desiredOutputs[i] == 1 and actualOutputs[i] > 0.75:
                outputErrors[i] = 0.0
                prevOutputWeightChanges[i] = np.zeros(prevOutputWeightChanges[i].shape)
            
            elif desiredOutputs[i] == 0 and actualOutputs[i] < 0.25:
                outputErrors[i] = 0.0
                prevOutputWeightChanges[i] = np.zeros(prevOutputWeightChanges[i].shape)

        # change output weights, store the weight changes for future use (because of momentum)
        outputWeights, prevOutputWeightChanges = calcWeightChanges(np.asmatrix(outputErrors), hiddenOutputs[len(hiddenOutputs) - 1], outputWeights, prevOutputWeightChanges, eta, alpha)

        # iterate backwards through hidden layers
        for h in range(len(hiddenLayers) - 1, -1, -1):
            hiddenErrors[h] = calcHiddenDeltaMatr(hiddenOutputs[h], outputWeights, outputErrors)                 

            # change hidden weights, store the weight changes for future use (because of momentum) 
            if h == 0: 
                hiddenLayers[h], prevHiddenWeightChanges[h] = calcWeightChanges(hiddenErrors[h], inputsDict[q], hiddenLayers[h], prevHiddenWeightChanges[h], eta, alpha)
            else:
                hiddenLayers[h], prevHiddenWeightChanges[h] = calcWeightChanges(hiddenErrors[h], hiddenOutputs[h-1], hiddenLayers[h], prevHiddenWeightChanges[h], eta, alpha)
            
        # record how many training inputs are classified wrong
        if dataSet[q]['label'] != actualOutputs.index(max(actualOutputs)):
            numWrong += 1

        stop = time.time()
        print(f"input {q}: {stop - start} seconds | predicted: {actualOutputs.index(max(actualOutputs))} | actual: {dataSet[q]['label']}")
    
    errorFrac = calcErrorFraction(numWrong, len(dataSet))
    averageLoss = totalLoss / len(dataSet)
            
    return (hiddenLayers, outputWeights, errorFrac, averageLoss)


def testBPNetwork(dataSet, hiddenLayers, outputWeights):
    numWrong = 0
    estimates = []

    for q in range(len(dataSet.keys())):

        start = time.time()
        hiddenNetInputs = {}
        actualOutputs = []
        outputNetInputs = []

        # get input vector for input pattern q
        inputs = [float(x) for x in dataSet[q]['image'].split()]

        # calculate outputs of hidden neurons
        hiddenOutputs = {}
        for h in range(len(hiddenLayers)):
            hiddenNetInputs[h] = []
            for j in range(len(hiddenLayers[h])): # for each hidden neuron
                if h == 0:
                    hiddenNetInputs[h].append(calcNetInput(inputs, hiddenLayers[h][j])) # pass in row of weight matrix (so, all weights for one hidden neuron) and all inputs from image q
                else:
                    hiddenNetInputs[h].append(calcNetInput(hiddenOutputs[h-1], hiddenLayers[h][j])) # pass in row of weight matrix (so, all weights for one hidden neuron) and all inputs from image q
            hiddenOutputs[h] = list(map(hiddenActivationFunc, hiddenNetInputs[h])) # store hidden neuron outputs

        # calculate actual outputs
        for i in range(len(outputWeights)):
            outputNetInputs.append(calcNetInput(hiddenOutputs[len(hiddenOutputs) - 1], outputWeights[i])) # get each output net input
        actualOutputs = list(map(outputActivationFunc, outputNetInputs)) # apply the activation function to each net input

        # record how many training inputs are classified wrong
        estimate = actualOutputs.index(max(actualOutputs))
        estimates.append(estimate)
        if dataSet[q]['label'] != estimate:
            numWrong += 1

        stop = time.time()
        print(f'test input {q}: {stop - start} seconds')
    
    errorFrac = calcErrorFraction(numWrong, len(dataSet))

    return estimates, errorFrac


def plotConfusionMatrices(trainEstimates, trainLabels, testEstimates, testLabels, title='', subtitles=['',''], show=False, save=True, fileName=''):
    
    # calculate the confusion matrix
    cm1 = confusion_matrix(y_true=trainLabels, y_pred=trainEstimates)
    cm2 = confusion_matrix(y_true=testLabels, y_pred=testEstimates)    
    
    # print the confusion matrix
    fig, axes = plt.subplots(ncols=2, figsize=(7.5, 7.5))
    axes[0].matshow(cm1, cmap=plt.cm.Blues, alpha=0.3)
    axes[0].set_xticks(range(10))
    axes[0].set_yticks(range(10))
    axes[0].set_xlabel('Predictions', fontsize=12)
    axes[0].set_ylabel('Actuals', fontsize=12)
    axes[0].set_title(subtitles[0], fontsize=14)

    axes[1].matshow(cm2, cmap=plt.cm.Blues, alpha=0.3)
    axes[1].set_xticks(range(10))
    axes[1].set_yticks(range(10))
    axes[1].set_xlabel('Predictions', fontsize=12)
    axes[1].set_ylabel('Actuals', fontsize=12)
    axes[1].set_title(subtitles[1], fontsize=14)

    # put the data on the matrices
    for i in range(cm1.shape[0]):
        for j in range(cm1.shape[1]):
            axes[0].text(x=j, y=i, s=cm1[i, j], va='center', ha='center', size='medium')
            axes[1].text(x=j, y=i, s=cm2[i, j], va='center', ha='center', size='medium')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    if show: plt.show()
    if save:
        if not os.path.isdir('figures'): os.mkdir('figures')
        plt.savefig(f'figures\\{fileName}')


def testNetwork(dataSet, hiddenLayers, outputWeights):
    testStart = time.time()

    totalLoss = 0
    numWrong = 0

    allActualOutputs = []
    estimates = []

    inputsDict = {q: [float(x) for x in dataSet[q]['image'].split()] for q in dataSet.keys()}
    inputCount = 0

    for q in dataSet.keys(): # for each input pattern

        start = time.time()

        hiddenNetInputs = {}
        hiddenOutputs = {}

        outputNetInputs = []
        actualOutputs = []

        desiredOutputs = [1 if (int(dataSet[q]['label']) == r) else 0 for r in range(10)]

        # calculate outputs of hidden neurons
        hiddenOutputs = {}

        for h in range(len(hiddenLayers)):
            hiddenNetInputs[h] = []
            if h == 0:
                hiddenNetInputs[h] = calcNetInputMatr(inputsDict[q], hiddenLayers[h])
            else:
                hiddenNetInputs[h] = calcNetInputMatr(hiddenOutputs[h-1], hiddenLayers[h])
            
            hiddenOutputs[h] = hiddenActivationFuncMatr(hiddenNetInputs[h])

        outputNetInputs = calcNetInputMatr(hiddenOutputs[max(hiddenOutputs.keys())], outputWeights) # get each output net input
                
        # calculate actual outputs
        actualOutputs = outputActivationFuncMatr(outputNetInputs) # apply the activation function to each net input
        actualOutputs = actualOutputs.tolist()[0]

        # record how many inputs are classified wrong
        estimate = actualOutputs.index(max(actualOutputs))
        estimates.append(estimate)
        if dataSet[q]['label'] != estimate:
            numWrong += 1

        loss = 0.5 * sum((np.square(np.subtract(actualOutputs, desiredOutputs))))
        totalLoss += loss

        stop = time.time()
        print(f"test input {inputCount}: {stop - start} seconds | predicted: {estimate} | actual: {dataSet[q]['label']}")
        inputCount += 1
    
    averageLoss = totalLoss / len(dataSet)
    errorFrac = calcErrorFraction(numWrong, len(dataSet))

    testStop = time.time()
    print(f'============ TOTAL TESTING TIME: {testStop - testStart} seconds ============')
    print(f'============ TESTING ERROR FRACTION: {errorFrac * 100}% ============')
    
    return estimates, averageLoss, errorFrac


if __name__ == "__main__":
    programStart = time.time()

    LETTERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    DEVELOPING = False
    NUM_EPOCHS = 130 
    NUM_HIDDEN_NEURONS = 150
    INPUTS_PER_EPOCH = 1000

    #################################### Pre-Processing ####################################

    labelsNeeded = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    allData = getDataDict('MNISTnumLabels5000_balanced.txt', 'MNISTnumImages5000_balanced.txt')
    # filteredData = filterData(allData, labelsNeeded) # ineffective for HW4 since all digits are used
    filteredData = allData # HW4 specific -- all digits needed, no need to filter

    # create/open files for each label
    if not os.path.isdir('dataSets'):
        os.makedirs('dataSets')
    fileNames = {0: open('dataSets\\0.txt', 'w+'),\
                1: open('dataSets\\1.txt', 'w+'),\
                2: open('dataSets\\2.txt', 'w+'),\
                3: open('dataSets\\3.txt', 'w+'),\
                4: open('dataSets\\4.txt', 'w+'),\
                5: open('dataSets\\5.txt', 'w+'),\
                6: open('dataSets\\6.txt', 'w+'),\
                7: open('dataSets\\7.txt', 'w+'),\
                8: open('dataSets\\8.txt', 'w+'),\
                9: open('dataSets\\9.txt', 'w+')}

    writeImagesByLabel(filteredData, fileNames) # write images to 0.txt, 1.txt, etc.

    # get first 400 of each digit for trainingSet
    trainingSet = storeSet(filteredData, list(range(10)), 400, 'dataSets\\trainingSet.txt', randomize=True)
    testSet = {}
    rowNum = 0
    # get the last 100 of each digit for the testSet
    for i in range(400, 500):
        # get last 100 of 0s, rowNums 0-99 in testSet
        # testSet[rowNum] = {'label': filteredData[i]['label'], 'image': filteredData[i]['image']}
        # get last 100 of 1s, rowNums 100-199 in testSet
        # testSet[rowNum + 100] = {'label': filteredData[i + 500]['label'], 'image': filteredData[i + 500]['image']}
        
        for testSetOffset in range(0, 1000, 100):
            filteredDataOffset = testSetOffset * 5
            testSet[rowNum + testSetOffset] = {'label': filteredData[i + filteredDataOffset]['label'], 'image': filteredData[i + filteredDataOffset]['image']}
        rowNum += 1

    # store testSet - usually storeSet filters by the list of labelsNeeded, but here it will be storing
    # all of the values it's given since we filtered beforehand to get only the last 100 of the 0s and 1s
    # only calling storeSet to write out to the file
    testSet = storeSet(testSet, list(range(10)), 100, 'dataSets\\testSet.txt', randomize=False)

    trainLabels = [trainingSet[q]['label'] for q in trainingSet.keys()]
    testLabels = [testSet[q]['label'] for q in testSet.keys()]

    #################################### Training and Testing ####################################

    neuronsPerHiddenLayer = [] # number of neurons in each hidden layer

    if not DEVELOPING:
        # during development, use 1 hidden layer, and constant number of hidden neurons 
        numHiddenLayers = 1
        neuronsPerHiddenLayer = [NUM_HIDDEN_NEURONS]

    #################################### HW 6 CODE ####################################

    # initialize weights for hidden layers
    hiddenLayers = {0: np.empty((150, 785))}

    # read in hidden weights from autoencoder in HW5
    print('importing weights...\n')
    with open('weights\\hw5_all_hidden_weights.txt', 'r') as f:
        pattern = '\[(.*?)\]'
        rowList = f.read().split('\n\n')
        for i in range(len(rowList)):
            if rowList[i] == '': continue
            weightsString = re.findall(pattern, rowList[i])[0]
            hiddenLayers[0][i,:] = np.asarray([float(x) for x in weightsString.split(', ')])

    print('finished importing weights!\n')


    #### CASE I - TRAIN USING LMS ####


    # initialize weights for output layer
    # XAVIER INITIALIZATION
    outputWeights = (np.random.rand(10, neuronsPerHiddenLayer[-1] + 1) - 0.5) * math.sqrt(6 / (neuronsPerHiddenLayer[-1] + 10))

    epochTimes = []

    # get initial averageTrainLoss and trainErrorFrac
    trainLosses = []
    trainErrorFracsLMS = []
    _, averageTrainLoss, trainErrorFrac = testNetwork(trainingSet, hiddenLayers, outputWeights)
    trainLosses.append(averageTrainLoss)
    trainErrorFracsLMS.append(trainErrorFrac)
    
    # get initial averageTestLoss and testErrorFrac
    testLosses = []
    testErrorFracsLMS = []
    _, averageTestLoss, testErrorFrac = testNetwork(testSet, hiddenLayers, outputWeights)
    testLosses.append(averageTestLoss)
    testErrorFracsLMS.append(testErrorFrac)

    # begin training 
    for epoch in range(NUM_EPOCHS):
        start = time.time()

        # initialize random training subset (for SGD)
        trainingSubset = {}
        # first, store trainingSubset in a randomly-ordered list 
        trainingSubsetList = [trainingSet[key] for key in random.sample(sorted(trainingSet), INPUTS_PER_EPOCH)]
        # then, put it in a dictionary with indexed keys 
        for p in range(len(trainingSubsetList)):
            trainingSubset[p] = trainingSubsetList[p]

        # train for an epoch        
        hiddenLayers, outputWeights, _, _ = trainLMSClassifier(trainingSubset, hiddenLayers, outputWeights, alpha=0.5)
        
        # record length of each epoch
        stop = time.time()
        epochTimes.append(stop-start)
        print(f'============ EPOCH {epoch} TIME: {stop - start} seconds ============')
        print(f'============ EPOCH {epoch} TRAINING ERROR: {100 * trainErrorFrac}% ============')

        # save loss and error fractions after every 10 epochs
        if epoch % 10 == 0 and epoch != 0:
            # store training set results
            _, averageTrainLoss, trainErrorFrac = testNetwork(trainingSet, hiddenLayers, outputWeights)
            trainErrorFracsLMS.append(trainErrorFrac)
            trainLosses.append(averageTrainLoss)
            
            # run through test set and store results
            _, averageTestLoss, testErrorFrac = testNetwork(testSet, hiddenLayers, outputWeights)
            testErrorFracsLMS.append(testErrorFrac)
            testLosses.append(averageTestLoss)

    # get final results of LMS network
    # run through training set and store results
    trainEstimates, averageTrainLoss, trainErrorFrac = testNetwork(trainingSet, hiddenLayers, outputWeights)
    trainErrorFracsLMS.append(trainErrorFrac)
    trainLosses.append(averageTrainLoss)
    
    # run through test set and store results
    testEstimates, averageTestLoss, testErrorFrac = testNetwork(testSet, hiddenLayers, outputWeights)
    testErrorFracsLMS.append(testErrorFrac)
    testLosses.append(averageTestLoss)

    # get training set organized by digit (for plotting the losses by digit)
    trainingSetByDigits = {i: {} for i in range(10)}
    for i in trainingSet.keys():
        lbl = int(trainingSet[i]['label'])
        trainingSetByDigits[lbl][i] = trainingSet[i]   

    # get test set organized by digit (for plotting the losses by digit)
    testSetByDigits = {i: {} for i in range(10)}
    for i in testSet.keys():
        lbl = int(testSet[i]['label'])
        testSetByDigits[lbl][i] = testSet[i]   
    
    trainLossesByDigit = {i: 0 for i in range(10)}
    testLossesByDigit = {i: 0 for i in range(10)}

    # get average loss for each digit (for training and test sets)
    for i in range(10):
        _, digitTrainLoss, _ = testNetwork(trainingSetByDigits[i], hiddenLayers, outputWeights)
        trainLossesByDigit[i] = digitTrainLoss
        _, digitTestLoss, _ = testNetwork(testSetByDigits[i], hiddenLayers, outputWeights)
        testLossesByDigit[i] = digitTestLoss

    print(f'TOTAL TRAINING TIME FOR LMS CLASSIFIER: {sum(epochTimes)}')

    ###### plot final LMS results ######
    plotConfusionMatrices(trainEstimates, trainLabels, testEstimates, testLabels,  \
        title='Figure 1: Confusion Matrices Using LMS Network', \
        subtitles=['Figure 1(a): Training Set', 'Figure 1(b): Test Set'], save=True, fileName='figure_1')
    
    plotFinalLosses(averageTrainLoss, averageTestLoss, \
        title='Figure 3: Final Mean Squared Errors\nUsing LMS Network', \
        save=True, filename='figure_3')

    plotLossesByDigit(trainLossesByDigit, testLossesByDigit, \
        title='Figure 5: Final Mean Squared Errors by Digit\nUsing LMS Network', \
        save=True, filename='figure_5')

    print('Finished with Case 1!')

    
    #### CASE II - TRAIN USING BACKPROPAGATION ####


    # initialize weights for output layer
    # XAVIER INITIALIZATION
    outputWeights = (np.random.rand(10, neuronsPerHiddenLayer[-1] + 1) - 0.5) * math.sqrt(6 / (neuronsPerHiddenLayer[-1] + 10))

    epochTimes = []

    # get initial averageTrainLoss and trainErrorFrac
    trainLosses = []
    trainErrorFracsBP = []
    _, averageTrainLoss, trainErrorFrac = testNetwork(trainingSet, hiddenLayers, outputWeights)
    trainLosses.append(averageTrainLoss)
    trainErrorFracsBP.append(trainErrorFrac)
    
    # get initial averageTestLoss and testErrorFrac
    testLosses = []
    testErrorFracsBP = []
    _, averageTestLoss, testErrorFrac = testNetwork(testSet, hiddenLayers, outputWeights)
    testLosses.append(averageTestLoss)
    testErrorFracsBP.append(testErrorFrac)

    # begin training 
    for epoch in range(NUM_EPOCHS):
        start = time.time()

        # initialize random training subset (for SGD)
        trainingSubset = {}
        # first, store trainingSubset in a randomly-ordered list 
        trainingSubsetList = [trainingSet[key] for key in random.sample(sorted(trainingSet), INPUTS_PER_EPOCH)]
        # then, put it in a dictionary with indexed keys 
        for p in range(len(trainingSubsetList)):
            trainingSubset[p] = trainingSubsetList[p]

        # train for an epoch        
        hiddenLayers, outputWeights, _, _ = trainBackpropClassifier(trainingSubset, hiddenLayers, outputWeights, alpha=0.5)
        
        # record length of each epoch
        stop = time.time()
        epochTimes.append(stop-start)
        print(f'============ EPOCH {epoch} TIME: {stop - start} seconds ============')
        print(f'============ EPOCH {epoch} TRAINING ERROR: {100 * trainErrorFrac}% ============')

        # save loss and error fractions after every 10 epochs
        if epoch % 10 == 0 and epoch != 0:
            # run through training set and store results
            _, averageTrainLoss, trainErrorFrac = testNetwork(trainingSet, hiddenLayers, outputWeights)
            trainErrorFracsBP.append(trainErrorFrac)
            trainLosses.append(averageTrainLoss)
            
            # run through test set and store results
            _, averageTestLoss, testErrorFrac = testNetwork(testSet, hiddenLayers, outputWeights)
            testErrorFracsBP.append(testErrorFrac)
            testLosses.append(averageTestLoss)

    # get final results of backpropagation network
    # run through training set and store results
    trainEstimates, averageTrainLoss, trainErrorFrac = testNetwork(trainingSet, hiddenLayers, outputWeights)
    trainErrorFracsBP.append(trainErrorFrac)
    trainLosses.append(averageTrainLoss)
    
    # run through test set and store results
    testEstimates, averageTestLoss, testErrorFrac = testNetwork(testSet, hiddenLayers, outputWeights)
    testErrorFracsBP.append(testErrorFrac)
    testLosses.append(averageTestLoss)
    
    # get training set organized by digit (for plotting the losses by digit)
    trainingSetByDigits = {i: {} for i in range(10)}
    for i in trainingSet.keys():
        lbl = int(trainingSet[i]['label'])
        trainingSetByDigits[lbl][i] = trainingSet[i]   

    # get test set organized by digit (for plotting the losses by digit)
    testSetByDigits = {i: {} for i in range(10)}
    for i in testSet.keys():
        lbl = int(testSet[i]['label'])
        testSetByDigits[lbl][i] = testSet[i]   
    
    trainLossesByDigit = {i: 0 for i in range(10)}
    testLossesByDigit = {i: 0 for i in range(10)}

    # get average loss for each digit (for training and test sets)
    for i in range(10):
        _, digitTrainLoss, _ = testNetwork(trainingSetByDigits[i], hiddenLayers, outputWeights)
        trainLossesByDigit[i] = digitTrainLoss
        _, digitTestLoss, _ = testNetwork(testSetByDigits[i], hiddenLayers, outputWeights)
        testLossesByDigit[i] = digitTestLoss

    print(f'TOTAL TRAINING TIME FOR BACKPROPAGATION CLASSIFIER: {sum(epochTimes)} seconds')

    ###### plot final Backpropagation results ######
    plotConfusionMatrices(trainEstimates, trainLabels, testEstimates, testLabels,  \
        title='Figure 2: Confusion Matrices Using Backpropagation Network', \
        subtitles=['Figure 2(a): Training Set', 'Figure 2(b): Test Set'], save=True, fileName='figure_2')
    
    plotFinalLosses(averageTrainLoss, averageTestLoss, \
        title='Figure 4: Final Mean Squared Errors\nUsing Backpropagation Network', \
        save=True, filename='figure_4')

    plotLossesByDigit(trainLossesByDigit, testLossesByDigit, \
        title='Figure 6: Final Mean Squared Errors by Digit\nUsing Backpropagation Network', \
        save=True, filename='figure_6')
    
    ##### plot final error fraction time-series for both networks #####
    plotErrorFractions(trainErrorFracsLMS, trainErrorFracsBP, title='Figure 7: Time Series of the Error Fraction of Both Networks During Training', save=True, filename='figure_7')

    print('Finished with Case 2!')

    programStop = time.time()

    print(f'============ TOTAL RUNTIME: {round((programStop - programStart) / 60, 3)} minutes ============')
    print('FINISHED!')