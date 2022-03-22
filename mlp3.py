import numpy as np
import pandas as pd
import matplotlib as plt
import random

class MLP:

    # multi-layer-perceptron constructor
    def __init__(self, inputCount, hiddenCount, outputCount):

    #generator = np.random.default_rng() #seed is parameter that saves the random values if left empty it would not save random values

        self.inputCount = inputCount
        self.hiddenCount = hiddenCount
        self.outputCount = outputCount
        
        wRange = [-2/inputCount, 2/inputCount]
        generator = np.random.default_rng()                               #  Row         column
        self.inputHiddenWeights = generator.uniform(wRange[0], wRange[1],(hiddenCount, inputCount))
        self.hiddenOutputWeights = generator.uniform(wRange[0], wRange[1],(outputCount, hiddenCount))

        self.hiddenBias = generator.uniform(wRange[0],wRange[1],(hiddenCount,1))
        self.outputBias = generator.uniform(wRange[0], wRange[1],(outputCount,1))

        self.activeFunc = "sigmoid"  #does nothing for now
        self.learningRate= 0.1
    
    def sigmoid(self, matrix):  
        dMatrix = np.zeros(matrix.shape) 

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                dMatrix[i,j] = 1 / (1 + np.e**(-matrix[i,j]))
        return dMatrix

    def dSigmoid(self, matrix):   
        dMatrix = np.zeros(matrix.shape)

        #as each value is already a sigmoid value just use itself for formula                  
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                dMatrix[i,j] = matrix[i,j]*(1-matrix[i,j])
        return dMatrix

    def feedForward(self, inputArray):

        #Get a row of data of size inputNodes!!
        inputRow = np.array(inputArray).reshape((-1,1))
        
        #Hidden Layer Calculations
        hiddens = np.dot(self.inputHiddenWeights, inputRow)
        hiddens = np.add(hiddens, self.hiddenBias) 
        hiddens = self.sigmoid(hiddens)

        #Output Layer Calculations
        outputs = np.dot(self.hiddenOutputWeights, hiddens)
        outputs = np.add(outputs, self.outputBias) 
        outputs = self.sigmoid(outputs)
        
        #might have to store output and hidden matrix !
        return inputRow,hiddens, outputs

    def backPropagation(self,inputRow, correctVals, outputs, hiddens): 
        #Get Output delta/s
        
        #Calculate output layer errors (Correct - Modelled)
        outputErrors = np.subtract(correctVals, outputs)

        #calculate gradient
        gradientOutputs = self.dSigmoid(outputs)   
        outputDelta = np.multiply(np.multiply(gradientOutputs, outputErrors),self.learningRate)
        # above is output delta = final gradientOutputs

        #Get output delta and apply them on all hidden to output weights!
        hoWeightDeltas= np.dot(outputDelta, hiddens.T) # if hidden count = 2 and output = 1 hoWeightDeltas = (1,2) makes sense


        #Calculate hidden node errors (hoWeights^T . outputErrors)
        hiddenErrors = np.dot(self.hiddenOutputWeights.T, outputErrors) 
        gradientHiddens = self.dSigmoid(hiddens)


        #Get hidden deltas and apply them on all input to hidden weights!
        hiddenDeltas = np.multiply(np.multiply(gradientHiddens, hiddenErrors), self.learningRate) #hidden delta shape is (2,1) makes sense
        ihWeightDeltas = np.dot(hiddenDeltas, inputRow.T)

        #What this returns:
        # -  Hidden/ Output Delta are only delta values 
        # -  ih/hoWeightDeltas are each deltas placed correctly with each weight via dot product!

        return ihWeightDeltas,hoWeightDeltas, hiddenDeltas,outputDelta
    
    def updateWeights(self, ihWeightDeltas, hoWeightDeltas, hiddenDeltas, outputDeltas):
        
        #adjust weights by deltas/gradient
        self.inputHiddenWeights = np.add(self.inputHiddenWeights, ihWeightDeltas)
        self.hiddenOutputWeights = np.add(self.hiddenOutputWeights, hoWeightDeltas) 

        #adjust biases by gradients new bias = old bias + learning rate * delta
        self.hiddenBias=np.add(self.hiddenBias, np.multiply(self.learningRate, hiddenDeltas))   
        self.outputBias=np.add(self.outputBias, np.multiply(self.learningRate, outputDeltas))

        
nn = MLP(2,2,1)


#XOR training set each array is [input,input, correct]
trainingData =[[0,1,1],[1,0,1],[1,1,0],[0,0,0]]

#train
for i in range(50000):
    data = random.choice(trainingData)
    inputRow, hiddenLayer, outputLayer = nn.feedForward(data[0:2])
    correctVal = np.array(data[2]).reshape((-1,1))  #can put this formatting inside backProp
    ihWeightDeltas, hoWeightDeltas, hDeltas, oDeltas = nn.backPropagation(inputRow,correctVal,outputLayer,hiddenLayer)
    nn.updateWeights(ihWeightDeltas, hoWeightDeltas, hDeltas, oDeltas)

#test
print("TEST RESULTS")
a,b,guess1 =nn.feedForward([0,0]) #should be 0
a,b,guess2=nn.feedForward([1,0])  #should be 1
a,b,guess3=nn.feedForward([0,1])  #should be 1
a,b,guess4=nn.feedForward([1,1])  #should be 0
 
print(guess1)
print(guess2)
print(guess3)
print(guess4)


