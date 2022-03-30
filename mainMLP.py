import math
import numpy as np
import pandas as pd


class MLP:

    # multi-layer-perceptron constructor
    def __init__(self, inputCount, hiddenCount, outputCount):

        #Set hidden weight structure
        self.inputCount = inputCount
        self.hiddenCount = hiddenCount
        self.outputCount = outputCount
        
        #Set randomized weights and biases
        wRange = [-2/inputCount, 2/inputCount]
        generator = np.random.default_rng()                              
        self.inputHiddenWeights = generator.uniform(wRange[0], wRange[1],(hiddenCount, inputCount))
        self.hiddenOutputWeights = generator.uniform(wRange[0], wRange[1],(outputCount, hiddenCount))

        self.hiddenBias = generator.uniform(wRange[0],wRange[1],(hiddenCount,1))
        self.outputBias = generator.uniform(wRange[0], wRange[1],(outputCount,1))
        
        #default learning rate
        self.learningRate = 0.2

        #improvement attributes
        self.validationErr = 1  #Set validation error to max to allow training to not be interrupted too early.
        self.learningRateRange = [0.2,0.01]  #for annealing 
        self.momentum = 0.9

        #Set current epoch to zero
        self.currentEp = 1

    
    #ACTIVATION FUNCTIONS AND IT'S RESPECTIVE DERIVATIVE CALCULATIONS:

    def sigmoid(self, matrix):  
        dMatrix = np.zeros(matrix.shape) 

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                dMatrix[i,j] = 1 / (1 + np.e**(-matrix[i,j]))
        return dMatrix

    def dSigmoid(self, matrix):   
        dMatrix = np.zeros(matrix.shape)
                 
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                dMatrix[i,j] = matrix[i,j]*(1-matrix[i,j])
        return dMatrix

    def tanH(self, matrix):
        dMatrix = np.zeros(matrix.shape) 

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                dMatrix[i,j] = (np.e**(matrix[i,j]) - np.e**(-matrix[i,j]))/(np.e**(matrix[i,j]) + np.e**(-matrix[i,j]))
        return dMatrix

    def dTanH(self, matrix):
        dMatrix = np.zeros(matrix.shape) 

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                dMatrix[i,j] = 1 - matrix[i,j]**2
        return dMatrix

    #Make Neural Network give a guess.
    def feedForward(self, inputArray):

        #Format row of data into a matrix.
        inputRow = np.array(inputArray).reshape((-1,1))
        
        #Hidden Layer matrix operations:   
        hiddens = np.dot(self.inputHiddenWeights, inputRow)
        hiddens = np.add(hiddens, self.hiddenBias) 
        hiddens = self.tanH(hiddens)

        #Output Layer matrix operations:
        outputs = np.dot(self.hiddenOutputWeights, hiddens)
        outputs = np.add(outputs, self.outputBias) 
        outputs = self.tanH(outputs)
        
        #These returned functions are for the back-propagation
        return inputRow,hiddens, outputs

    def backPropagation(self,inputRow, correctVals, outputs, hiddens): 
        
        #Get Output and Hidden deltas
        
        #Calculate output layer errors (Correct - Modelled)
        outputErrors = np.subtract(correctVals, outputs)
        
        #weight decay improvement not used for final model!

        #outputErrors = np.add(outputErrors, self.applyWeightDecay(self.currentEp))
        
        #Calculate output delta via output gradient
        gradientOutputs = self.dTanH(outputs)   
        outputDelta = np.multiply(gradientOutputs, outputErrors)
        
        #Get output delta and apply them on all hidden to output weights.
        hoWeightDeltas= np.dot(outputDelta, hiddens.T) 

        #Calculate hidden node errors and gradient of hidden.
        hiddenErrors = np.dot(self.hiddenOutputWeights.T, outputErrors) 
        gradientHiddens = self.dTanH(hiddens)

        #Get hidden deltas and apply them on all input to hidden weights.
        hiddenDeltas = np.multiply(gradientHiddens, hiddenErrors)
        ihWeightDeltas = np.dot(hiddenDeltas, inputRow.T)

        #What this returns:
        # -  Hidden/ Output Delta are only delta values used for bias update
        # -  ih/hoWeightDeltas are each correctly placed deltas for weight update.

        return ihWeightDeltas,hoWeightDeltas, hiddenDeltas,outputDelta
    
    def updateWeights(self, ihWeightDeltas, hoWeightDeltas, hiddenDeltas, outputDeltas):
        
        #adjust weights by deltas/gradient: ih/hoWeightDeltas contain biases, deltas,self.learning rate 
        
        self.inputHiddenWeights = np.add(self.inputHiddenWeights, np.multiply(ihWeightDeltas,self.learningRate))
        self.hiddenOutputWeights = np.add(self.hiddenOutputWeights, np.multiply(hoWeightDeltas,self.learningRate))

        #Add Momentum to quicken the development
        self.inputHiddenWeights = self.applyMomentum(self.inputHiddenWeights, np.multiply(ihWeightDeltas,self.learningRate))
        self.hiddenOutputWeights = self.applyMomentum(self.hiddenOutputWeights, np.multiply(hoWeightDeltas,self.learningRate))
        
        #adjust biases by gradients: new bias = old bias + learning rate * delta * 1
        self.hiddenBias=np.add(self.hiddenBias, np.multiply(self.learningRate, hiddenDeltas))   
        self.outputBias=np.add(self.outputBias, np.multiply(self.learningRate, outputDeltas))

    #IMPROVEMENTS TO THE NEURAL NETWORK:
    
    #improvement for adapting learning rate.
    def simAnnealing(self, startLRate, endLRate, tEpochs, epoch):
        
        return endLRate + (startLRate-endLRate)*(1-(1/(1+np.e**(10 - ((20*epoch)/tEpochs)))))
    
    #This function helps calculate omega for weight decay.
    def sumWeights(self,matrix):   
        sumMatrix = 0

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                sumMatrix += matrix[i,j]
        return sumMatrix       

    #Improvement that penalize large weight values to reduce over-fitting.
    def applyWeightDecay(self, ep):
        #calculate no. of weights in nn.
        weightCount = self.inputHiddenWeights.shape[0]*self.inputHiddenWeights.shape[1]
        + self.hiddenOutputWeights.shape[0]*self.hiddenOutputWeights.shape[1]
 
        #sum the sqaures of every weight
        ihSum = self.sumWeights((np.square(self.inputHiddenWeights)))
        hoSum = self.sumWeights((np.square(self.hiddenOutputWeights)))

        #calculate Upsilon and Omega
        omega = ((1/2)*weightCount)*(ihSum+hoSum)
        upsilon = (1/(self.learningRate*ep))

        #omega * upsilon
        return (omega*upsilon)

    def applyMomentum(self, weights, adjustVal):
        #Added Momentum to quicken the development
        return np.add(weights, np.multiply(self.momentum, adjustVal))
    
    # METHODS TO TRAIN NEURAL NETWORK.

    def train(self, epochs):
        #Retrieve dataset and shuffle order each epoch.
        trainDF = pd.read_excel (r'C:\Users\deror\OneDrive\Desktop\AICourseworkDataset.xlsx', sheet_name='Train')  
        

        #Set of assessment models to empty set after each epoch
        predictions=[]
        observations=[]
        TrainRMSE=[]
        TrainMSRE=[]
        TrainCE=[]
        TrainRSqr=[]


        #Loop for every epoch
        for ep in range(1,epochs+1,1):
            trainDF = trainDF.sample(frac=1) #shuffle dataset for each epoch
            print("Epoch:", ep)
            self.currentEp = ep

            #Loop through the shuffled data set.
            for index, row in trainDF.iterrows():
                #get row of data
                dataRow = [row.sCrakehill, row.sSkipBridge, row.sWestwick,row.sSkeltonPre, row.sMonthNum, row.sArkengart,row.sEastCowt,row.sMalham,row.sSnaizehol] 
                #get Answer
                answer = row.sSkelton 

                #Forward pass through the nn.
                inputRow, hiddenLayer, predicted = nn.feedForward(dataRow)
                #Format answer into a numpy matrix
                correctVal = np.array(answer).reshape((-1,1))  
                #Store weight adjustments from back-propagation
                ihWeightDeltas, hoWeightDeltas, hDeltas, oDeltas = nn.backPropagation(inputRow,correctVal,predicted,hiddenLayer)
                


                nn.updateWeights(ihWeightDeltas, hoWeightDeltas, hDeltas, oDeltas)

                #Append guess and actual answer to respective lists
                predictions.append(predicted.item(0,0))
                observations.append(answer)

            #Get assessment model results for current epoch.
            TrainRMSE.append(self.RMSE(observations,predictions))
            TrainMSRE.append(self.MSRE(observations,predictions))
            TrainCE.append(self.CE(observations,predictions))
            TrainRSqr.append(self.RSqr(observations,predictions))

            #SIMULATED ANNEALING: adjust learning rate towards desired learning rate boundary.
            self.learningRate = self.simAnnealing(self.learningRateRange[0], self.learningRateRange[1], epochs, ep)

            #Forward pass the validation set every 1 epochs to prevent over-fitting.
            if ep in [1*x for x in range(1, epochs)]:

                oldValidationErr = self.validationErr
                self.validationErr = self.validate()

                #custom formula to terminate training cycle with validation set.
                #Formula checks a if a large positive gradient is between ep and ep-1
                if (self.validationErr - oldValidationErr) > 0: 
                    self.recordData(predictions, observations, TrainRMSE, TrainMSRE, TrainCE, TrainRSqr)
                    print("Detect over-fitting, validation test delta RMSE: ", self.validationErr - oldValidationErr )
                    return

        #record assessment model results in a text file.
        self.recordData(predictions, observations, TrainRMSE, TrainMSRE, TrainCE, TrainRSqr)



    #Validation set 
    def validate(self):
        print("validate")
        validDF = pd.read_excel (r'C:\Users\deror\OneDrive\Desktop\AICourseworkDataset.xlsx', sheet_name='Validation')
        #Empty array every for every call.
        predictions=[]
        observations=[]
        ValidRMSErrors=[]
        ValidMSRErrors=[]
        ValidCE=[]
        ValidRSqr=[]

        #Forward pass through each data point
        for index, row in validDF.iterrows():  
            dataRow = [row.sCrakehill, row.sSkipBridge, row.sWestwick,row.sSkeltonPre, row.sMonthNum, row.sArkengart,row.sEastCowt,row.sMalham,row.sSnaizehol] 
            answer = row.sSkelton 

            inputRow, hiddenLayer, predicted = nn.feedForward(dataRow)

            predictions.append(predicted.item(0,0))
            observations.append(answer) 

        #Find each Assessment Models results.
        valRMSE=self.RMSE(observations,predictions) #We need to store this error for return func.
        ValidRMSErrors.append(valRMSE)
        ValidMSRErrors.append(self.MSRE(observations,predictions))
        ValidCE.append(self.CE(observations,predictions))
        ValidRSqr.append(self.RSqr(observations,predictions))

        with open('VpredictedResults.txt', 'w') as f:
            for result in predictions:
                f.write("%s\n" % result)
    
        with open('VobservedResults.txt', 'w') as f:
            for result in observations:
                f.write("%s\n" % result)

        #Append Results for respective validation assessment model.
        with open('Validrmse.txt', 'a') as f:
            for result in ValidRMSErrors:
                f.write("%s\n" % result) 

        with open('Validmsre.txt', 'a') as f:
            for result in ValidMSRErrors:
                f.write("%s\n" % result) 

        with open('Validce.txt', 'a') as f:
            for result in ValidCE:
                f.write("%s\n" % result)
        
        with open('ValidRSqr.txt', 'a') as f:
            for result in ValidRSqr:
                f.write("%s\n" % result)
        
        #return rmse to detech over-fitting during training cycle.
        return valRMSE

    #Test set 
    def Test(self):
        print("Test")
        
        #Retrieve test results
        testDF = pd.read_excel (r'C:\Users\deror\OneDrive\Desktop\AICourseworkDataset.xlsx', sheet_name='Test')
        
        #Empty array every for every call.
        predictions=[]
        observations=[]
        testRMSErrors=[]
        testMSRErrors=[]
        testCE=[]
        testRSqr=[]

        #Forward pass through each data point
        for index, row in testDF.iterrows():  
            dataRow = [row.sCrakehill, row.sSkipBridge, row.sWestwick,row.sSkeltonPre, row.sMonthNum, row.sArkengart,row.sEastCowt,row.sMalham,row.sSnaizehol] 
            answer = row.sSkelton 

            inputRow, hiddenLayer, predicted = nn.feedForward(dataRow)

            predictions.append(self.deStand(predicted.item(0,0)))
            observations.append(self.deStand(answer)) 

        #append Assessment Models results.
        testRMSErrors.append(self.RMSE(observations,predictions))
        testMSRErrors.append(self.MSRE(observations,predictions))
        testCE.append(self.CE(observations,predictions))
        testRSqr.append(self.RSqr(observations,predictions))

        with open('TpredictedResults.txt', 'w') as f:
            for result in predictions:
                f.write("%s\n" % result)
    
        with open('TobservedResults.txt', 'w') as f:
            for result in observations:
                f.write("%s\n" % result)

        #Record Results for respective test assessment model.
        with open('Testrmse.txt', 'w') as f:
            for result in testRMSErrors:
                f.write("%s\n" % result) 

        with open('Testmsre.txt', 'w') as f:
            for result in testMSRErrors:
                f.write("%s\n" % result) 

        with open('Testce.txt', 'w') as f:
            for result in testCE:
                f.write("%s\n" % result)
        
        with open('TestRSqr.txt', 'w') as f:
            for result in testRSqr:
                f.write("%s\n" % result)

    def deStand(self, value):
        return ((value-0.1)/0.8)*(448.1-3.694)+3.694
    

    # ASSESSMENT MODEL CALCULATIONS:

    #Mean Sqaure Error:
    def MSE(self, observations,predictions):
        meanSqrErr=0
        i=0
        for obs, pred in zip(observations,predictions):
            meanSqrErr += (pred - obs)**2
            i+=1
        
        return meanSqrErr/i  

    #R-Squared:
    def RSqr(self,observations,predictions):

        correlmatrix = np.corrcoef(observations, predictions)
        corr = correlmatrix[0,1]
        rSqr = corr**2

        print("R-Squared: ", rSqr)
        return rSqr

    #Coefficient of Efficiency:
    def CE(self,observations,predictions):
        ceNumer=0
        ceDenom=0
        meanObs = sum(observations) / len(observations)

        for obs, pred in zip(observations,predictions):
            ceNumer += (pred-obs)**2
            ceDenom += (obs-meanObs)**2
        
        ce = 1 - (ceNumer/ceDenom)

        print("Coefficient of Efficiency: ", ce)
        return ce

    #Mean Square Relative Error
    def MSRE(self, observations, predictions):
        msre=0
        i=0
        for obs, pred in zip(observations,predictions):
            msre += ((pred - obs)/obs)**2
            i+=1
        msre=(1/i)*msre

        print("mean square relative error: ", msre)
        return msre
        
    #Root Mean Square Error
    def RMSE(self, observations, predictions):
        meanSqrErr=0
        i=0
        for obs, pred in zip(observations,predictions):
            meanSqrErr += (pred - obs)**2
            i+=1
        meanSqrErr = (meanSqrErr)/i
        rMeanSqrErr = math.sqrt(meanSqrErr)

        print("root mean square error: ",rMeanSqrErr)
        return rMeanSqrErr
            

    #add to an empty text file after epoch
    def recordData(self, predictions, observations, rmsErrors, msrErrors, ce, rSqr):
        
        with open('predictedResults.txt', 'w') as f:
            for result in predictions:
                f.write("%s\n" % result)
    
        with open('observedResults.txt', 'w') as f:
            for result in observations:
                f.write("%s\n" % result)
 
        with open('rmse.txt', 'w') as f:
            for result in rmsErrors:
                f.write("%s\n" % result)

        with open('msre.txt', 'w') as f:
            for result in msrErrors:
                f.write("%s\n" % result)
        
        with open('ce.txt', 'w') as f:
            for result in ce:
                f.write("%s\n" % result)

        with open('RSqr.txt', 'w') as f:
            for result in rSqr:
                f.write("%s\n" % result)



#Set Neural Network Structure    
nn = MLP(9,16,1)
nn.train(200) 
nn.Test()


#1000 can break 0.04 rmse #200epochs + tanh + 9,16,1 can break 0.04 epochs



