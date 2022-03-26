import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

        self.validationErr = 1

        self.learningRate = 0.1
        self.learningRateRange = [0.1,0.01]  #for annealing 
        self.momentum = 0.9
    
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

    def tanH(self, matrix):
        dMatrix = np.zeros(matrix.shape) 

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                dMatrix[i,j] = (np.e**(matrix[i,j]) - np.e**(-matrix[i,j]))/ (np.e**(matrix[i,j]) + np.e**(-matrix[i,j]))
        return dMatrix

    def dTanH(self, matrix):
        dMatrix = np.zeros(matrix.shape) 

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                dMatrix[i,j] = 1 - matrix[i,j]**2
        return dMatrix

    def feedForward(self, inputArray):

        #Get a row of data of size inputNodes!!
        inputRow = np.array(inputArray).reshape((-1,1))
        
        #Hidden Layer Calculations
        hiddens = np.dot(self.inputHiddenWeights, inputRow)
        hiddens = np.add(hiddens, self.hiddenBias) 
        hiddens = self.tanH(hiddens)

        #Output Layer Calculations
        outputs = np.dot(self.hiddenOutputWeights, hiddens)
        outputs = np.add(outputs, self.outputBias) 
        outputs = self.tanH(outputs)
        
        #might have to store output and hidden matrix !
        return inputRow,hiddens, outputs

    def backPropagation(self,inputRow, correctVals, outputs, hiddens): 
        #Get Output delta/s
        
        #Calculate output layer errors (Correct - Modelled)
        outputErrors = np.subtract(correctVals, outputs)

        #calculate gradient
        gradientOutputs = self.dTanH(outputs)   
        outputDelta = np.multiply(np.multiply(gradientOutputs, outputErrors),self.learningRate)
        # above is output delta = final gradientOutputs

        #Get output delta and apply them on all hidden to output weights!
        hoWeightDeltas= np.dot(outputDelta, hiddens.T) # if hidden count = 2 and output = 1 hoWeightDeltas = (1,2) makes sense


        #Calculate hidden node errors (hoWeights^T . outputErrors)
        hiddenErrors = np.dot(self.hiddenOutputWeights.T, outputErrors) 
        gradientHiddens = self.dTanH(hiddens)


        #Get hidden deltas and apply them on all input to hidden weights!
        
        hiddenDeltas = np.multiply(np.multiply(gradientHiddens, hiddenErrors), self.learningRate) #hidden delta shape is (2,1) makes sense
        ihWeightDeltas = np.dot(hiddenDeltas, inputRow.T)

        #What this returns:
        # -  Hidden/ Output Delta are only delta values 
        # -  ih/hoWeightDeltas are each deltas placed correctly with each weight via dot product!

        return ihWeightDeltas,hoWeightDeltas, hiddenDeltas,outputDelta
    
    def updateWeights(self, ihWeightDeltas, hoWeightDeltas, hiddenDeltas, outputDeltas):
        
        #adjust weights by deltas/gradient: ih/hoWeightDeltas contain biases, deltas,self.learning rate 
        
        self.inputHiddenWeights = np.add(self.inputHiddenWeights, ihWeightDeltas)
        self.hiddenOutputWeights = np.add(self.hiddenOutputWeights, hoWeightDeltas) 

        #Add Momentum to quicken the development
        self.inputHiddenWeights = self.applyMomentum(self.inputHiddenWeights,ihWeightDeltas)
        self.hiddenOutputWeights = self.applyMomentum(self.hiddenOutputWeights, hoWeightDeltas)
        
        #adjust biases by gradients new bias = old bias + learning rate * delta
        self.hiddenBias=np.add(self.hiddenBias, np.multiply(self.learningRate, hiddenDeltas))   
        self.outputBias=np.add(self.outputBias, np.multiply(self.learningRate, outputDeltas))
    
    def simAnnealing(self, startLRate, endLRate, tEpochs, epoch):
        #this changes learning rate from highest to lowest
        return endLRate + (startLRate-endLRate)*(1-(1/(1+np.e**(10 - ((20*epoch)/tEpochs)))))

    def weightDecay(self):
        pass

    def applyBoldDriver(self):
        #check if mean square error has increased
        pass

    def applyMomentum(self, weights, adjustVal):
        #Added Momentum to quicken the development
        return np.add(weights, np.multiply(self.momentum, adjustVal))
    
    def train(self, epochs):
        #epochs the number of times we go through the data set once
        trainDF = pd.read_excel (r'C:\Users\deror\OneDrive\Desktop\AICourseworkDataset.xlsx', sheet_name='Train')
        trainDF = trainDF.sample(frac=1)
        predictions=[]
        observations=[]
        TrainRMSE=[]
        TrainMSRE=[]
        TrainCE=[]
        TrainRSqr=[]

       # epochBD = [epochs/4 ,epochs/2, epochs*(3/4)] #intervals for applying bold driver 

        for ep in range(epochs):
            print("Epoch:", ep)
            #print(self.learningRate)
            for index, row in trainDF.iterrows():
                #get data
                
                dataRow = [row.sCrakehill, row.sSkipBridge, row.sWestwick, row.sSkeltonPre, row.sMonthNum, row.sArkengart,row.sEastCowt,row.sMalham,row.sSnaizehol]
                answer = row.sSkelton # put this outside an array
                #print(index,"Actual: ",answer) # put this outside an array
                inputRow, hiddenLayer, predicted = nn.feedForward(dataRow)
                #print(index,"Modelled: ", predicted.item(0))
                correctVal = np.array(answer).reshape((-1,1))  #can put this formatting inside backProp
                ihWeightDeltas, hoWeightDeltas, hDeltas, oDeltas = nn.backPropagation(inputRow,correctVal,predicted,hiddenLayer)
                nn.updateWeights(ihWeightDeltas, hoWeightDeltas, hDeltas, oDeltas)

                predictions.append(predicted.item(0,0))
                observations.append(answer)

            TrainRMSE.append(self.RMSE(observations,predictions))
            TrainMSRE.append(self.MSRE(observations,predictions))
            TrainCE.append(self.CE(observations,predictions))
            TrainRSqr.append(self.RSqr(observations,predictions))

            #SIMULATED ANNEALING: adjust learning rate.
            #self.learningRate = self.simAnnealing(self.learningRateRange[0], self.learningRateRange[1], epochs, ep)
            
            #BOLD DRIVER: check current epoch is the end of an interval to apply new learning rate.
            #if ep in epochBD:
            #    self.applyBoldDriver()

            if ep in [50*x for x in range(1, epochs)]:

                oldValidationErr = self.validationErr
                self.validationErr = self.validate()

                if (self.validationErr > oldValidationErr) and (self.validationErr - oldValidationErr) > 0.01: #custom formula to terminate training cycle with validation set.
                    self.recordData(predictions, observations, TrainRMSE, TrainMSRE, TrainCE, TrainRSqr)
                    print("Detect over-fitting, validation test delta RMSE: ", self.validationErr - oldValidationErr )
                    return

        self.recordData(predictions, observations, TrainRMSE, TrainMSRE, TrainCE, TrainRSqr)



    #Validation set is just a feedforward through vaildation set
    def validate(self):
        print("validate")
        validDF = pd.read_excel (r'C:\Users\deror\OneDrive\Desktop\AICourseworkDataset.xlsx', sheet_name='Validation')
        predictions=[]
        observations=[]
        ValidRMSErrors=[]
        ValidMSRErrors=[]
        ValidCE=[]
        ValidRSqr=[]

        for index, row in validDF.iterrows():  
            dataRow = [row.sCrakehill, row.sSkipBridge, row.sWestwick, row.sSkeltonPre, row.sMonthNum, row.sArkengart,row.sEastCowt,row.sMalham,row.sSnaizehol]
            answer = row.sSkelton 
            #print(index,"Actual: ",answer) 
            inputRow, hiddenLayer, predicted = nn.feedForward(dataRow)
            #print(index,"Modelled: ", predicted.item(0))
            

            predictions.append(predicted.item(0,0))
            observations.append(answer) 

        valRmse = self.RMSE(observations,predictions)
        ValidRMSErrors.append(valRmse)
        ValidMSRErrors.append(self.MSRE(observations,predictions))
        ValidCE.append(self.CE(observations,predictions))
        ValidRSqr.append(self.RSqr(observations,predictions))

        with open('Validrmse.txt', 'a') as f:
            for result in ValidRMSErrors:
                f.write("%s\n" % result) 

        # with open('Validmsre.txt', 'a') as f:
        #     for result in ValidMSRErrors:
        #         f.write("%s\n" % result) 

        # with open('Validce.txt', 'a') as f:
        #     for result in ValidCE:
        #         f.write("%s\n" % result)
        
        # with open('ValidRSqr.txt', 'a') as f:
        #     for result in ValidRSqr:
        #         f.write("%s\n" % result)
        
        return valRmse

        
    def MSE(self, observations,predictions):
        meanSqrErr=0
        i=0
        for obs, pred in zip(observations,predictions):
            meanSqrErr += (pred - obs)**2
            i+=1
        
        return meanSqrErr/i  

    def RSqr(self,observations,predictions):

        correlmatrix = np.corrcoef(observations, predictions)
        corr = correlmatrix[0,1]
        rSqr = corr**2

        print("R-Squared: ", rSqr)
        return rSqr

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


    def MSRE(self, observations, predictions):
        msre=0
        i=0
        for obs, pred in zip(observations,predictions):
            msre += ((pred - obs)/obs)**2
            i+=1
        msre=(1/i)*msre

        print("mean square relative error: ", msre)
        return msre
        
    
    def RMSE(self, observations, predictions): #MSErrors meansSquareErrors
        meanSqrErr=0
        i=0
        for obs, pred in zip(observations,predictions):
            meanSqrErr += (pred - obs)**2
            i+=1
        meanSqrErr = (meanSqrErr)/i
        rMeanSqrErr = math.sqrt(meanSqrErr)

        print("root mean square error: ",rMeanSqrErr)
        return rMeanSqrErr
            

    #add to text file
    def recordData(self, predictions, observations, rmsErrors, msrErrors, ce, rSqr):
        
        # with open('predictedResults.txt', 'w') as f:
        #     for result in predictions:
        #         f.write("%s\n" % result)
    
        # with open('observedResults.txt', 'w') as f:
        #     for result in observations:
        #         f.write("%s\n" % result)
 
    
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



       
nn = MLP(9,16,1)
nn.train(150) #1000 can break 0.04 rmse #200epochs + tanh + 9,16,1 can break 0.04 epochs


testDF = pd.read_excel (r'C:\Users\deror\OneDrive\Desktop\AICourseworkDataset.xlsx', sheet_name='Test')

