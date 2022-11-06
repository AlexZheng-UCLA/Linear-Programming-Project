from audioop import reverse
from random import sample
from select import select
from traceback import print_tb
from sklearn.metrics import mean_absolute_error
import numpy as np
from scipy.optimize import linprog


class MyRegressor:
    def __init__(self, alpha):
        self.weight = None
        self.bias = None
        self.training_cost = 0   # N * M
        self.alpha = alpha
        
    def select_features(self):
        ''' Task 1-3
            Todo: '''
        weight = self.weight
        selected_feat = np.flip(np.argsort(weight))

        return selected_feat # The index List of selected features
        
        
    def select_sample(self, trainX, trainY):
        ''' Task 1-4
            Todo: '''
        N = trainX.shape[0]

        y, _ = self.evaluate(trainX, trainY)
        train_error = np.abs(trainY - y)

        ## METHOD 1: use data with smallest/largest training error
        selected_ind = np.argsort(train_error)
        # selected_ind = np.flip(selected_ind)
        
        selected_trainX = trainX[selected_ind]
        selected_trainY = trainY[selected_ind]


        # METHOD 2: cluster the data into 5 groups by the training error
        # sorted_ind = np.argsort(train_error)
        # selected_ind = []
        # steps = int(N/5)
        # for i in range(steps):
        #     selected_ind.extend(sorted_ind[i::steps])
        # print(len(selected_ind))
        # selected_trainX = trainX[selected_ind, :]
        # selected_trainY = trainY[selected_ind]

        # ## METHOD 3: random selection 
        # selected_ind = np.arange(N)
        # np.random.shuffle(selected_ind)    

        # selected_trainX = trainX[selected_ind, :]
        # selected_trainY = trainY[selected_ind]

        return selected_trainX, selected_trainY    # A subset of trainX and trainY


    def select_data(self, trainX, trainY):
        ''' Task 1-5
            Todo: '''
        N = trainX.shape[0]
        # METHOD 1: select all sample then reduce samples to increase features 

        for feat_num in [0.4, 0.3, 0.2, 0.1, 0.05]:  
            
            sample_num = int(self.training_cost / feat_num)


        
        return selected_trainX, selected_trainY
    
    
    def train(self, trainX, trainY):
        ''' Task 1-2
            Todo: '''
        N, M = trainX.shape
        trainY = trainY.reshape(N, -1)

        obj = [0] * M + [0] + [1/N] * N + [self.alpha] * M

        # l_ineq = [-X, -1, -I, 0 
        #            X, 1, -I, 0] 
        #            I, 0,  0, -I
        #           -I, 0,  0, -I]
        ones = np.ones((N, 1))
        zeros_theta0 = np.zeros((N, M))
        zeros_b = np.zeros((M, 1))
        zeros_z = np.zeros((M, N))

        l1 = np.hstack([-trainX, -ones, -np.eye(N), zeros_theta0])
        l2 = np.hstack([trainX, ones, -np.eye(N), zeros_theta0])
        l3 = np.hstack([np.eye(M), zeros_b, zeros_z, -np.eye(M)])
        l4 = np.hstack([-np.eye(M), zeros_b, zeros_z, -np.eye(M)])
        
        l = np.vstack([l1, l2, l3, l4])
        r = np.vstack([-trainY, trainY, np.zeros((M, 1)), np.zeros((M, 1))])

        opt = linprog(c=obj, A_ub=l, b_ub=r)
        print("If success:", opt.success)
        self.weight = opt.x[:M]
        # print("weight shape", self.weight.shape)
        self.bias = opt.x[M:M+1]

        # print("optimal value", opt.fun)

        # y = np.matmul(trainX, self.weight.reshape(M, -1)) + ones * self.bias
        # y = y.squeeze(1)

        y = trainX @ self.weight + self.bias
        trainY = trainY.squeeze(1)
        train_error = mean_absolute_error(trainY, y)

        return train_error
    
    
    def train_online(self, trainX, trainY):
        ''' Task 2 '''

        # we simulate the online setting by handling training data samples one by one
        for index, x in enumerate(trainX):
            y = trainY[index]

            ### Todo:
            
        return self.training_cost, train_error

    
    def evaluate(self, X, Y):
        predY = X @ self.weight + self.bias
        error = mean_absolute_error(Y, predY)
        
        return predY, error
    
    
    def get_params(self):
        return self.weight, self.bias