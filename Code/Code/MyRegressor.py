from audioop import reverse
from select import select
from traceback import print_tb
from sklearn.metrics import mean_absolute_error
import numpy as np
from scipy.optimize import linprog
import numpy.linalg as la


class MyRegressor:
    def __init__(self, alpha):
        self.weight = None
        self.bias = None
        self.training_cost = 0   # N * N
        self.alpha = alpha
        
    def select_features(self):
        ''' Task 1-3
            Todo: '''
        weight = list(self.weight)
        weight = weight.sort().reversed()
        
        per = 0.01 

        N = int(per * len(weight))
        selected_weight = weight[:N]

        selected_feat = []
        for ele in selected_weight:
            selected_feat.append(self.weight.index(ele))

        print(selected_feat)
        return selected_feat # The index List of selected features
        
        
    def select_sample(self, trainX, trainY):
        ''' Task 1-4
            Todo: '''
        
        
        return selected_trainX, selected_trainY    # A subset of trainX and trainY


    def select_data(self, trainX, trainY):
        ''' Task 1-5
            Todo: '''
        
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
        # print("l1 shape:", l1.shape)
        l2 = np.hstack([trainX, ones, -np.eye(N), zeros_theta0])
        # print("l2 shape:", l2.shape)
        l3 = np.hstack([np.eye(M), zeros_b, zeros_z, -np.eye(M)])
        # print("l3 shape:", l3.shape)
        l4 = np.hstack([-np.eye(M), zeros_b, zeros_z, -np.eye(M)])
        # print("l4 shape:", l4.shape)
        
        l = np.vstack([l1, l2, l3, l4])
        # print("l shape:", l.shape)
        r = np.vstack([-trainY, trainY, np.zeros((M, 1)), np.zeros((M, 1))])
        # print("r shape:", r.shape)

        opt = linprog(c=obj, A_ub=l, b_ub=r)
        # print("If success:", opt.success)
        self.weight = opt.x[:M]
        # print("weight shape", self.weight.shape)
        self.bias = opt.x[M:M+1]

        # print("optimal value", opt.fun)
        # print("optimal theta shape", opt_theta.shape)
        # print("optimal b", opt_b)

        y = np.matmul(trainX, self.weight.reshape(M, -1)) + ones * self.bias
        y = y.squeeze(1)
        trainY = trainY.squeeze(1)
        # print("y_pred shape", y.shape)
        # train_error = la.norm(trainY - y, ord=1) / N
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