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
        self.features = None
        self.sampleX = None 
        self.sampleY = None 

        self.trainX = None
        self.trainY = None  
        self.flag = False
        
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
        # selected_ind = np.argsort(train_error)
        # # selected_ind = np.flip(selected_ind)
        
        # selected_trainX = trainX[selected_ind]
        # selected_trainY = trainY[selected_ind]


        # METHOD 2: cluster the data into 5 groups by the training error
        sorted_ind = np.argsort(train_error)
        selected_ind = []
        steps = int(N/5)
        for i in range(steps):
            selected_ind.extend(sorted_ind[i::steps])
        print(len(selected_ind))
        selected_trainX = trainX[selected_ind, :]
        selected_trainY = trainY[selected_ind]

        # ## METHOD 3: random selection 
        # selected_ind = np.arange(N)
        # np.random.shuffle(selected_ind)    

        # selected_trainX = trainX[selected_ind, :]
        # selected_trainY = trainY[selected_ind]

        return selected_trainX, selected_trainY    # A subset of trainX and trainY


    def select_data(self, trainX, trainY):
        ''' Task 1-5
            Todo: '''
        N, M = trainX.shape[0], trainX.shape[1]
        features = self.select_features()
        sampleX, sampleY = self.select_sample(trainX, trainY)

        opt_err = np.inf
        opt_trainX, opt_trainY = sampleX, sampleY
        opt_feat = features

        possible_K = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1]
        feat_per_grid = []
        for k in possible_K:
            if k >= self.training_cost:
                feat_per_grid.append(k)

        for feat_per in feat_per_grid:

            feat_num = int(M * feat_per)
            sample_num = int(self.training_cost / feat_per * N)
            
            selected_features = features[:feat_num]
            selected_sampleX = sampleX[:sample_num]
            selected_sampleY = sampleY[:sample_num]
            
            selected_trainX = selected_sampleX[:, selected_features]
            selected_trainY = selected_sampleY

            model = MyRegressor(alpha=0)
            err = model.train(selected_trainX, selected_trainY)

            if err < opt_err:
                opt_err = err
                opt_trainX = selected_trainX
                opt_trainY = selected_trainY
                opt_feat = features[:feat_num]
                opt_feat_per = feat_num / M
                opt_sample_per = sample_num / N

        selected_trainX = opt_trainX
        selected_trainY = opt_trainY
        self.features = opt_feat
        print(f"cost: {self.training_cost}")
        print(f"feat_per: {opt_feat_per}")
        print(f"sample_per: {opt_sample_per}")

        return opt_trainX, opt_trainY
    
    
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
        train_error = 0
        for index, x in enumerate(trainX):  
            y = trainY[index]
            ### Todo:
            
            trainX_s, trainY_s = self.sensorNode(x, y)
            if trainX_s:
                N, M = trainX_s.shape
                trainY = trainY_s.reshape(N, -1)

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

                y = trainX @ self.weight + self.bias
                trainY = trainY.squeeze(1)
                train_error = mean_absolute_error(trainY, y)

        return self.training_cost, train_error
    
    def sensorNode(self, x, y):
        
        batch_size = 10
        x = x.reshape(-1, 1)
        self.flag = False 
        if self.flag:
            self.trainX = np.append(self.trainX, x)
            self.trainY = np.append(self.trainY, y)
        else: 
            self.trainX = x 
            self.trainY = y 
            self.flag = True

        trainX = self.trainX
        trainY = self.trainY 

        N, M = trainX.shape[0], trainX.shape[1]

        if N % batch_size == 0:
            print(N)
            sample_num = int(self.training_cost * N)

            if N >= 10 * batch_size:
                sampleX, sampleY = self.select_sample(trainX, trainY)
                trainX_s = sampleX[:sample_num]
                trainY_s = sampleY[:sample_num]

                return trainX_s, trainY_s
            elif N >= 5 * batch_size:
                selected_ind = np.arange(N)
                np.random.shuffle(selected_ind)    

                trainX_s = trainX[selected_ind[:sample_num], :]
                trainY_s = trainY[selected_ind[:sample_num]]
            else:
                return None, None 
        else: 
            return None, None



        

    
    def evaluate(self, X, Y):
        predY = X @ self.weight + self.bias
        error = mean_absolute_error(Y, predY)
        
        return predY, error
    
    
    def get_params(self):
        return self.weight, self.bias