# Contains Logistic regression and Naive Bayes classes
# For COMP551 mini-project 1
# Lia Formenti
# February 1, 2020

#Trevor: -added combined naive bayes class for adult data set
#        -added +0.000001 to logs (4 of them) in LogisticRegression1   

import numpy as np

class Gaussian_Naive_Bayes:
    ''' Takes in a pandas dataframe with a column called 'class' provides functionalities to train and test a Naive Bayes model.
    Assumes classes are labeled 0, 1, 2 . . . and are not one-hot encoded'''

    def fit(self, X, y):
        N, D = X.shape # get number of features and instances
        C = len(np.unique(y)) # get the number of classes
        self.__C = C
        self.__getCatLogPriors(C,y)
        self.__getGausLikelihoodParams(C, D, X, y)
    
    def predict(self, X):
        '''Takes in test data and classes and outputs GNB predicted classes. Should be called after GNBfit'''
        N, D = X.shape
        C = self.__C
        logPosts = np.zeros(C) # will store posterior for each class
        preds = -np.ones(N) # One prediction per instance
        for i in range(N): # for each instance
            for c in range(C):
                logPosts[c] = self.__logPost(X[i,:], c)
            preds[i] = np.argmax(logPosts)
                
        preds = preds.astype('int')
        return preds
    
    def __logPost(self, xInst, c):
        '''Takes in an instance vector and calculates the log posterior - (some constant)'''
        logPrior = self.__logPriors[c]
        sumLogSigmas = np.sum(-0.5*np.log(self.__varis[c, :]), axis=0)
        gausExp = -np.sum(np.power(xInst - self.__means[c, :],2)/(2*self.__varis[c, :]))
        logLikelihood = sumLogSigmas + gausExp
        return logPrior + logLikelihood


    def __getCatLogPriors(self, C, y):
        '''Takes in training data classes and outputs prior probabilities.'''
        self.__logPriors = np.zeros(C)
        for c in range(C): # get the indices at which y == class c
            inds = np.where(y == c)[0] # incides of instances with class c
            self.__logPriors[c] = np.log(len(inds)/len(y))
        return 

    def __getGausLikelihoodParams(self, C, D, X, y):
        ''' Separates data by class and sets mean and variance of each feature by class in a C X D array '''
        # one mean for each feature of each class
        self.__means = np.zeros((C, D)) 
        self.__varis = np.zeros((C,D))
        for c in range(C):
            inds = np.where(y == c)[0]
            X_c = X[inds, :] # get instances of class c
            self.__means[c, :] = np.mean(X_c, axis=0)
            self.__varis[c, :] = np.var(X_c, axis=0)
             
        # print(self.__varis)
        # In experiment three, sometimes small vars are calculatedfor small number of instances, in this case just replace them.
        self.__varis[abs(self.__varis) < 1e-6] = 1e-6 
        
        return

    def __getCatLogPriors(self, C, y):
        '''Takes in training data classes and outputs prior probabilities.'''
        self.__logPriors = np.zeros(C)
        for c in range(C): # get the indices at which y == class c
            inds = np.where(y == c)[0] # incides of instances with class c
            self.__logPriors[c] = np.log(len(inds)/len(y))
        return 

class LogisticRegression1:
    
    def __init__(self, learning_rate, n_iters,max_iter, min_err,verbose=False):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.max_iter = max_iter
        self.min_err = min_err
        self.verbose = verbose 
        #X_traing (m(samples)x n (features)
        #Y_traing (m(samples)

    def fit(self,X_train, y_train):
        n_samples, n_features = X_train.shape
        # Initialize weights
        self.weights = np.array([np.zeros(n_features)]).T
        # self.weights = np.array([np.random.rand(n_features)]).T
        self.bias = 0
        #Gradient Descent and sigmoid function over the linear model
        loss_grad_train1 = []
        errArr = []
        # Calculate cost
        LinearModel = np.dot(X_train,self.weights) + self.bias
                       
        y_predicted = self._sigmoidFunction(LinearModel)
                      
        observations = len(y_train)
        class1_cost = -y_train*np.log(y_predicted+0.000001)
        class2_cost = (1-y_train)*np.log(1-y_predicted+0.000001)
        loss_grad_train = class1_cost - class2_cost
        loss_grad_train = (loss_grad_train.sum() / observations)
        # Add cost to list
        loss_grad_train1.append(loss_grad_train)
        # print(loss_grad_train1)
        # Update weights
        dw = (1/n_samples) * np.dot(X_train.T, (y_predicted-y_train))
        db = (1/n_samples) * np.sum(((y_predicted.ravel())-(y_train.ravel()))) 

        self.weights1 = self.weights
        self.weights = self.weights1 - (self.learning_rate*dw)
        self.bias -= self.learning_rate*db
         
        for i in range(self.max_iter):
            
            # Calculate cost
            LinearModel = np.dot(X_train,self.weights) + self.bias
                        
            y_predicted = self._sigmoidFunction(LinearModel)
           
            observations = len(y_train)
            class1_cost = -y_train*np.log(y_predicted+0.000001)
            class2_cost = (1-y_train)*np.log(1-y_predicted+0.000001)
            loss_grad_train = class1_cost - class2_cost
            loss_grad_train = (loss_grad_train.sum() / observations)
            # Add cost to list
            loss_grad_train1.append(loss_grad_train)
            #print('loss', loss_grad_trainl)
            #print(loss_grad_train[(_+1)-1],'loss entropydifference')
            # Calculate error
            l1 = loss_grad_train1[i]
            l2 = loss_grad_train1[i+1]
            
            #Stopping criteria based on the different loss entropy values per iteration
            
            err = abs(l2-l1)
            errArr.append(err)
            # print(err,'error', l2, l1)
            # dw=0
            # db=0
            
            if err < self.min_err or i >= self.max_iter - 1:
                # if self.verbose:
                print(f'Error: {err}')
                # print(self.weights)
                print(f'Finished after {i} iterations')
                return errArr
                break
            
            
            if  i % (self.max_iter // 10) == 0 and  self.verbose:
                print(f'Error: {err}')

            # If the error is not less than min_error, update the weights again
            dw = (1/n_samples) * np.dot(X_train.T, (y_predicted-y_train))
            db = (1/n_samples) * np.sum(((y_predicted.ravel())-(y_train.ravel()))) 
            
            #print ('dw',dw.shape)
            #print ('db',db.shape)
            #print ('ypredicted'+str(y_predicted.shape))
            #print('difference'+str((y_predicted-y_train)))
            #print ('dif'+str((y_predicted-y_train).shape))
            
            self.weights1 = self.weights
            self.weights = self.weights1 - (self.learning_rate*dw)
            self.bias -= self.learning_rate*db
            
                        
    def predict(self,X_test):
         
        LinearModel = np.dot(X_test,self.weights) + self.bias
        y_predicted = self._sigmoidFunction(LinearModel)
        y_predicted_classes = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_classes)
        
    def _sigmoidFunction(self, x):
        return  1/(1 + np.exp(-x))

class Combined_Naive_Bayes(object):
    
    def __init__(self):
        pass

    def fit(self, X, y):

        X_train_cont = X.filter(regex='cont')
        X_train_cat = X.filter(regex='cat')

        X_train_cont = X_train_cont.to_numpy(dtype=float)
        X_train_cat = X_train_cat.to_numpy(dtype=float)
        y_train = y.to_numpy(dtype=float)
        
        separated_g = [[x for x, t in zip(X_train_cont, y_train) if t == c] for c in np.unique(y_train)]    


        self.model = np.array([np.c_[np.mean(i, axis=0), np.std(i, axis=0)] 
                    for i in separated_g])

        count_sample = X_train_cat.shape[0]   

        separated_m = [[x for x, t in zip(X_train_cat, y_train) if t == c] for c in np.unique(y_train)]     
       
        self.class_log_prior_ = [np.log(len(i) / count_sample) for i in separated_m]      

        count = np.array([np.array(i).sum(axis=0) for i in separated_m]) +  1

        self.feature_log_prob_ = np.log(count / count.sum(axis=1)[np.newaxis].T)    
        
        return self



    def _prob(self, x, mean, std):
        exponent = np.exp(- ((x - mean)**2 / (2 * std**2)))                 
        return np.log(exponent / (np.sqrt(2 * np.pi) * std))            

    def predict_log_prob_g(self, X):
        return [[sum(self._prob(i, *s) for s, i in zip(g_param, x))     
                for g_param in self.model] for x in X]

    def predict_log_prob_m(self, X):
        return [(self.feature_log_prob_ * x).sum(axis=1) + self.class_log_prior_
                for x in X]

    def predict(self, X):
        X_test_cont = X.filter(regex='cont')
        X_test_cat = X.filter(regex='cat')

        X_test_cont = X_test_cont.to_numpy(dtype=float)
        X_test_cat = X_test_cat.to_numpy(dtype=float)

        total_predict = np.asarray(self.predict_log_prob_g(X_test_cont)) + self.predict_log_prob_m(X_test_cat)

        return np.argmax(total_predict, axis=1)

