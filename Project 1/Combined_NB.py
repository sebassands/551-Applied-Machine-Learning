import numpy as np

class Combined_Naive_Bayes(object):
    #Combines Gaussian and Multinomial Naive Bayes into one class for the adult data set
    #which has both continuous and categorical variables
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
        print('-------------------------------------------------------------------')
        print('X shape:')
        print(X.shape)
        print('log prob shape: ')
        print(self.feature_log_prob_.shape)
        print('-------------------------------------------------------------------')

        return [(self.feature_log_prob_ * x).sum(axis=1) + self.class_log_prior_
                for x in X]

    def predict(self, X):
        X_test_cont = X.filter(regex='cont')
        X_test_cat = X.filter(regex='cat')

        X_test_cont = X_test_cont.to_numpy(dtype=float)
        X_test_cat = X_test_cat.to_numpy(dtype=float)

        total_predict = np.asarray(self.predict_log_prob_g(X_test_cont)) + self.predict_log_prob_m(X_test_cat)

        return np.argmax(total_predict, axis=1)