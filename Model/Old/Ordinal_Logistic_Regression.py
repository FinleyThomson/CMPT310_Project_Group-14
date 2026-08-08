# This is an ordinal logistic regression implementation for analyzing the townhouse data
# expects an input file, this should be the town house data 
import sys
import numpy as np 
import pandas as pd
from PreprocessedDataForRegression import preprocess

class Regress:
    '''Regression Math: 
        probability of observation i falls into category j given X =X_i1, X_i2 ... X_im:
            p_ij = P(Y_i = j|X) = P(Y_i <= j|X) - P(Y_i <= j - 1| X)

        P(Y_i <= j|X) = sigmoid(T_j - B @ X), with P(Y_i <= highest_category|X) = 1
        and P(Y_i < lowest_category | X) = 0

        Must determine coeffiecients B = (B_1, B_2, ... , B_m), and thresholds T = (T_1, T_2, ... , T_m)

        Will use gradient descent

    '''
    def __init__(self, max_iter, learning_rate, num_classes):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.num_classes = num_classes
    
    def A_to_T(A, j):
        return A[0] + np.sum(np.exp(A[1:j]))

    vector_A_to_T = np.vectorize(A_to_T, excluded = ['A'])

    def d_loss_B(x,y,B,T):
        '''Inputs: Observation x, ie row of observations matrix X
                   label of observation y, ie entry from labels vector
                   Current regressors B
                   Current thresholds T

            gradient formula:
            dL/dB = x * (sig_y(1 - sig_y) - sig_{y-1}(1 - sig_{y-1})) / (sig_y - sig_{y-1})
            output: gradient of the loss for the current values of B
        '''

        if y == 0:
            return x * (1 - sigmoid((T[y]) - x @ B))

        elif y == T.size:
            return x * (-sigmoid(T[y-1] - x @ B))

        sig = sigmoid((T[y]) - x @ B)
        sig_1 = sigmoid(T[y-1] - x @ B)
        return x * (sig * (1 - sig) - sig_1 * (1 - sig_1)) / (sig - sig_1)
    
    def y_leq_j(x,y,B,T):
        if y < 0:
            return 0
        elif y == 0:
            return sigmoid((T[0]) - x @ B)
        elif y == T.size:
            return 1
        
        return sigmoid((T[y]) - x @ B)
        
    def d_loss_A_helper(x, y,B, A, m, T):
        '''
           Helper function, treating the gradient wrt A as a peicewise function
           There may be more clever implementations, leaving that to the future

           Note that this calculates the gradient for one index of the vector A

           This function is meant to be vectorized to iterate over each index m
        '''
        
        # sig = sigmoid((T[y]) - x @ B)
        # sig_1 = sigmoid(T[y-1] - x @ B)

        sig = Regress.y_leq_j(x,y,B,T)
        sig_1 = Regress.y_leq_j(x,y-1,B,T)
        gradient = 0
        d_shared = (sig_1 * (1 - sig_1) - sig * (1 - sig)) / (sig - sig_1)
        if m == 0:
            gradient = d_shared
        elif m < y:
            gradient = np.exp(A[m]) * d_shared
        elif m == y:
            gradient = -np.exp(A[m]) * (sig * (1- sig))/(sig - sig_1)
        
        return gradient
    
    vec_d_loss_A_helper = np.vectorize(d_loss_A_helper, excluded = ['x','y','B', 'A', 'T'])

    def d_loss_A(x, y, B, A, T):
        '''Calculates gradient for each index, by applying vectorized function to j'''
        j = np.arange(0, A.size)
        return Regress.vec_d_loss_A_helper(m = j, x = x, y = y, B = B, A = A, T = T)

    def Gradient_Descent(self, d_loss_B, d_loss_A, num_classes, X,y):
        '''Input: gradient of loss function for B, regression paremeters
                gradient of loss function for A, threshold paremeters
            both functions should be able to take an X vector, y value, and weight vector B

            B is initialized to 0
            T is initialized to evenly spaced partitions, it's reparameterized with log differences 
            to get A, which after finding optimal values, is converted back to T

            returns B and T
        '''
        K = num_classes
        delta = np.log(2/K)

        # initializing A, B, and T
        T = -1 + (2 * np.arange(1, K))/K
        A = np.zeros(T.size)
        # print(T[1:] - T[:-1])
        # exit()
        A[0] = T[0]
        A[1:] = np.log(T[1:] - T[:-1])
        B = np.zeros(X[0].size)

        # vectorize loss functions, exclude the weight vector so the function doesn't
        # loop over the weights
        vector_d_loss_B = np.vectorize(d_loss_B, excluded = ['B', 'T'], signature = '(m),()->(m)')
        vector_d_loss_A = np.vectorize(d_loss_A, excluded = ['B', 'A', 'T'], signature = '(m),()->(n)')

        num_points = y.size

        # will use this in loop and later as well
        J = np.arange(1, A.size + 1)

        # if X[0].size != B.size:
        #     print("ERROR not enough Rows!!")
        #     print("B: ", B)
        #     print("X: ", X[0])
        #     exit()
        
        for i in range(self.max_iter):
            T = Regress.vector_A_to_T(j = J, A = A)
            B = B - self.learning_rate * np.sum(vector_d_loss_B(X,y, B = B, T = T), axis = 0)/num_points
            A = A - self.learning_rate * np.sum(vector_d_loss_A(X,y, B = B, A = A, T = T), axis = 0)/num_points
            

        
        T = Regress.vector_A_to_T(j = J, A = A)

        return B, T
        
    def fit(self, X,y):
        '''Fits the model
           Has no return, sets model's B and T vectors
        '''
        B, T = self.Gradient_Descent(Regress.d_loss_B, Regress.d_loss_A, self.num_classes, X , y)

        self.B = B
        self.T = T

    def predict_helper_1(j,x, B, T):
        '''
        to calculate p_j = P(Y_i <= j|X) - P(Y_i <= j - 1| X)
        '''
        # print(x.size)
        # print(x)
        # exit()
        return Regress.y_leq_j(x, j, B, T) - Regress.y_leq_j(x, j - 1, B, T)
    
    vec_pred_helper_1 = np.vectorize(predict_helper_1, excluded = ['x', 'B', 'T'])

    def predict_helper_2(x, num_classes, B, T):
        '''
        inputs: a row, ie vector of features x
        '''
        j = np.arange(0,num_classes)
        p_j = Regress.vec_pred_helper_1(j, x = x, B = B, T = T)
        return np.argmax(p_j)

    vec_pred_helper_2 = np.vectorize(predict_helper_2, excluded = ['num_classes','B', 'T'], signature = '(m)->()')

    def predict(self, X):
        '''input: X, matrix of data points, each row being an observation
            
            returns y, an array of predicted labels
            must have self.B and self.T from fit defined

            returns vector of labels 
        '''
   
        return Regress.vec_pred_helper_2(X, num_classes = self.num_classes, B = self.B, T = self.T)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def prepare_data(data, Feature_Columns):
    '''
    This function is borrowed and adjusted from a different script by Finley
    '''
    X = data[Feature_Columns]
    X = preprocess(X, X.columns)
    
    y = data["Classification"].values

    split_index = int(np.floor(np.size(y) * 0.75))

    X_train = X[:split_index, :].copy()
    X_test = X[split_index:,:].copy()

    y_train = y[:split_index].copy()
    y_test = y[split_index:].copy()

    return X_train, y_train, X_test, y_test

def main(input_file):

    data = pd.read_csv(input_file)

    Feature_Columns = ["Initial Assessment","Income",
                            "Distance To Downtown","Distance To Nearest Transit Stop",
                            "Cost Per Unit","Distance to Nearest Park",
                            "Distance to Nearest Public School"]
    
    #print(data[Feature_Columns].dtypes)

    # data = data[Feature_Columns]
    # data = pd.DataFrame(preprocess(data, data.columns), columns = Feature_Columns)
    # print(data)
    X_train, y_train, X_test, y_test = prepare_data(data, Feature_Columns)

    model = Regress(1000, 0.1, 3)
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    print("Training Performance")
    print("Size: ", len(y_train))
    print("True Preds: ", sum(y_train == train_preds))
    print("Train Accuracy: ", sum(y_train == train_preds)/len(y_train))

    print("Test Performance")
    print("Size: ", len(y_test))
    print("True Preds: ", sum(y_test == test_preds))
    print("Train Accuracy: ", sum(y_test == test_preds)/len(y_test))  

    print("Model Parameters")
    print("B: ", model.B)
    print("T: ", model.T)
    return 0

if __name__ == '__main__':
    main(sys.argv[1])