import pandas as pd
import numpy as np
import CrossValidation as cv
import os
import PreprocessedDataForRegression as pr

class Regressor:

    def __init__(self, max_iter, learning_rate, num_classes, batch_size):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.batch_size = batch_size


    def tandb(self, x_values):
        self.b = np.zeros(np.size(x_values,axis=1))
        self.t = np.arange(self.num_classes - 1)

    def A_to_T(self, A, j):
        return A[0] + np.sum(np.exp(A[1:j]))

    def getAllLogits(self, x_values):
        """Gets every logit for every datapoint"""

        dot_product = np.dot(x_values, self.b).reshape(-1, 1)

        return self.t - dot_product

    def sigmoid(self, values):

        return 1 / (1 + np.exp(-values))

    def getLoss(self, logits, y_values):
        """Finds the negative log-likelihood of a set of probabilities

        Paramters
        ---------
        values: (nparray) set of logits for each data point in X
        
        Returns
        -------
        negative_log_likelihoods: (float) loss of the regression
        
        """

        cum_probs = self.sigmoid(logits)

        size = np.size(y_values)

        probs = np.hstack(((np.zeros((size,1)),cum_probs,np.ones((size,1)))))

        probs = probs[:, 1:] - probs[:, :-1]

        true_probs = probs[np.arange(size), y_values]

        return -np.sum(np.log(true_probs))


    def miniBatchGradientDescent(self, x_values, y_values):

        gen = np.random.default_rng()

        size = np.size(x_values, axis=0)

        indices = gen.integers(0, size, size = size)

        num_batches = len(y_values) // self.batch_size

        x_batches = np.array_split(x_values[indices], num_batches)
        y_batches = np.array_split(y_values[indices], num_batches)

        num_batches = len(y_batches)
        

        for i in range(self.max_iter):
            for batch in range(num_batches):
                logits = self.getAllLogits(x_batches[batch])
                probs = self.sigmoid(logits)

                grad_b = np.zeros(self.b.shape)
                grad_t = np.zeros(self.t.shape)
                
                for j in range(self.batch_size):
                    grad_b += self.bDerivative(x_batches[batch], y_batches[batch], probs, j)
                grad_b = grad_b * (1/self.batch_size)
                self.b = self.b - self.learning_rate * grad_b

                # A = np.zeros(T.size)

                # A[0] = T[0]
                # A[1:] = np.log(t[1:] - t[:-1])
                # B = np.zeros(X[0].size)
                
                for j in range(self.batch_size):
                    for k in range(self.num_classes - 1):
                        grad_t[k] += self.tDerivative(y_batches[batch], probs, j, k)
                grad_t = grad_t * 1/self.batch_size
                self.t = self.t - self.learning_rate * grad_t



    def bDerivative(self, x_values, y_values, probs, i):

        if y_values[i] == 0:
            lower = 0
            upper = probs[i,y_values[i]]
        elif y_values[i] == self.num_classes - 1:
            upper = 1   
            lower = probs[i,y_values[i] - 1] 
        else:
            lower = probs[i,y_values[i] - 1] 
            upper = probs[i,y_values[i]] 

        denom = upper-lower

        if denom == 0:
            denom = 1e-15

        sig_lower_prime = lower*(1-lower)
        sig_upper_prime = upper*(1-upper)

        grad_b = x_values[i] * 1/denom * (sig_upper_prime - sig_lower_prime)

        return grad_b


    def tDerivative(self, y_values, probs, i, k):

        if y_values[i] != k and y_values[i] != k+1:
            return 0

        if y_values[i] == 0:
            lower = 0
            upper = probs[i,y_values[i]]
        elif y_values[i] == self.num_classes - 1:
            upper = 1
            lower = probs[i,y_values[i]-1]
        else:
            lower = probs[i,y_values[i]-1]
            upper = probs[i,y_values[i]]

        denom = upper - lower
        if denom == 0:
            denom = 1e-15

        if y_values[i] == k:
            return -(upper*(1-upper))/denom
        elif y_values[i] == k+1:
            return (lower*(1-lower))/denom

    def fit(self, x_values, y_values):

        self.tandb(x_values)
        self.miniBatchGradientDescent(x_values, y_values)


    def getProbs(self, x_values):

        logits = self.getAllLogits(x_values)

        cum_probs = self.sigmoid(logits)

        size = np.size(x_values, axis = 0)

        probs = np.hstack((np.zeros((size,1)),cum_probs,np.ones((size,1))))

        probs = probs[:,1:] - probs[:, :-1]

        return probs

    def predict(self, x_values):

        probs = self.getProbs(x_values)

        classifications = np.argmax(probs, axis = 1)

        return classifications




def main():
    """Run the original custom-versus-scikit random-forest comparison."""

    k_folds = 5
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(
        parent_dir,
        "Data",
        "CSVs",
        "Sorted",
        "TH_DATA_BY_PROJECT_FINAL.csv",
    )
    df = pd.read_csv(path)
    

    feature_columns_n = [
            "Initial Assessment",
            "Income",
            "Distance To Downtown",
            "Distance To Nearest Transit Stop",
            "Cost Per Unit",
            "Distance to Nearest Park",
            "Distance to Nearest Public School",
        ]

    synth_columns_n = [
            "EstProjectCost",
            "Initial Assessment",
            "Final Assessment",
            "Income",
            "Distance To Downtown",
            "Distance To Nearest Transit Stop",
            "Cost Per Unit",
            "Distance to Nearest Park",
            "Distance to Nearest Public School",
            "Classification"
        ]

    regressor = Regressor(100, 0.01, 3, 32)
    
    custom_results = cv.syntheticKFoldCrossValidation(
        df,
        feature_columns_n,
        synth_columns_n,
        regressor,
        k_folds,
        3000,
        random_state=310,
        preprocess = True
    )

    print("Custom ordinal regressor")
    print("Confusion matrix (actual rows, predicted columns):")
    print(custom_results[0])
    print("Train accuracy:", custom_results[1])
    print("Train macro-F1:", custom_results[2])
    print("Test accuracy:", custom_results[3])
    print("Test macro-F1:", custom_results[4])


if __name__ == "__main__":
    main()





    






    