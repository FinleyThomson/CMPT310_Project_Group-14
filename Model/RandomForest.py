import numpy as np
import pandas as pd
import DecisionTree as dt
import SyntheticData as sd


# ------------
# Constants
# ------------

NUM_CLASSIFICATIONS = 3

# ------------------
# Hyperparameters
# ------------------

NUM_TREES = 100
NUM_SPLITTING_FEATURES = 5
BOOTSTRAP_SAMPLE_SIZE = 163
MAX_DEPTH = 7
MIN_SAMPLES = 1
MIN_INFORMATION = 0
WITH_REPLACEMENT = True


# ----------------------
# Random Forest Class
# ----------------------

class RandomForest():

    def __init__(self, num_trees, num_splitting_features, bootstrap_sample_size, max_depth, min_samples, min_information, num_classifications,with_replacement):
        """Creat Random Forest, takes self, num_trees, num_splitting_features, bootstrap_sample_size, max_depth, min_samples, min_information, num_classifications"""
        self.num_trees = num_trees
        self.num_splitting_features = num_splitting_features
        self.bootstrapping_sample_size = bootstrap_sample_size
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.min_information = min_information
        self.num_classifications = num_classifications
        self.with_replacement = with_replacement


    def bootstrappingSample(self, data):
        """Takes in  data (classifications included) and outputs the indices of a random sample with replacement"""

        gen = np.random.default_rng()

        high = np.size(data)

        indices = gen.choice(high, self.bootstrapping_sample_size, self.with_replacement)

        return indices


    def getSplittingFeatures(self, data):
        """Takes in data outouts a random sample of the features with size num_splitting_features"""

        high = np.size(data,1)

        gen = np.random.default_rng()

        splitting_features = gen.integers(0, high, self.num_splitting_features)

        return splitting_features

    def createForest(self, X_values, y_values):
        """Creates the forest"""

        forest = np.empty(self.num_trees, dtype = dt.DecisionTree)

        for i in range(self.num_trees):
            bootstrap_indices = self.bootstrappingSample(y_values)
            bootstrap_X = X_values[bootstrap_indices]
            bootstrap_Y = y_values[bootstrap_indices]
            
            tree = dt.DecisionTree(self.max_depth, self.min_samples, self.min_information, self.num_classifications)
            tree.train(bootstrap_X, bootstrap_Y)

            forest[i] = tree

        return forest

        
    def train(self, X_values, y_values):
    
        self.forest = self.createForest(X_values, y_values)


    def treePreditcion(self, tree: dt.DecisionTree, data):
         """Returns Prediction for a single tree on a set of data returns the array of predictions."""

         prediction_probs = tree.predictionProbs(data)

         return prediction_probs #2D Array


    def forestPrediction(self, data):
        """Returns the prediction for the whole forest of a set of data."""

        prediction_probs = []

        for i in range(self.num_trees):
            prediction_probs.append(self.treePreditcion(self.forest[i], data)) #3D array, contains the probailites for predictions for each data point by every tree

        averaged_probs = np.mean(prediction_probs, axis = 0) #2D Array with averages for each classification probability for each data point
        
        return averaged_probs


    def prediction(self, data):

        preds = np.argmax(self.forestPrediction(data), axis = 1)

        return preds

    def printForestSize(self):

        print(np.size(self.forest))



# ---------------------------------
# Testing the forest with the data (need to add k-fold cross validation)
# ---------------------------------

path = "TH_DATA_BY_PROJECT_FINAL.csv" #change directory if needed, defaulting to current directory
df = pd.read_csv(path) 

# Testing here with integrating the synthetic data, considering implementing it with cross validation when that's implemented
# We would need to generate a synthetic data set for every fold in order to prevent data leakage

s_data = sd.multivariateLognormalDistribution(df)
s_data = sd.assignClasses(s_data)

FeatureColumns = ["Initial Assessment","Income",
                        "Distance To Downtown","Distance To Nearest Transit Stop",
                         "Cost Per Unit","Distance to Nearest Park",
                        "Distance to Nearest Public School"]

X_r = df[FeatureColumns].values
y_r = df["Classification"].values

X_s = s_data[FeatureColumns].values
y_s = s_data["Class"].values

X = np.concatenate((X_r,X_s),0) 
y = np.concatenate((y_r,y_s),0)

split_index = int(np.floor(np.size(y)*0.75))

X_train = X[:split_index, :].copy()
X_test = X[split_index:,:].copy()

y_train = y[:split_index].copy()
y_test = y[split_index:].copy()

tree = RandomForest(NUM_TREES, NUM_SPLITTING_FEATURES, BOOTSTRAP_SAMPLE_SIZE, MAX_DEPTH, MIN_SAMPLES, MIN_INFORMATION, NUM_CLASSIFICATIONS, WITH_REPLACEMENT)
tree.train(X_train,y_train)

train_preds = tree.prediction(X_train)

print("Training Performance")
print("Size: ", len(y_train))
print("True Preds: ", sum(y_train == train_preds))
print("Train Accuracy: ", sum(y_train == train_preds)/len(y_train))

test_preds = tree.prediction(X_test)

print("Test Performance")
print("Size: ", len(y_test))
print("True Preds: ", sum(y_test == test_preds))
print("Train Accuracy: ", sum(y_test == test_preds)/len(y_test))  


    
