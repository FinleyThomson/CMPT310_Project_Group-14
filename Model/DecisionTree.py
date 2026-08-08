import numpy as np
import pandas as pd

# ------------
# Constants
# ------------

# NUM_CLASSIFICATIONS = 3

# ------------------
# Hyperparameters
# ------------------

# MAX_DEPTH = 7
# MIN_SAMPLES = 1
# MIN_INFORMATION = 0

# ------------------
# Tree Node Class
# ------------------

class TreeNode():
    """Tree Node"""

    def __init__(self, data, feature_index, feature_val, prediction_probs,information_gain): #ADD DEFAULT VALUES
        self.data = data
        self.feature_index = feature_index
        self.feature_val = feature_val
        self.prediction_probs = prediction_probs
        self.information_gain = information_gain
        self.left = None
        self.right = None

# ----------------------
# Decision Tree Class
# ----------------------

class DecisionTree():
    """ Decision Tree Classifier """

    def __init__(
        self,
        max_depth,
        min_samples,
        min_information,
        num_classifications,
        num_splitting_features=None,
        random_state=None,
    ):

        """ Decision Tree Classifier, takes in max_depth, min_samples, min_information, num_classification"""

        self.max_depth = max_depth
        self.min_samples = min_samples
        self.min_information = min_information
        self.num_classifications = num_classifications
        self.num_splitting_features = num_splitting_features
        self.rng = np.random.default_rng(random_state)

    def bestSplit(self, data):
        """Finds the best split based of the lowest Gini impurity by splitting on every unique value in a given feature column
            Parameters
            ----------
            data : 
                (numpy array) Array of size number_of_features x number_of_samples
            Returns
            -------
            g1_min : 
                (numpy array) Array of values below and equal to the splitting value
            g2_min : 
                (numpy array) Array of values above the splitting value
            min_feature_val : 
                (float) Value of the splitting feature with lowest impurity
            min_feature_index : 
                (float) Index of the splitting feature with lowest impurity
            gini_min : 
                (float) Impurity of the split that has the mimimum impurity """
        gini_min = 1
        min_feature_index = None
        min_feature_val = None
        g1_min, g2_min = np.array([]), np.array([])

        total_features = np.size(data,1) - 1
        if self.num_splitting_features is None:
            num_features_to_check = int(np.sqrt(total_features)) + 1
        else:
            num_features_to_check = self.num_splitting_features
        num_features_to_check = min(max(1, num_features_to_check), total_features)

        feature_indices = self.rng.choice(
            total_features,
            num_features_to_check,
            replace=False,
        )
        for i in feature_indices:
            unique_vals = np.unique(data[:,i]) 
            if len(unique_vals) > 20:
                unique_vals = self.rng.choice(unique_vals, 20, replace=False)
            for val in unique_vals:
                g1,g2 = self.split(data,val,i)
                if len(g1) == 0 or len(g2) == 0:
                    continue
                weight1 = len(g1) / len(data)
                weight2 = len(g2) / len(data)
                gini = (weight1 * self.impurity(g1)) + (weight2 * self.impurity(g2))
                if gini < gini_min:
                    gini_min = gini
                    min_feature_index = i
                    min_feature_val = val
                    g1_min, g2_min = g1, g2

        return g1_min, g2_min, min_feature_val, min_feature_index, gini_min


    def split(self, data, val, col):
        """Splits data into two groups based on val for feature col"""

        g1 = data[data[:, col] <= val]
        g2 = data[data[:,col] > val]

        return g1, g2


    def impurity(self, data):
        """ Finds Gini impurity for a given group of data"""

        length = len(data)

        if length == 0:
            return 0

        _, counts = np.unique(data[:,-1], return_counts = True)

        probs = counts / length

        return 1 - np.sum(probs ** 2)
    

    def getClassificationProb(self, data, classification):
         """Finds the frequency of a given classification for a group of data"""

         return np.sum(data[:, -1] == classification) / np.size(data, axis=0)

    def createTree(self, data, current_depth):

        split1, split2, split_feature_val, split_feature_index, split_impurity = self.bestSplit(data)

        class_probs = []

        for i in range(self.num_classifications):
            class_probs.append(self.getClassificationProb(data, i))

        information_gain = self.impurity(data) - split_impurity

        node = TreeNode(data, split_feature_index, split_feature_val, class_probs, information_gain)

        if current_depth >= self.max_depth:
            return node

        if split_feature_index is None:
            return node

        if self.min_samples >= np.size(split1, axis = 0) or self.min_samples >= np.size(split2, axis = 0):
            return node
        elif information_gain < self.min_information:
            return node

        current_depth += 1

        node.left = self.createTree(split1, current_depth)
        node.right = self.createTree(split2, current_depth)

        return node


    def train(self, X_values, y_values):

        train_data = np.column_stack((X_values, y_values))

        self.tree = self.createTree(train_data, 0)


    def predictSample(self, x):

        node = self.tree

        while node:
            probs = node.prediction_probs

            if node.left is None and node.right is None:
                return node.prediction_probs

            if x[node.feature_index] <= node.feature_val:
                node = node.left
            else:
                node = node.right

        return probs

    def predictionProbs(self, X):

        pred_probs = np.apply_along_axis(self.predictSample, 1, X)

        return pred_probs


    def prediction(self, X):

        pred_probs = self.predictionProbs(X)
        preds = np.argmax(pred_probs, axis = 1)

        return preds

# -------------------
# Testing the tree
# -------------------

# path = "TH_DATA_BY_PROJECT_FINAL.csv" #change directory if needed, defaulting to current directory
# df = pd.read_csv(path) 

# FeatureColumns = ["Initial Assessment","Income",
#                         "Distance To Downtown","Distance To Nearest Transit Stop",
#                          "Cost Per Unit","Distance to Nearest Park",
#                         "Distance to Nearest Public School"]

# X = df[FeatureColumns].values
# y = df["Classification"].values

# split_index = int(np.floor(np.size(y)*0.75))

# X_train = X[:split_index, :].copy()
# X_test = X[split_index:,:].copy()

# y_train = y[:split_index].copy()
# y_test = y[split_index:].copy()

# tree = DecisionTree(6, 1, 0)
# tree.train(X_train,y_train)

# train_preds = tree.prediction(X_train)

# print("Training Performance")
# print("Size: ", len(y_train))
# print("True Preds: ", sum(y_train == train_preds))
# print("Train Accuracy: ", sum(y_train == train_preds)/len(y_train))

# test_preds = tree.prediction(X_test)

# print("Test Performance")
# print("Size: ", len(y_test))
# print("True Preds: ", sum(y_test == test_preds))
# print("Train Accuracy: ", sum(y_test == test_preds)/len(y_test))

