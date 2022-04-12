""""""  		  	   		   	 			  		 			 	 	 		 		 	
"""  		  	   		   	 			  		 			 	 	 		 		 	
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 			  		 			 	 	 		 		 	
Atlanta, Georgia 30332  		  	   		   	 			  		 			 	 	 		 		 	
All Rights Reserved  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
Template code for CS 4646/7646  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 			  		 			 	 	 		 		 	
works, including solutions to the projects assigned in this course. Students  		  	   		   	 			  		 			 	 	 		 		 	
and other users of this template code are advised not to share it with others  		  	   		   	 			  		 			 	 	 		 		 	
or to make it available on publicly viewable websites including repositories  		  	   		   	 			  		 			 	 	 		 		 	
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 			  		 			 	 	 		 		 	
or edited.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
We do grant permission to share solutions privately with non-students such  		  	   		   	 			  		 			 	 	 		 		 	
as potential employers. However, sharing with other current or future  		  	   		   	 			  		 			 	 	 		 		 	
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 			  		 			 	 	 		 		 	
GT honor code violation.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
-----do not edit anything above this line---  		  	   		   	 			  		 			 	 	 		 		 	
"""  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
import numpy as np  
from scipy import stats		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
class RTLearner(object):                   
    """                   
    This is a Linear Regression Learner. It is implemented correctly.                   
                   
    :param verbose: If “verbose” is True, your code can print out information for debugging.                   
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.                   
    :type verbose: bool                   
    """                   
    def __init__(self, leaf_size=5, classification=True, verbose=False):                   
        """                   
        Constructor method           
        """                   
        self.leaf_size = leaf_size
        self.root = None
        self.data_y = None
        self.verbose = verbose
        self.classification = classification
                   
    def author(self):                   
        """                   
        :return: The GT username of the student                   
        :rtype: str                   
        """                   
        return "sshepherd35"  # replace tb34 with your Georgia Tech username

    def build_tree(self, data, leaf_size=1):
        '''
        Builds the tree in a table data structure
        Each row is a node with [column index of predictor, splitval, left index addition, right index addition]
        If row is a leaf, the splitval represents the predicted label
        To get the index of the row containing the left node off the current, add the (left index addition) to the current index
        '''

        ## leaf values to return
        if self.classification:
            leaf_return = np.array([['leaf', stats.mode(data[:,0]).mode[0], None, None]])
        else:
            leaf_return = np.array([['leaf', np.mean(data[:,0]), None, None]])

        ## case when smaller than leaf size
        if data.shape[0] <= leaf_size:
            return leaf_return
        ## case when all target values are the same
        if min(data[:,0]) == max(data[:,0]):
            return leaf_return

        ## otherwise make sub-trees
        else:
            i = np.random.choice(range(1, data.shape[1])) ## features start on second column
            SplitVal = np.median(data[:,i]) ## get split value

            ## case when all values
            if np.round(SplitVal, 6) == np.round(np.min(data[:,i]), 6) == np.round(np.max(data[:,i]), 6):
                return leaf_return
            ## case when lopsided split
            if np.round(SplitVal, 6) in [np.round(np.min(data[:,i]), 6), np.round(np.max(data[:,i]), 6)]:
                return leaf_return
            else:
                lefttree  = self.build_tree(data[data[:,i] <= SplitVal], leaf_size=leaf_size)
                righttree = self.build_tree(data[data[:,i] >  SplitVal], leaf_size=leaf_size)

            ## i - 1 to account for target variable being first column learning set
            root = np.array([[i - 1, SplitVal, 1, lefttree.shape[0] + 1]])

            return np.concatenate((root, lefttree, righttree))

    def query_tree(self, tree, point):
        '''
        Queries tree to predict a particular data point
        '''
        idx = 0 ## start at root

        while True:
            node = tree[idx] ## get next node
            if node[2] is None and node[3] is None: ## if leaf, return predicted value
                return node[1]

            ## test appropriate feature value for a split
            if point[int(node[0])] <= node[1]:
                idx += int(node[2]) ## fork left
            else:
                idx += int(node[3]) ## fork right

                   
    def add_evidence(self, data_x, data_y):                   
        """                   
        Add training data to learner                   
                   
        :param data_x: A set of feature values used to train the learner                   
        :type data_x: numpy.ndarray                   
        :param data_y: The value we are attempting to predict given the X data                   
        :type data_y: numpy.ndarray                   
        """         
        self.data_y = data_y

        data = np.append(np.reshape(data_y, (-1,1)), data_x, 1) ## make y the first column
        self.root = self.build_tree(data, self.leaf_size)
        
                   
    def query(self, points):                   
        """                   
        Estimate a set of test points given the model we built.                   
                   
        :param points: A numpy array with each row corresponding to a specific query.                   
        :type points: numpy.ndarray                   
        :return: The predicted result of the input data according to the trained model                   
        :rtype: numpy.ndarray                   
        """                   
        preds = np.array([])
        for p in points:
            pred = self.query_tree(self.root, p)
            preds = np.append(preds, pred)

        ## fill with y avg if prediction is null for some reason
        # for i in range(len(preds)):
        #     p = preds[i]
        #     if not np.isfinite(p):
        #         if self.classification:
        #             preds[i] = stats.mode(self.data_y).mode[0]
        #         else:
        #             preds[i] = np.mean(self.data_y)

        return preds
                   
                   
if __name__ == "__main__":                   
    print("construct then train learner, please.")
