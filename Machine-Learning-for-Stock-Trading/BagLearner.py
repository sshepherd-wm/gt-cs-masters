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
  		  	   		   	 			  		 			 	 	 		 		 	
class BagLearner(object):                   
    """                   
    This is a wrapper for a BagLearner that trains an ensemble of learners on bootstrap samples of training data.                   
                   
    :param verbose: If “verbose” is True, your code can print out information for debugging.                   
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.                   
    :type verbose: bool                   
    """                   
    def __init__(self, learner, classification=True, kwargs={}, bags=10, boost=False, verbose=False):                   
        """                   
        Constructor method                   
        """                   
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.learners = []
        self.data_y = None
        self.classification = classification

        ## initialize a learner for each bag of data
        for i in range(bags):
            self.learners.append(learner(**kwargs))
                   
    def author(self):                   
        """                   
        :return: The GT username of the student                   
        :rtype: str                   
        """                   
        return "sshepherd35"  # replace tb34 with your Georgia Tech username

                   
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

        ## bootstrap sampling and training
        for l in self.learners:
            idx = np.random.choice(data.shape[0], data.shape[0], replace=True)
            samp = data[idx,:]
            l.add_evidence(samp[:,1:], samp[:,0])
        
                   
    def query(self, points):                   
        """                   
        Estimate a set of test points given the model we built.                   
                   
        :param points: A numpy array with each row corresponding to a specific query.                   
        :type points: numpy.ndarray                   
        :return: The predicted result of the input data according to the trained model                   
        :rtype: numpy.ndarray                   
        """
        ## initialize all predictions array by having the first learner query results
        all_preds = np.array(self.learners[0].query(points))
        
        ## query remaining learners (if there are any), track predictions
        for l in self.learners[1:]:
            preds = l.query(points)
            all_preds = np.vstack((all_preds,preds))

        if len(self.learners) > 1:
            ## average the predictions to return an ensemble consensus/vote
            if self.classification:
                ensemble_preds = stats.mode(all_preds).mode
            else:
                ensemble_preds = np.mean(all_preds, axis=0)
        else:
            return all_preds

        ## fill with y avg if prediction is null for some reason
        # for i in range(len(ensemble_preds)):
        #     p = ensemble_preds[i]
        #     if not np.isfinite(p):
        #         if self.classification:
        #             ensemble_preds[i] = stats.mode(self.data_y).mode[0]
        #         else:
        #             ensemble_preds[i] = np.mean(self.data_y)
        
        return ensemble_preds.flatten()
                   
                   
if __name__ == "__main__":                   
    print("construct then train learner, please.")