import numpy as np
import scipy as sp
import scipy.linalg as linalg


def predict(X, projected_centroid, W):
    
    """Apply the trained LDA classifier on the test data 
    X: test data
    projected_centroid: centroid vectors of each class projected to the new space
    W: projection matrix computed by LDA
    """


    # Project test data onto the LDA space defined by W
    projected_data  = np.dot(X, W)

    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in the code to implement the classification
    # part of LDA. Follow the steps given in the assigment.
    
    label = []
    num, dim = projected_data.shape
    classNum, dim = projected_centroid.shape

    for i in range(0, num):
        distances = []
        for j in range(0, classNum):
            distances.append(np.linalg.norm(projected_data[i,:] - projected_centroid[j,:]))
        label.append(np.argmin(distances))
    
    
    
    # =============================================================

    # Return the predicted labels of the test data X
    return np.array(label)
