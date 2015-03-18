import numpy as np
import scipy as sp
import scipy.linalg as linalg
from numpy.linalg import inv

def my_LDA(X, Y):
    """
    Train a LDA classifier from the training set
    X: training data
    Y: class labels of training data

    """    
    
    classLabels = np.unique(Y) # different class labels on the dataset
    classNum = len(classLabels) # number of classes
    datanum, dim = X.shape # dimensions of the dataset
    totalMean = np.mean(X,0) # total mean of the data

    # ====================== YOUR CODE HERE ======================
    # Instructions: Implement the LDA technique, following the
    # steps given in the pseudocode on the assignment.
    # The function should return the projection matrix W,
    # the centroid vector for each class projected to the new
    # space defined by W and the projected data X_lda.
    
    Mj = np.zeros((classNum, dim))
    Nj = np.zeros((classNum))

    Sw = np.zeros((dim, dim))
    for j_index in range(0,classNum):
        j = classLabels[j_index]
        index_class = np.where(Y == j)[0]
        Nj[j_index] = index_class.shape[0]
        Mj[j_index,:] = np.mean(X[index_class, :],0)
        for i in index_class:
            Sw += np.outer((X[i,:] - Mj[j_index,:]),(X[i,:] - Mj[j_index,:]))

    Sb = np.zeros((dim,dim))
    for j in range(0,classNum):
        Sb += Nj[j] * np.outer((Mj[j,:] - totalMean),(Mj[j,:] - totalMean))

    St = np.zeros((dim,dim))
    for i in range(0, datanum):
        St += np.outer((X[i,:] - totalMean),(X[i,:] - totalMean))

    e_values, U = np.linalg.eig(inv(Sw).dot(Sb))
    index_sort = np.argsort(e_values)[::-1]
    e_values = e_values[index_sort]
    U = U[:, index_sort]

    W = U[:, :classNum-1]
    X_lda = X.dot(W)
    projected_centroid = Mj.dot(W)

    W = np.real(W)
    X_lda = np.real(X_lda)
    projected_centroid = np.real(projected_centroid)

    # =============================================================

    return W, projected_centroid, X_lda