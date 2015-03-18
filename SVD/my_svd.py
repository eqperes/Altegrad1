from numpy import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
from numpy.linalg import eigh
from numpy.linalg import norm

# Load the "gatlin" image data
X = loadtxt('gatlin.csv', delimiter=',')

#================= ADD YOUR CODE HERE ====================================
# Perform SVD decomposition
## TODO: Perform SVD on the X matrix
# Instructions: Perform SVD decomposition of matrix X. Save the 
#               three factors in variables U, S and V
#

w, U = eigh(dot(X, transpose(X)))
index_w = max(argsort(w)) - argsort(w)
w = w[index_w]
U = U[:, index_w]

w2, V = eigh(dot(transpose(X),X))
index_w2 = max(argsort(w2)) - argsort(w2)
w2 = w2[index_w2]
V = V[:, index_w2]

Sdiag = diag(sqrt(absolute(w)))
S = zeros(X.shape)
min_size = min(X.shape)
S[:min_size,:min_size] = Sdiag


#=========================================================================

# Plot the original image
plt.figure(1)
plt.imshow(X,cmap = cm.Greys_r)
plt.title('Original image (rank 480)')
plt.axis('off')
plt.draw()


#================= ADD YOUR CODE HERE ====================================
# Matrix reconstruction using the top k = [10, 20, 50, 100, 200] singular values
## TODO: Create four matrices X10, X20, X50, X100, X200 for each low rank approximation
## using the top k = [10, 20, 50, 100, 200] singlular values 
#

Ks = [10, 20, 50, 100, 200]
Xs = []

for k in Ks:
	Sk = S[:k, :k]
	Uk = U[:, :k]
	Vk = V[:, :k]
	Xs.append(dot(dot(Uk,Sk),transpose(Vk)))


#=========================================================================



#================= ADD YOUR CODE HERE ====================================
# Error of approximation
## TODO: Compute and print the error of each low rank approximation of the matrix
# The Frobenius error can be computed as |X - X_k| / |X|
#

error = []
for x in Xs:
	error.append((norm(X-x)/norm(X)))

plt.figure(3)
plt.plot(Ks, error)
plt.show()


#=========================================================================

Xgen = (x for x in Xs)


# Plot the optimal rank-k approximation for various values of k)
# Create a figure with 6 subfigures
plt.figure(2)

# Rank 10 approximation
plt.subplot(321)
plt.imshow(Xgen.next(),cmap = cm.Greys_r)
plt.title('Best rank' + str(5) + ' approximation')
plt.axis('off')

# Rank 20 approximation
plt.subplot(322)
plt.imshow(Xgen.next(),cmap = cm.Greys_r)
plt.title('Best rank' + str(20) + ' approximation')
plt.axis('off')

# Rank 50 approximation
plt.subplot(323)
plt.imshow(Xgen.next(),cmap = cm.Greys_r)
plt.title('Best rank' + str(50) + ' approximation')
plt.axis('off')

# Rank 100 approximation
plt.subplot(324)
plt.imshow(Xgen.next(),cmap = cm.Greys_r)
plt.title('Best rank' + str(100) + ' approximation')
plt.axis('off')

# Rank 200 approximation
plt.subplot(325)
plt.imshow(Xgen.next(),cmap = cm.Greys_r)
plt.title('Best rank' + str(200) + ' approximation')
plt.axis('off')

# Original
plt.subplot(326)
plt.imshow(X,cmap = cm.Greys_r)
plt.title('Original image (Rank 480)')
plt.axis('off')

plt.draw()


#================= ADD YOUR CODE HERE ====================================
# Plot the singular values of the original matrix
## TODO: Plot the singular values of X versus their rank k




#=========================================================================

plt.show() 

