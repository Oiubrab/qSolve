# Plot ad hoc mnist instances
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy
import os
import sys

# plot show select
if not len(sys.argv)==2 or not sys.argv[1]=="show" and not sys.argv[1]=="noshow":
   print("This is a restricted area. Pleas pass in either 'show' or 'noshow' into the thing")
   exit()

# load (downloaded if needed) the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

if sys.argv=="show":
   # plot 4 images as gray scale
   plt.subplot(221)
   plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
   plt.subplot(222)
   plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
   plt.subplot(223)
   plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
   plt.subplot(224)
   plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
   # show the plot
   plt.show()

# training output
mypath = "eternalnightmare"
if not os.path.isdir(mypath):
   os.makedirs(mypath)

for x,cnt in zip(X_train,range(len(X_train))):
    numpy.savetxt("eternalnightmare/eternalnightmare"+str(cnt)+".txt",x,'%.3i')

numpy.savetxt("eternalnightmare/eternalnightmareY.txt",y_train,'%.1i')

# testing output
mypath = "byinheritance"
if not os.path.isdir(mypath):
   os.makedirs(mypath)

for x,cnt in zip(X_test,range(len(X_test))):
    numpy.savetxt("byinheritance/byinheritance"+str(cnt)+".txt",x,'%.3i')

numpy.savetxt("byinheritance/byinheritanceY.txt",y_test,'%.1i')
