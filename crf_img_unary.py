import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
import numpy as np
import matplotlib.pyplot as plt


### generate the probability out of network
from scipy.stats import multivariate_normal
H,W,NLABELS=400,512,2

pos=np.stack(np.mgrid[0:H,0:W],axis=2)
rv = multivariate_normal([H//2, W//2], (H//4)*(W//4))
probs = rv.pdf(pos)

probs=(probs-probs.min())/(probs.max()-probs.min())
probs=0.5+0.2*(probs-0.5)

probs=np.tile(probs[np.newaxis,:,:],(2,1,1))#if the segmentation task has 2 classes, the shape is 2,h,w.
probs[1,:,:]=1-probs[0,:,:]

### take a look at the probability
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.imshow(probs[0],cmap=plt.get_cmap('gray_r'))
plt.subplot(122)
plt.imshow(probs[1],cmap=plt.get_cmap('gray_r'))
plt.show()


### run crf inference with unary potential
U=unary_from_softmax(probs)
d=dcrf.DenseCRF2D(W,H,NLABELS)
d.setUnaryEnergy(U)

Q_unary=d.inference(10)
map_soln_unary=np.argmax(Q_unary,axis=0)
map_soln_unary=map_soln_unary.reshape((H,W))

plt.imshow(map_soln_unary,cmap=plt.get_cmap('gray_r'))
plt.show()
