import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax,create_pairwise_bilateral
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


### generate the img

NCHAN=1
img=np.zeros((H,W,NCHAN),np.uint8)
img[H//3:2*H//3,W//4:3*W//4,:]=1

### take a look at the img
plt.imshow(np.squeeze(img),cmap=plt.get_cmap("gray_r"))
plt.show()

### use densecrf to optimize the segmentation result
pairwise_energy=create_pairwise_bilateral(sdims=(10,10),schan=(0.01,),img=img,chdim=2)
U=unary_from_softmax(probs)
d=dcrf.DenseCRF2D(W,H,NLABELS)
d.setUnaryEnergy(U)
d.addPairwiseEnergy(pairwise_energy,compat=10)

### look at intermediate solutions
Q,tmp1,tmp2=d.startInference()
for _ in range(5):
    d.stepInference(Q,tmp1,tmp2)
kl1=d.klDivergence(Q)/(H*W)
map_soln1=np.argmax(Q,axis=0).reshape((H,W))

for _ in range(20):
    d.stepInference(Q,tmp1,tmp2)
kl2=d.klDivergence(Q)/(H*W)
map_soln2=np.argmax(Q,axis=0).reshape((H,W))

for _ in range(50):
    d.stepInference(Q,tmp1,tmp2)
kl3=d.klDivergence(Q)/(H*W)
map_soln3=np.argmax(Q,axis=0).reshape((H,W))

fig=plt.figure(figsize=(15,5))
fig.add_subplot(131)
plt.title("Step 5, KL :"+str(kl1))
plt.imshow(map_soln1,cmap=plt.get_cmap("gray_r"))
fig.add_subplot(132)
plt.title("Step 25, KL :"+str(kl2))
plt.imshow(map_soln2,cmap=plt.get_cmap("gray_r"))
fig.add_subplot(133)
plt.title("Step 75, KL :"+str(kl3))
plt.imshow(map_soln3,cmap=plt.get_cmap("gray_r"))
plt.show()
