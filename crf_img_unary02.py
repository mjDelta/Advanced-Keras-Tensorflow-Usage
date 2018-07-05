import numpy as np
import pydensecrf.densecrf as dcrf
import scipy.io
### from https://github.com/lucasb-eyer/pydensecrf/issues/58

# Load prediction image, each pixel has three probabilities, one for each class
Predictions = scipy.io.loadmat('Predictions_Image.mat')
predictions = Predictions['prediction_image']

# Define the CRF, we have three classes in this case
d = dcrf.DenseCRF2D(predictions.shape[0], predictions.shape[1], 3)

# Unary potentials
U = predictions.transpose(2,0,1).reshape((3, -1))

# Take negative logarithm since these are probabilities
d.setUnaryEnergy(-np.log(U))

d.addPairwiseGaussian(sxy=(3,3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization = dcrf.NORMALIZE_SYMMETRIC)

# Inference
Q = d.inference(5)

map = np.argmax(Q, axis=0).reshape((predictions.shape[0], predictions.shape[1]))

crf_results = np.array(map)

scipy.io.savemat("Predictions_After_CRF.mat", mdict={'CRF_Results': crf_results})

from matplotlib import pyplot as plt

plt.imshow(predictions)
plt.show() 

plt.imshow(map)
plt.show()
