"""
from https://kexue.fm/archives/4493
recorded for quickly finding purpose
"""

from keras.layers import Input,Embedding,LSTM,Dense
from keras.models import Model
import keras.backend as K


word_size=10000
nb_features=128
nb_classes=10

input=Input(shape=(None,))
encoder=Embedding(word_size,nb_features)(input)
predict=Dense(nb_classes,activation="softmax")(encoder)

def my_crossentropy(y_true,y_pred,e=0.1):
  loss1=K.categorical_crossentropy(y_true,y_pred)
  loss2=K.categorical_crossentropy(K.ones_like(y_pred)/nb_classes,y_pred)
  return (1-e)*loss1+e*loss2

model=Model(input,predict)
model.compile(loss=my_crossentropy,optimizer="adam")

