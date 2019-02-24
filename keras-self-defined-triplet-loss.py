"""
from https://kexue.fm/archives/4493
recorded for quickly finding purpose
"""

from keras.layers import Input,Embedding,LSTM,Dense,Lambda
from keras.models import Model
import keras.backend as K
from keras.layers import dot


word_size=10000
nb_features=128
encode_size=64
margin=0.1

embedding=Embedding(word_size,nb_features)
lstm=LSTM(encode_size)

def encoder(input_):
    return lstm(embedding(input_))

q_input=Input(shape=(None,))
a_right=Input(shape=(None,))
a_wrong=Input(shape=(None,))

q_encoded=encoder(q_input)
a_right_encode=encoder(a_right)
a_wrong_encode=encoder(a_wrong)

q_encode_dense=Dense(encode_size)(q_encoded)

right_cos=dot([q_encode_dense,a_right_encode],-1,normalize=True)
wrong_cos=dot([q_encode_dense,a_wrong_encode],-1,normalize=True)

triplet_loss=Lambda(lambda x:K.relu(margin+x[0]-x[1]))([wrong_cos,right_cos])

model_train=Model([q_input,a_right,a_wrong],outputs=triplet_loss)

model_train.compile(loss=lambda y_true,y_pred:y_pred,optimizer="adam")

###the shape of y is any matrix with shape: [len(q),1]
model_train.fit([q,a1,a2],y,epochs=10)
