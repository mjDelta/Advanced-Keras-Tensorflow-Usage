from keras.layers.core import Lambda
from keras.datasets import mnist
from keras.layers import Conv2D,Dense,Input,Flatten
from keras.models import Model
import numpy as np
from keras import backend as K
import tensorflow as tf
from keras.utils import to_categorical
from keras.engine.topology import Layer

##self defined layer with keras.engine.topology.Layer
class MyDenseLayer(Layer):
	def __init__(self,output_dim,activation,**kwargs):
		self.output_dim=output_dim
		self.activation=activation
		super(MyDenseLayer,self).__init__(**kwargs)
	def build(self,input_shape):
		self.kernel=self.add_weight(name="kernel",
			shape=(input_shape[1],self.output_dim),
			initializer="uniform",
			trainable=True)
		super(MyDenseLayer,self).build(input_shape)
	def call(self,x):
		temp=K.dot(x,self.kernel)
		##only support relu and sigmoid
		if self.activation=="relu":
			return K.relu(temp,alpha=0.,max_value=None)
		elif self.activation=="sigmoid":
			return K.sigmoid(temp)
		else:
			raise("Only support relu and sigmoid") 
	def compute_output_shape(self,input_shape):
		return (input_shape[0],self.output_dim)
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=np.expand_dims(x_train,axis=-1)
x_test=np.expand_dims(x_test,axis=-1)
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

##define a layer with Lambda: x/255.
preprocess_layer=Lambda(lambda x:x/255.)
def model():
	input_=Input(shape=(28,28,1))
	pre_=preprocess_layer(input_)
	print(K.shape(pre_))
	print(pre_.get_shape().as_list())
	conv_=Conv2D(16,3,padding="same",activation="relu")(pre_)
	conv_=Conv2D(16,3,padding="same",strides=2,activation="relu")(conv_)#os=2,14
	conv_=Conv2D(32,3,padding="same",activation="relu")(conv_)
	conv_=Conv2D(32,3,padding="same",strides=2,activation="relu")(conv_)#os=4,7
	conv_=Flatten()(conv_)
	dense_=MyDenseLayer(128,activation="relu")(conv_)
	dense_=MyDenseLayer(10,activation="sigmoid")(conv_)

	model=Model(input_,dense_)
	model.summary()
	return model
model=model()
model.compile(loss="categorical_crossentropy",optimizer="rmsprop")
model.fit(x_train,y_train,
	batch_size=100,
	epochs=5,
	validation_data=(x_test,y_test))
print(model.evaluate(x_test,y_test))
