"""
from Chollet, F. (2007). MEAP Edition: Deep Learning with Python. Online, 80(1), 453. https://doi.org/citeulike-article-id:10054678
recorded for quickly finding purpose
"""
from keras.models import Model
from keras.layers import Input,Embedding,LSTM,Dense
from keras.layers import concatenate

text_vocabulary_size=10000
question_vocabulary_size=10000
answer_vocabulary_size=500
max_length=64

##Text Input Encoder
text_input=Input(shape=(None,),dtype="int32",name="text")
embedded_text=Embedding(max_length,text_vocabulary_size)(text_input)
encoded_text=LSTM(32)(embedded_text)

##Question Input Encoder
question_input=Input(shape=(None,),dtype="int32",name="question")
embedded_question=Embedding(max_length,question_vocabulary_size)(question_input)
encoded_question=LSTM(32)(embedded_question)

concatenated=concatenate([encoded_text,encoded_question],axis=-1)

answer=Dense(answer_vocabulary_size,activation="softmax")(concatenated)

model=Model([text_input,question_input],answer)
model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["acc"])

import numpy as np
num_samples=10000
text=np.random.randint(1,text_vocabulary_size,size=(num_samples,max_length))
question=np.random.randint(1,question_vocabulary_size,size=(num_samples,max_length))
answer=np.random.randint(0,1,size=(num_samples,answer_vocabulary_size))

model.fit({"text":text,"question":question},
          answer,
          epochs=10,
          batch_size=32)
