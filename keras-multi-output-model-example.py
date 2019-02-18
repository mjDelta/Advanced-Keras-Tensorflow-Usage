"""
from Chollet, F. (2007). MEAP Edition: Deep Learning with Python. Online, 80(1), 453. https://doi.org/citeulike-article-id:10054678
recorded for quickly finding purpose
codes can't be run directly
"""

from keras.layers import Input,Dense,Conv1D,MaxPooling1D,GlobalMaxPooling1D,Embedding
from keras.models import Model

vocabulary_size=50000
num_income_groups=10

posts_input=Input(shape=(None,),dtype="int32",name="posts")
embedded_posts=Embedding(256,vocabulary_size)(posts_input)
x=Conv1D(128,5,activation="relu")(embedded_posts)
x=MaxPooling1D(5)(x)
x=Conv1D(256,5,activation="relu")(x)
x=Conv1D(256,5,activation="relu")(x)
x=MaxPooling1D(5)(x)
x=Conv1D(256,5,activation="relu")(x)
x=Conv1D(256,5,activation="relu")(x)
x=GlobalMaxPooling1D()(x)
x=Dense(128,activation="relu")(x)

##multi-outputs
age_prediction=Dense(1,name="age")(x)
income_prediction=Dense(num_income_groups,name="income",activation="softmax")(x)
gender_prediction=Dense(1,activation="sigmoid",name="gender")(x)
model=Model(posts_input,[age_prediction,income_prediction,gender_prediction])

###multi-outputs compile
model.compile(optimizer="rmsprop",
              loss={"age":"mse",
                    "income":"categorical_crossentropy",
                    "gender":"binary_crossentropy"},
              loss_weights={"age":0.25,
                            "income":1,
                            "gender":10})

###suppose we have data already
model.fit(posts,{"age":age_targets,
                 "income":income_targets,
                 "gender":gender_targets},
                 epochs=10,batch_size=32)
