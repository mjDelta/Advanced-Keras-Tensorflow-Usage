from __future__ import print_function
from keras.models import Sequential
from keras.layers import Activation,LSTM,TimeDistributed,RepeatVector,Dense
import numpy as np
from six.moves import range

class  CharacterTable(object):
	"""docstring for  CharacterTable"""
	def __init__(self, chars):
		self.chars=sorted(set(chars))
		self.char_indices=dict((c,i) for i,c in enumerate(self.chars))
		self.indices_char=dict((i,c) for i,c in enumerate(self.chars))
	def encode(self,s,num_rows):
		"""turn the character into one hot"""
		x=np.zeros((num_rows,len(self.chars)))
		for i,c in enumerate(s):
			x[i,self.char_indices[c]]=1
		return x
	def decode(self,x,calc_argmax=True):
		if calc_argmax:
			x=x.argmax(axis=-1)
		return "".join(self.indices_char[i] for i in x)

class colors:
	ok="\033[92m"
	fail="\033[91m"
	close="\033[0m"

TRAINING_SIZE=50000
DIGITS=3
REVERSE=True

MAXLEN=DIGITS+1+DIGITS

chars="0123456789+ "
ctable=CharacterTable(chars)

questions=[]
expected=[]
seen=set()
print("Begin data generating...")
while len(questions)<TRAINING_SIZE:
	f=lambda:int("".join(
		np.random.choice(list("0123456789")) for i in range(np.random.randint(1,DIGITS+1))
		)
	)
	a,b=f(),f()

	key=tuple(sorted((a,b)))
	if key in seen:
		continue
	seen.add(key)
	q="{}+{}".format(a,b)
	query=q+' '*(MAXLEN-len(q))
	ans=str(a+b)
	ans+=' '*(DIGITS+1-len(ans))
	if REVERSE:
		query=query[::-1]
	questions.append(query)
	expected.append(ans)
print("Total addition questions:",len(questions))

print("Vectorizing...")
x=np.zeros((len(questions),MAXLEN,len(chars)),dtype=np.bool)
y=np.zeros((len(expected),DIGITS+1,len(chars)),dtype=np.bool)

for i in range(len(questions)):
	x[i]=ctable.encode(questions[i],MAXLEN)
	y[i]=ctable.encode(expected[i],DIGITS+1)

##Shuffle the indices
indices=np.arange(len(questions))
np.random.shuffle(indices)
x=x[indices]
y=y[indices]

split_at=int(0.8*len(x))

x_train,x_val=x[:split_at],x[split_at:]
y_train,y_val=y[:split_at],y[split_at:]

HIDDEN_SIZE=128
BATCH_SIZE=128
LAYERS=1

model=Sequential()
model.add(LSTM(HIDDEN_SIZE,input_shape=(MAXLEN,len(chars))))
model.add(RepeatVector(DIGITS+1))

for _ in range(LAYERS):
	model.add(LSTM(HIDDEN_SIZE,return_sequences=True))

model.add(TimeDistributed(Dense(len(chars))))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy",
							optimizer="adam",
							metrics=["accuracy"])

model.summary()

for i in range(1,200):
	print("Iteration ",i)
	model.fit(x_train,y_train,
						batch_size=BATCH_SIZE,
						epochs=1,
						validation_data=(x_val,y_val))
	for j in range(10):
		ind=np.random.randint(0,len(x_val))
		q_,a_=x_val[ind],y_val[ind]
		pred=model.predict_classes(np.expand_dims(q_,axis=0),verbose=0)
		q=ctable.decode(q_)
		true=ctable.decode(a_)
		print(q[::-1] if REVERSE else q,end=' ')
		print('= ?',end=' ')
		print(true,end=' ')
		guess=ctable.decode(pred[0],calc_argmax=False)
		if guess==true:
			print(colors.ok+'yes'+colors.close,end=' ')
		else:
			print(colors.fail+'no'+colors.close,end=' ')
		print(guess)
