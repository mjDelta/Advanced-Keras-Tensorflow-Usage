# Advanced-Keras-Usage
Some advanced keras usage, like self-defined layer...</br>
1.<a href="https://github.com/mjDelta/Advanced-Keras-Usage/blob/master/self_defined_layer.py">self-defined layer</a></br>
Self-defined Layer by `keras.layers.core.Lambda` or `keras.engine.topology.Layer`.</br>
2.<a href="https://github.com/mjDelta/Advanced-Keras-Usage/blob/master/seq_to_seq_addition.py">seq_to_seq_addition</a></br>
Use the `many-to-many` structure in RNN to complete the addition of two numbers. Encode numbers in a new encoding way, and use the seq_to_seq model to solve it.</br>
3.<a href="https://github.com/mjDelta/Advanced-Keras-Usage/blob/master/unpooling.py">unpooling</a></br>
Unpooling is a popular method in the semantic segmentation task to segment not so-clearly border between objects. It works like below.</br>
![Image text](https://github.com/mjDelta/Advanced-Keras-Tensorflow-Usage/blob/master/imgs/unpooling.png)</br>
4.<a href="https://github.com/mjDelta/Advanced-Keras-Tensorflow-Usage/blob/master/crf_text.py">crf_text</a></br>
CRF belongs to the sequence family, considering "neighboring" smaples, as well as the full context, using it ```keras_contrib.layers.CRF```</br>
5.<a href="https://github.com/mjDelta/Advanced-Keras-Tensorflow-Usage/blob/master/crf_img_unary.py">crf_img_unary</a></br>
6.<a href="https://github.com/mjDelta/Advanced-Keras-Tensorflow-Usage/blob/master/crf_img_bilateral.py">crf_img_bilateral</a></br>
CRF belongs to the sequence family, considering "neighboring" smaples, as well as the full context, using it ```pydensecrf```</br>

