import tensorflow as tf
import numpy as np
import time
def unpool2(pool,indexs,ksize=2,stride=2,padding='SAME'):
  pool = tf.transpose(pool, perm=[0,3,1,2])
  indexs=tf.transpose(indexs,perm=[0,3,1,2])
  pool_shape = pool.shape.as_list()
  if padding == 'VALID':
    size = (pool_shape[2] - 1) * stride + ksize
  else:
    size = pool_shape[2] * stride
  unpool_shape = [pool_shape[0], pool_shape[1], size, size]
  unpool = tf.Variable(np.zeros(unpool_shape), dtype=tf.float32)
  for batch in range(pool_shape[0]):
    for channel in range(pool_shape[1]):
      for w in range(pool_shape[2]):
        for h in range(pool_shape[3]):
          index=indexs[batch,channel,w,h]
          diff_matrix = tf.sparse_tensor_to_dense(
              tf.SparseTensor(
#                  indices=[[batch,channel,w*stride,h*stride]],
                  indices=[[batch,channel,index%size,index//size]],
                  values=tf.expand_dims(pool[batch][channel][w][h],axis=0),
                  dense_shape = [pool_shape[0],pool_shape[1],size,size]
                  ))
          unpool = unpool + diff_matrix
  unpool=tf.transpose(unpool,perm=[0,3,2,1])
  return unpool
def unpool(pool, ind, ksize=(1, 2, 2, 1), scope='unpool'):
    """
       Unpooling layer after max_pool_with_argmax.
       Args:
           pool:   max pooled output tensor
           ind:      argmax indices (produced by tf.nn.max_pool_with_argmax)
           ksize:     ksize is the same as for the pool
       Return:
           unpooled:    unpooling tensor
    """
    with tf.variable_scope(scope):
        pooled_shape = pool.get_shape().as_list()

        flatten_ind = tf.reshape(ind, (pooled_shape[0], pooled_shape[1] * pooled_shape[2] * pooled_shape[3]))
        # sparse indices to dense ones_like matrics
        one_hot_ind = tf.one_hot(flatten_ind,  pooled_shape[1] * ksize[1] * pooled_shape[2] * ksize[2] * pooled_shape[3], on_value=1., off_value=0., axis=-1)
        one_hot_ind = tf.reduce_sum(one_hot_ind, axis=1)
        one_like_mask = tf.reshape(one_hot_ind, (pooled_shape[0], pooled_shape[1] * ksize[1], pooled_shape[2] * ksize[2], pooled_shape[3]))
        # resize input array to the output size by nearest neighbor
        img = tf.image.resize_nearest_neighbor(pool, [pooled_shape[1] * ksize[1], pooled_shape[2] * ksize[2]])
        unpooled = tf.multiply(img, tf.cast(one_like_mask, img.dtype))
        return unpooled
      
def max_pool(inp,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME"):
  return tf.nn.max_pool_with_argmax(inp,ksize,strides,padding)
  
test_tensor=tf.placeholder(dtype=tf.float32,shape=[1,4,4,1])
pool,indices=max_pool(test_tensor)

out=unpool(pool,indices)

s=time.time()
for i in range(1):
  input_=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]];input_=np.array(input_);input_=np.expand_dims(input_,0);input_=np.expand_dims(input_,3)
  print(input_)
  sess=tf.Session()
  init=tf.initialize_all_variables()
  sess.run(init)
  pool_,indices_=sess.run([pool,indices],feed_dict={test_tensor:input_})
  print(pool_.shape)
  print(indices_.shape)
  print(pool_)
  print(indices_)

  out=sess.run(out,feed_dict={test_tensor:input_})
  print(out.shape)
  print(out)
print(time.time()-s)
