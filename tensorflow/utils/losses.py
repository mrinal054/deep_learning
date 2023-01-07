import tensorflow as tf
import tensorflow.keras.backend as K
  
  
#%%
def weighted_cce_loss(y_true, y_pred, num_classes=2, loss_weights = [1, 150.0]):
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    
    cce = tf.keras.losses.CategoricalCrossentropy()
    cce_loss = cce(y_true, y_pred)

    weighted_loss = tf.reshape(tf.constant(loss_weights), [1, 1, num_classes]) # Format to the right size
    
    #weighted_loss = tf.cast(weighted_loss, float32)
    
    weighted_one_hot = tf.reduce_sum(weighted_loss*y_true, axis = -1)
    cce_loss = cce_loss * weighted_one_hot
    
    return tf.reduce_mean(cce_loss)


