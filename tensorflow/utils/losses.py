import tensorflow as tf

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

#%%
def sce_loss(y_true, y_pred):
    y_true = tf.image.convert_image_dtype(y_true, tf.float32)
    y_pred = tf.image.convert_image_dtype(y_pred, tf.float32)
    
    # Softmax cross-entropy loss
    sce_loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
    
    return sce_loss

  
#%%
def weighted_sce_loss(y_true, y_pred, num_classes=4, loss_weights=[1, 150, 100, 1.0]):

    y_true = tf.image.convert_image_dtype(y_true, tf.float32)
    y_pred = tf.image.convert_image_dtype(y_pred, tf.float32)
    
    # print('----------------------', y_true.get_shape())
    # print('----------------------', y_pred.get_shape())
        
    # Weighted cross entropy: approach adapts following code: https://stackoverflow.com/questions/44560549/unbalanced-data-and-weighted-cross-entropy
    ce_loss = tf.keras.metrics.categorical_crossentropy(y_true, y_pred)
    
    # ce_loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)

    weighted_loss = tf.reshape(tf.constant(loss_weights), [1, 1, 1, 1, num_classes]) # Format to the right size
    weighted_one_hot = tf.reduce_sum(weighted_loss*y_true, axis = -1)
    ce_loss = ce_loss * weighted_one_hot
    
    return tf.reduce_mean(ce_loss) # Get loss
