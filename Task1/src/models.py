import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings(action='ignore',category=DeprecationWarning)
import tensorflow as tf


def get_resnet152(shape,classes,fre=False):
    
    i = tf.keras.layers.Input(shape, dtype = tf.uint8)

    x = tf.cast(i, tf.float32)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    
    core = tf.keras.applications.resnet.ResNet152(include_top=False,weights='imagenet'
                                                ,input_tensor=x,pooling='max')
    
    if not fre:
        for layer in core.layers:layer.trainable = fre
        print('Resnet layers has be freeze')
            
    x = core(x) 
    if len(classes) == 2: 
        out = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)
        #out = tf.cast(out, tf.float32)
    else:
        out = tf.keras.layers.Dense(len(classes), activation = 'softmax')(x)
    model = tf.keras.Model(inputs=[i], outputs=[out]) 
        
    # core.training = False
    # model.summary() 
    return model


def get_EfficientNetV2M(shape,classes,fre=False):
    
    i = tf.keras.layers.Input(shape, dtype = tf.uint8)
    x = tf.cast(i, tf.float32)
    
    core = tf.keras.applications.efficientnet_v2.EfficientNetV2M(include_top=False,weights='imagenet'
                                                ,input_tensor=x,pooling='max')
    
    if not fre:
        for layer in core.layers:layer.trainable = fre
        print('EfficientNetV2M layers has be freeze')    
    x = core(x) 
    
    if len(classes) == 2: 
        out = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)
    else:
        out = tf.keras.layers.Dense(len(classes), activation = 'softmax')(x)
    model = tf.keras.Model(inputs=[i], outputs=[out]) 
    
    return model