
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings(action='ignore',category=DeprecationWarning)
import tensorflow as tf
import argparse
from src.models import get_resnet152,get_EfficientNetV2M
from src.utils import class_check,show_results


Root = os.getcwd()

def argumets():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tp',"--train_path", default=os.path.join(Root, 'New Masks Dataset\Train'), help='folder Path for training images') 
    parser.add_argument('-vp',"--val_path", default=os.path.join(Root,"New Masks Dataset\Validation" ), help='folder Path for validation images') 
    parser.add_argument('-f',"--freeze", default=False, help='freeze the base model layers') 
    parser.add_argument('-v',"--verbose", default=True, help='verbose to show the model outputs') 
    parser.add_argument('-M',"--model", default=1, help="select the model 1->'resnet154',2->'EfficientNetV2'") 
    parser.add_argument('-r',"--result", default=os.path.join(Root, 'results'), help='Folder for to save the results') 
    parser.add_argument('-e',"--epoch", default=50, help='Epochs for training') 

    args = parser.parse_args()
    return args


def train():
    
    model_name = {1:'resnet154',2:'EfficientNetV2'}
    
    if tf.test.is_gpu_available():
        print("GPU Available: ", tf.config.list_physical_devices('GPU'))
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    else :
        print("Cannot find GPU in the system")
    
    # arguments 
    arg = argumets()
    
    # Training images are loaded
    train_img = tf.keras.utils.image_dataset_from_directory( arg.train_path, image_size=(224, 224)) 
    classes = {c:i for i,c in enumerate(train_img.class_names)} 

    # validation images are loaded
    val_img = tf.keras.utils.image_dataset_from_directory( arg.val_path, image_size=(224, 224)) 
    val_cla = {c:i for i,c in enumerate(val_img.class_names)} 
    
    # checking classes are same
    class_check(classes,val_cla)
    
    # checkpoint saving path
    checkpoint_path = Root + '\\model_weights\\' + model_name[arg.model] + "/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    # get model
    if arg.model == 1:
        model = get_resnet152((224,224,3),classes,arg.freeze) 
    elif arg.model == 2:
        model = get_EfficientNetV2M((224,224,3),classes,arg.freeze) 

    # complie the model
    model.compile(loss='binary_crossentropy',metrics='accuracy',optimizer= 'adam')
    model.summary() 

    # Earlystopping if it reachs the creteria given 
    early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=25,verbose=arg.verbose
                                             ,restore_best_weights=True)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=arg.verbose
                                                    ,save_freq=5*19) 
    
    # train the model
    history = model.fit(train_img,validation_data=val_img,epochs=arg.epoch,verbose=arg.verbose ,callbacks=[early,cp_callback]) 
    
    model.save( Root + '\\model_weights\\' + model_name[arg.model] + ".h5") 
    
    # Training Results 
    show_results(history ,arg.result +'\\'+ model_name[arg.model]) 
    

if __name__ == "__main__":
    train()


