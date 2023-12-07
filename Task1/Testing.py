
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings(action='ignore',category=DeprecationWarning)
import tensorflow as tf
import argparse
from src.models import get_resnet152,get_EfficientNetV2M
from src.utils import plot_roc_curve
import numpy as np
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt


Root = os.getcwd()

def argumets():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tp',"--train_path", default=os.path.join(Root, 'New Masks Dataset\Train'), help='folder Path for training images') 
    parser.add_argument('-vp',"--val_path", default=os.path.join(Root,"New Masks Dataset\Validation" ), help='folder Path for validation images') 
    parser.add_argument('-tsp',"--test_path", default=os.path.join(Root,"New Masks Dataset\Validation" ), help='folder Path for testing images') 
    parser.add_argument('-f',"--freeze", default=False, help='freeze the base model layers') 
    parser.add_argument('-v',"--verbose", default=True, help='verbose to show the model outputs') 
    parser.add_argument('-M',"--model", default=1, help="select the model 1->'resnet154',2->'EfficientNetV2'") 
    parser.add_argument('-r',"--result", default=os.path.join(Root, 'results'), help='Folder for to save the results') 
    parser.add_argument('-e',"--epoch", default=50, help='Epochs for training') 

    args = parser.parse_args()
    return args


def test():
    
    model_name = {1:'resnet154',2:'EfficientNetV2'}
    
    
    
    
    
    
    if tf.test.is_gpu_available():
        print("GPU Available: ", tf.config.list_physical_devices('GPU'))
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    else :
        print("Caninot find GPU in the system and running on cpu")
    
    # arguments 
    arg = argumets()
    
    # Training images are loaded
    testData = tf.keras.utils.image_dataset_from_directory( arg.test_path, image_size=(224, 224)) 
    classes = {c:i for i,c in enumerate(testData.class_names)} 
    
    # checkpoint saving path
    checkpoint_path = Root + '\\model_weights\\' + model_name[arg.model] + "/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    # get model
    if arg.model == 1:
        model = get_resnet152((224,224,3),classes,arg.freeze) 
    elif arg.model == 2:
        model = get_EfficientNetV2M((224,224,3),classes,arg.freeze) 
    
    model.compile(loss='binary_crossentropy',metrics='accuracy',optimizer= 'adam')
    
    # Loads the weights
    model.load_weights(checkpoint_path).expect_partial()

    # Re-evaluate the model
    # loss, acc = model.evaluate(testData, verbose=2)
    # print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
    
    predictions = np.array([])
    labels =  np.array([])
    for x, y in testData:
        pred = model.predict(x)
        # print(np.squeeze(pred),y.numpy())
        predictions = np.concatenate([predictions, np.squeeze(pred)])
        labels = np.concatenate([labels, y.numpy()]) 
        
        
    #### Calculate the ROc Curve
    fpr, tpr, thresholds = metrics.roc_curve(labels, predictions)
    
    accuracy_ls = [];f1score_ls = []
    for thres in thresholds:
        y_pred = np.where(predictions>thres,1,0)
        accuracy_ls.append(metrics.accuracy_score(labels, y_pred, normalize=True))
        f1score_ls.append(metrics.accuracy_score(labels, y_pred, normalize=True))

    values = pd.concat([pd.Series(thresholds), pd.Series(accuracy_ls), pd.Series(f1score_ls)],axis=1)
    values.columns = ['thresholds', 'accuracy','f1score'] 
    values.sort_values(by='accuracy', ascending=False, inplace=True)
    values.to_csv(arg.result +'\\'+ model_name[arg.model] + '_performace.csv',index=False) 
    
    values.plot.bar(x='thresholds')
    plt.savefig(arg.result +'\\'+ model_name[arg.model] + '_performace.png')
    # plt.show() 
    # print(values) 
    
    plot_roc_curve(fpr,tpr,arg.result +'\\'+ model_name[arg.model])
    
    # predictions[predictions > 0.5] = 1
    # predictions[predictions <= 0.5] = 0
     
    # print(tf.math.confusion_matrix(labels=labels, predictions=predictions).numpy())
    
    # f1 = metrics.f1_score(labels, predictions)
    # print("F1score : ",f1)
    
    # Training Results 
    # show_results(history ,arg.result +'\\'+ model_name[arg.model])
    
    

if __name__ == "__main__":
    test()


