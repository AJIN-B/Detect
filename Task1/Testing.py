
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings(action='ignore',category=DeprecationWarning)
import tensorflow as tf
from src.models import get_resnet152,get_EfficientNetV2M
from src.utils import plot_roc_curve
from src.helper_functions import create_dataloader
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import numpy as np
import argparse
import torch
from torch import nn
import torchvision

Root = os.getcwd()

def argumets():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tsp',"--test_path", default=os.path.join(Root,"New Masks Dataset\Validation" ), help='folder Path for testing images') 
    parser.add_argument('-f',"--freeze", default=False, help='freeze the base model layers') 
    parser.add_argument('-v',"--verbose", default=True, help='verbose to show the model outputs') 
    parser.add_argument('-M',"--model", default=3, help="select the model 1->'resnet154',2->'EfficientNetV2'") 
    parser.add_argument('-r',"--result", default=os.path.join(Root, 'results'), help='Folder for to save the results') 
    parser.add_argument('-imgpth',"--imgpath", default=os.path.join(Root, 'New Masks Dataset/Test/Mask/2070.jpg'), help='Folder for to save the results') 
    parser.add_argument('-ims',"--img_size", default=(224,224), help='Epochs for training') 
    args = parser.parse_args()
    return args


def test():
    # arguments 
    arg = argumets()
    
    model_name = {1:'resnet154',2:'EfficientNetV2',3:"ViT"}
    
    if arg.model in (1,2):
            
        if tf.test.is_gpu_available():
            print("GPU Available: ", tf.config.list_physical_devices('GPU'))
            print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        else :
            print("Cannot find GPU in the system and running on cpu")
        
        # Training images are loaded
        testData = tf.keras.utils.image_dataset_from_directory( arg.test_path, image_size=(224, 224)) 
        classes = {c:i for i,c in enumerate(testData.class_names)} 
        
        # checkpoint saving path
        checkpoint_path = Root + '/model_weights/' + model_name[arg.model] + "/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        
        # get model
        if arg.model == 1:
            model = get_resnet152((224,224,3),classes,arg.freeze) 
        elif arg.model == 2:
            model = get_EfficientNetV2M((224,224,3),classes,arg.freeze) 
        
        model.compile(loss='binary_crossentropy',metrics='accuracy',optimizer= 'adam')
        
        # Loads the weights
        model.load_weights(checkpoint_path).expect_partial()
        
        predictions = np.array([])
        labels =  np.array([])
        for x, y in testData:
            pred = model.predict(x)
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
        values.to_csv(arg.result +'/'+ model_name[arg.model] + '_performace.csv',index=False) 
        
        values.plot.bar(x='thresholds')
        plt.savefig(arg.result +'/'+ model_name[arg.model] + '_performace.png')
        plt.show() 
        # print(values) 
        
        plot_roc_curve(fpr,tpr,arg.result +'/'+ model_name[arg.model])
        
    elif arg.model == 3:

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Get pretrained weights for ViT-Base
        pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT 

        # Setup a ViT model instance with pretrained weights
        pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)

        # Freeze the base parameters
        for parameter in pretrained_vit.parameters():
            parameter.requires_grad = False
        
        class_names = ['Mask','Non Mask']
        # Change the classifier head 
        pretrained_vit.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)
        
        # print(Root + '/model_weights/' , model_name[arg.model] + ".pt") 
        # Loading the model weights
        pretrained_vit.load_state_dict(torch.load(Root + '/model_weights/' + model_name[arg.model] + ".pt"))
        
        # Get automatic transforms from pretrained ViT weights
        pretrained_vit_transforms = pretrained_vit_weights.transforms()
        NUM_WORKERS = os.cpu_count()
    
        test_dataloader, class_names = create_dataloader( test_dir=arg.test_path, 
                                                                     transform=pretrained_vit_transforms,   
                                                                     batch_size=32,  
                                                                     num_workers=NUM_WORKERS)
        
        predictions = np.array([])
        labels =  np.array([])
        for batch, (X, y) in enumerate(test_dataloader):
            X, y = X.to(device), y.to(device)
            test_pred_logits = pretrained_vit(X)
            test_pred_labels = test_pred_logits.argmax(dim=1)
            # print(test_pred_logits,y,test_pred_labels)
            predictions = np.concatenate([predictions, test_pred_labels.numpy()])
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
        values.to_csv(arg.result +'/'+ model_name[arg.model] + '_performace.csv',index=False) 
        
        values.plot.bar(x='thresholds')
        plt.savefig(arg.result +'/'+ model_name[arg.model] + '_performace.png')
        plt.show() 
        
        plot_roc_curve(fpr,tpr,arg.result +'/'+ model_name[arg.model])
        

def visualize(imgs,pred,classes):
    for img,p in zip(imgs,pred):
        plt.figure() 
        plt.imshow(img/255)
        print(img.shape)
        plt.text(int(img.shape[0]//10),int(img.shape[1]//10),classes[p], fontsize=15, color="green" if p==0 else 'red')
        plt.axis('off')
        plt.show()


def test_one_image():
    
    # arguments 
    arg = argumets()
    model_name = {1:'resnet154',2:'EfficientNetV2',3:"ViT"}
    classes = {0:'yes_mask',1:"no_mask"} 
    
    if arg.model in (1,2):
        # checkpoint saving path
        checkpoint_path = Root + '/model_weights/' + model_name[arg.model] + "/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        # get model
        if arg.model == 1:
            model = get_resnet152((224,224,3),classes,arg.freeze) 
        elif arg.model == 2:
            model = get_EfficientNetV2M((224,224,3),classes,arg.freeze) 
        
        model.compile(loss='binary_crossentropy',metrics='accuracy',optimizer= 'adam')
        # Loads the weights
        model.load_weights(checkpoint_path).expect_partial()
        
        import pandas as pd
        thresold  =  pd.read_csv(arg.result + os.sep + model_name[arg.model] + '_performace.csv').iloc[0,0]  
        
        image = tf.keras.utils.load_img(arg.imgpath,target_size=arg.img_size)
        input_arr = tf.keras.utils.img_to_array(image)
        input_arr = np.array([input_arr])  
        predictions = model.predict(input_arr)
        predictions = predictions.reshape(-1)
        predictions[predictions > thresold] = 1
        predictions[predictions <= thresold] = 0
        
        # show the output
        visualize(input_arr,predictions,classes)
        
    elif arg.model == 3:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Get pretrained weights for ViT-Base
        pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT 
        # Setup a ViT model instance with pretrained weights
        pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)
        # Freeze the base parameters
        for parameter in pretrained_vit.parameters():
            parameter.requires_grad = False
        # Change the classifier head 
        pretrained_vit.heads = nn.Linear(in_features=768, out_features=len(classes)).to(device)
        # Loading the model weights
        pretrained_vit.load_state_dict(torch.load(Root + '/model_weights/' + model_name[arg.model] + ".pt"))
        # Get automatic transforms from pretrained ViT weights
        pretrained_vit_transforms = pretrained_vit_weights.transforms()

        import pandas as pd
        thresold = pd.read_csv(arg.result + os.sep + model_name[arg.model] + '_performace.csv') 
        
        # Load in image and convert the tensor values to float32
        target_image = torchvision.io.read_image(str(arg.imgpath)).type(torch.float32)
        # Divide the image pixel values by 255 to get them between [0, 1]
        target_image = target_image / 255.0
        target_image = pretrained_vit_transforms(target_image)
        #  Make sure the model is on the target device
        pretrained_vit.to(device)
        #  Turn on model evaluation mode and inference mode
        pretrained_vit.eval()
        with torch.inference_mode():
            # Add an extra dimension to the image
            target_image = target_image.unsqueeze(dim=0)
            # Make a prediction on image with an extra dimension and send it to the target device
            target_image_pred = pretrained_vit(target_image.to(device))

        # Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
        target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
        # Convert prediction probabilities -> prediction labels
        target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
        
        visualize([plt.imread(arg.imgpath)] ,target_image_pred_label.numpy(),classes) 

if __name__ == "__main__":
    # test()
    test_one_image()


