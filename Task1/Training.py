
"""  Training the models  """
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings(action='ignore',category=DeprecationWarning)
import tensorflow as tf 
import torch
import torchvision
from torch import nn
from torchinfo import summary
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import argparse
from src import engine
from src.models import get_resnet152,get_EfficientNetV2M
from src.utils import class_check,show_results
from src.helper_functions import save_model,create_dataloaders,plot_loss_curves

Root = os.getcwd()

def argumets():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tp',"--train_path", default=os.path.join(Root, 'New Masks Dataset\Train'), help='folder Path for training images') 
    parser.add_argument('-vp',"--val_path", default=os.path.join(Root,"New Masks Dataset\Validation" ), help='folder Path for validation images') 
    parser.add_argument('-f',"--freeze", default=False, help='freeze the base model layers') 
    parser.add_argument('-v',"--verbose", default=True, help='verbose to show the model outputs') 
    parser.add_argument('-M',"--model", default=3, help="select the model 1->'resnet154',2->'EfficientNetV2',3->'ViT'") 
    parser.add_argument('-r',"--result", default=os.path.join(Root, 'results'), help='Folder for to save the results') 
    parser.add_argument('-e',"--epoch", default=5, help='Epochs for training') 
    parser.add_argument('-ims',"--img_size", default=(224,224), help='Epochs for training') 
    
    args = parser.parse_args()
    return args


def train():
    # arguments 
    arg = argumets()
        
    model_name = {1:'resnet154',2:'EfficientNetV2',3:"ViT"}
    
    if arg.model in (1,2):
        
        if tf.test.is_gpu_available():
            # print("GPU Available: ", tf.config.list_physical_devices('GPU'))
            print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        else :
            print("Cannot find GPU in the system")
        
        # Training images are loaded
        train_img = tf.keras.utils.image_dataset_from_directory( arg.train_path, image_size=arg.img_size) 
        classes = {c:i for i,c in enumerate(train_img.class_names)} 

        # validation images are loaded
        val_img = tf.keras.utils.image_dataset_from_directory( arg.val_path, image_size=arg.img_size) 
        val_cla = {c:i for i,c in enumerate(val_img.class_names)} 
        
        # checking classes are same
        class_check(classes,val_cla)
        
        # checkpoint saving path
        checkpoint_path = os.path.join(Root , 'model_weights' , model_name[arg.model] , "cp.ckpt")
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
        
        model.save( os.path.join(Root, 'model_weights' , model_name[arg.model] + ".h5") )
        # Training Results 
        show_results(history ,arg.result + os.sep + model_name[arg.model]) 
    
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

        # Print a summary using torchinfo 
        summary(model=pretrained_vit, 
                input_size=(32,3, 224, 224), 
                # (batch_size, color_channels, height, width)
                # col_names=["input_size"], 
                col_names=["input_size", "output_size", "num_params", "trainable"],
                col_width=20,
                row_settings=["var_names"]
                )

        # Setup directory paths to train and test images
        train_dir = arg.train_path
        test_dir = arg.val_path

        # Get automatic transforms from pretrained ViT weights
        pretrained_vit_transforms = pretrained_vit_weights.transforms()
        print(pretrained_vit_transforms)

        NUM_WORKERS = os.cpu_count()
        # Setup dataloaders
        train_dataloader_pretrained, test_dataloader_pretrained, class_names = create_dataloaders(train_dir=train_dir,
                                                                                                            test_dir=test_dir,
                                                                                                            transform=pretrained_vit_transforms,
                                                                                                            batch_size=32,
                                                                                                            num_workers=NUM_WORKERS) 

        # Create optimizer and loss function
        optimizer = torch.optim.Adam(params=pretrained_vit.parameters(),  lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()

        # Train the classifier head of the pretrained ViT feature extractor model
        pretrained_vit_results = engine.train(model=pretrained_vit,
                                            train_dataloader=train_dataloader_pretrained,
                                            test_dataloader=test_dataloader_pretrained,
                                            optimizer=optimizer,
                                            loss_fn=loss_fn,
                                            epochs=arg.epoch,
                                            device=device)
        
        plot_loss_curves(pretrained_vit_results,arg.result + os.sep + model_name[arg.model])
        
        save_model(pretrained_vit,os.path.join(Root ,'model_weights' ), model_name[arg.model] + ".pt")
        

if __name__ == "__main__":
    train()


