import os
import matplotlib.pyplot as plt




def show_results(history ,path,save_results=True):
    
    plt.figure()
    plt.plot(history.history["accuracy"], label="training accuracy")
    plt.plot(history.history["val_accuracy"], label="validation accuracy")
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    if save_results:plt.savefig(path+'_Accuracy.png')
    plt.show()

    plt.figure()
    plt.plot(history.history["loss"], label="training loss")
    plt.plot(history.history["val_loss"], label="validation loss")
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    if save_results:plt.savefig(path+'_loss.png')
    plt.show()

def plot_roc_curve(fpr, tpr,path,save_results=True):
    plt.figure()
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    if save_results:plt.savefig(path+'_roc.png')
    plt.show()

def class_check(tr,vl):
    if len(tr) != len(vl):
        raise ValueError(f'Training and validation classes are not equal') 
    
    for t,v in zip(tr.items(),vl.items()):
        if t[0] != v[0] or t[1] != v[1]:
            raise ValueError(f'Mismatch in classes inside the Training and validation ') 

