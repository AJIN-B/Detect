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
    # plt.show()
    if save_results:plt.savefig(path+'_Accuracy.png')

    plt.figure()
    plt.plot(history.history["loss"], label="training loss")
    plt.plot(history.history["val_loss"], label="validation loss")
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    # plt.show()
    if save_results:plt.savefig(path+'_loss.png')

def plot_roc_curve(fpr, tpr,path,save_results=True):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    # plt.show()
    if save_results:plt.savefig(path+'_roc.png')

def class_check(tr,vl):
    if len(tr) != len(vl):
        raise ValueError(f'Training and validation classes are not equal') 
    
    for t,v in zip(tr.items(),vl.items()):
        if t[0] != v[0] or t[1] != v[1]:
            raise ValueError(f'Mismatch in classes inside the Training and validation ') 

def path_exist(path):
    if os.path.exists(path):
        pass
    # else:
    #     raise()

# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# # Create figure with secondary y-axis
# fig = make_subplots(specs=[[{"secondary_y": True}]])
# # Add traces
# fig.add_trace( go.Scatter( y=history.history['val_loss'], name="val_loss"), secondary_y=False,)
# fig.add_trace( go.Scatter( y=history.history['loss'], name="loss"), secondary_y=False,)
# fig.add_trace(go.Scatter( y=history.history['val_accuracy'], name="val accuracy"),secondary_y=True,)
# fig.add_trace( go.Scatter( y=history.history['accuracy'], name="val accuracy"),    secondary_y=True,)
# # Add figure title
# fig.update_layout( title_text="Loss/Accuracy of LSTM Model")
# # Set x-axis title
# fig.update_xaxes(title_text="Epoch")
# # Set y-axes titles
# fig.update_yaxes(title_text="<b>primary</b> Loss", secondary_y=False)
# fig.update_yaxes(title_text="<b>secondary</b> Accuracy", secondary_y=True)
# fig.show()