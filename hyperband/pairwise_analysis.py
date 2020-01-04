import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def gen_heatmap(csv_path, csv_label, csv_label_len, output_img):
    dfcsv = pd.read_csv(csv_path, header=None)
    plt.figure(figsize=(15,9))
    sns.heatmap(dfcsv,cmap="Greens")
    plt.xticks(np.arange(0.5, csv_label_len+0.5, 1), csv_label, rotation=270)
    plt.yticks(np.arange(0.5, csv_label_len+0.5, 1), csv_label, rotation='horizontal')
    plt.tight_layout()
    plt.savefig('/home/ruiliu/Development/mtml-tf/hyperband/'+output_img,format='pdf')

def compute_variance():
   None 

if __name__ == "__main__":
    #csvlabel=('(mlp,32,Adam,1,relu)','(mlp,32,SGD,1,relu)','(mlp,128,Adam,1,relu)','(mlp,128,SGD,1,relu)','(mobilenet,32,Adam)','(mobilenet,32,SGD)','(mobilenet,128,Adam)','(mobilenet,128,SGD)','(resnet,32,Adam)','(resnet,32,SGD)','(resnet,128,Adam)','(resnet,128,SGD)','(densenet,32,Adam)','(densenet,32,SGD)','(densenet, 128,Adam)','(densenet,128,SGD)')
    
    #csvlabel=('(mlp,32,Adam,1,sigmoid)','(mlp,32,Adam,1,relu)','(mlp,32,Adam,3,sigmoid)','(mlp,32,Adam,3,relu)','(mlp,32,SGD,1,sigmoid)','(mlp,32,SGD,1,relu)','(mlp,32,SGD,3,sigmoid)','(mlp,32,SGD,3,relu)','(mlp,50,Adam,1,sigmoid)','(mlp,50,Adam,1,relu)','(mlp,50,Adam,3,sigmoid)','(mlp,50,Adam,3,relu)','(mlp,50,SGD,1,sigmoid)','(mlp,50,SGD,1,relu)','(mlp,50,SGD,3,sigmoid)','(mlp,50,SGD,3,relu)')

    #csvlabel=('(mlp,32,Adam,1,sigmoid)','(mlp,32,Adam,1,relu)','(mlp,32,Adam,3,sigmoid)','(mlp,32,Adam,3,relu)','(mlp,32,SGD,1,sigmoid)','(mlp,32,SGD,1,relu)','(mlp,32,SGD,3,sigmoid)','(mlp,32,SGD,3,relu)','(mlp,50,Adam,1,sigmoid)','(mlp,50,Adam,1,relu)','(mlp,50,Adam,3,sigmoid)','(mlp,50,Adam,3,relu)','(mlp,50,SGD,1,sigmoid)','(mlp,50,SGD,1,relu)','(mlp,50,SGD,3,sigmoid)','(mlp,50,SGD,3,relu)','(mobilenet,32,Adam)','(mobilenet,32,Adam)','(mobilenet,32,Adam)','(mobilenet,32,Adam)','(mobilenet,32,SGD)','(mobilenet,32,SGD)','(mobilenet,32,SGD)','(mobilenet,32,SGD)','(mobilenet,50,Adam)','(mobilenet,50,Adam)','(mobilenet,50,Adam)','(mobilenet,50,Adam)','(mobilenet,50,SGD)','(mobilenet,50,SGD)','(mobilenet,50,SGD)','(mobilenet,50,SGD)')

    csvlabel=('(mlp, 32, Adam, relu)','(mlp, 32, SGD, relu)','(mlp, 32, Adagrad, relu)','(mlp, 32, Momentum, relu)','(mlp, 64, Adam, relu)','(mlp, 64, SGD, relu)','(mlp, 64, Adagrad, relu)','(mlp, 64, Momentum, relu)','(mlp, 128, Adam, relu)','(mlp, 128, SGD, relu)','(mlp, 128, Adagrad, relu)','(mlp, 128, Momentum, relu)','(mobilenet, 32, Adam, relu)','(mobilenet, 32, SGD, relu)','(mobilenet, 32, Adagrad, relu)','(mobilenet, 32, Momentum, relu)','(mobilenet, 64, Adam, relu)','(mobilenet, 64, SGD, relu)','(mobilenet, 64, Adagrad, relu)','(mobilenet, 64, Momentum, relu)','(mobilenet, 128, Adam, relu)','(mobilenet, 128, SGD, relu)','(mobilenet, 128, Adagrad, relu)','(mobilenet, 128, Momentum, relu)','(resnet, 32, Adam, relu)','(resnet, 32, SGD, relu)','(resnet, 32, Adagrad, relu)','(resnet, 32, Momentum, relu)','(resnet, 64, Adam, relu)','(resnet, 64, SGD, relu)','(resnet, 64, Adagrad, relu)','(resnet, 64, Momentum, relu)', '(resnet, 128, Adam, relu)','(resnet, 128, SGD, relu)','(resnet, 128, Adagrad, relu)','(resnet, 128, Momentum, relu)','(densenet, 32, Adam, relu)','(densenet, 32, SGD, relu)','(densenet, 32, Adagrad, relu)','(densenet, 32, Momentum, relu)','(densenet, 64, Adam, relu)','(densenet, 64, SGD, relu)','(densenet, 64, Adagrad, relu)','(densenet, 64, Momentum, relu)','(densenet, 128, Adam, relu)','(densenet, 128, SGD, relu)','(densenet, 128, Adagrad, relu)','(densenet, 128, Momentum, relu)')

    csv_path = 'pairwise-exp4.csv'
    output_img = 'heatmap4.pdf'
    gen_heatmap(csv_path, csvlabel, 48, output_img)