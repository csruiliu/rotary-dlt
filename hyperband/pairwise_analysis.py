import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def heatmap(csv_path):
    dfcsv = pd.read_csv(csv_path, header=None)
    plt.figure(figsize=(15,9))
    sns.heatmap(dfcsv,cmap="Greens")
    plt.xticks(np.arange(0.5,32.5,1),('(mlp,32,Adam,1,sigmoid)','(mlp,32,Adam,1,relu)','(mlp,32,Adam,3,sigmoid)','(mlp,32,Adam,3,relu)','(mlp,32,SGD,1,sigmoid)','(mlp,32,SGD,1,relu)','(mlp,32,SGD,3,sigmoid)','(mlp,32,SGD,3,relu)','(mlp,50,Adam,1,sigmoid)','(mlp,50,Adam,1,relu)','(mlp,50,Adam,3,sigmoid)','(mlp,50,Adam,3,relu)','(mlp,50,SGD,1,sigmoid)','(mlp,50,SGD,1,relu)','(mlp,50,SGD,3,sigmoid)','(mlp,50,SGD,3,relu)','(mobilenet,32,Adam,sigmoid)','(mobilenet,32,Adam,relu)','(mobilenet,32,Adam,sigmoid)','(mobilenet,32,Adam,relu)','(mobilenet,32,SGD,sigmoid)','(mobilenet,32,SGD,relu)','(mobilenet,32,SGD,sigmoid)','(mobilenet,32,SGD,relu)','(mobilenet,50,Adam,sigmoid)','(mobilenet,50,Adam,relu)','(mobilenet,50,Adam,sigmoid)','(mobilenet,50,Adam,relu)','(mobilenet,50,SGD,sigmoid)','(mobilenet,50,SGD,relu)','(mobilenet,50,SGD,sigmoid)','(mobilenet,50,SGD,relu)'), rotation=270)
    #plt.gca().invert_xaxis()
    plt.yticks(np.arange(0.5,32.5,1),('(mlp,32,Adam,1,sigmoid)','(mlp,32,Adam,1,relu)','(mlp,32,Adam,3,sigmoid)','(mlp,32,Adam,3,relu)','(mlp,32,SGD,1,sigmoid)','(mlp,32,SGD,1,relu)','(mlp,32,SGD,3,sigmoid)','(mlp,32,SGD,3,relu)','(mlp,50,Adam,1,sigmoid)','(mlp,50,Adam,1,relu)','(mlp,50,Adam,3,sigmoid)','(mlp,50,Adam,3,relu)','(mlp,50,SGD,1,sigmoid)','(mlp,50,SGD,1,relu)','(mlp,50,SGD,3,sigmoid)','(mlp,50,SGD,3,relu)','(mobilenet,32,Adam,sigmoid)','(mobilenet,32,Adam,relu)','(mobilenet,32,Adam,sigmoid)','(mobilenet,32,Adam,relu)','(mobilenet,32,SGD,sigmoid)','(mobilenet,32,SGD,relu)','(mobilenet,32,SGD,sigmoid)','(mobilenet,32,SGD,relu)','(mobilenet,50,Adam,sigmoid)','(mobilenet,50,Adam,relu)','(mobilenet,50,Adam,sigmoid)','(mobilenet,50,Adam,relu)','(mobilenet,50,SGD,sigmoid)','(mobilenet,50,SGD,relu)','(mobilenet,50,SGD,sigmoid)','(mobilenet,50,SGD,relu)'), rotation='horizontal')

    plt.tight_layout()
    plt.savefig('/home/ruiliu/Development/mtml-tf/hyperband/heatmap.pdf',format='pdf')
    #plt.show()

def compute_variance():
   None 

if __name__ == "__main__":
    csv_path = 'pairwise-exp.csv'
    heatmap(csv_path)