import os
from sps import network
from sps.config import Config
from pathlib import Path
import argparse
import numpy as np
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='File path input images in the form of matrices',required=True)

args = parser.parse_args()

#x_train = []
#x_test = []
#y_train = []
#y_test = []
x =  []
y = []
i = 0 
cwd = Path(os.getcwd())
dataset = Path(str(cwd)+'/'+str(args.i))

for d in dataset.glob('*/'):
    print(str(d))
    for im in d.glob('*'):
        #print(str(im))
        image = np.loadtxt(str(im), delimiter=' ')
        image = image.reshape(Config.IMG_SHAPE,Config.IMG_SHAPE)
        M = np.max(image)
        m = np.min(image)
        image = image * 255
        image = (image-m) // (M-m)
        x.append(image.tolist())
        #label = str(d).split('/')[-1]
        #print(x)
        y.append(i)
    
    i += 1
    

x = np.array(x,dtype=int)
y = np.array(y,dtype=int)
#print(x[1][:][:])

Config.compute_k_range()
print("---starting classification---")
#network.launch_mnist()
network.launch_dataset(x,y)
