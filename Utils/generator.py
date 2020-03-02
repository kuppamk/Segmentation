import numpy as np
import cv2
from Utils.augmentation import augmentation
from Utils.data_utils import  labeling_img, read_label
from model_parameters import BSIZE
from data_parameters import N


def generator(train,batchsize=BSIZE,size=(2048,1024,3),val=False,n_cls=N):
    N = len(train)
    steps = int(np.ceil(N/batchsize))
    train_sf = np.random.permutation(train)
    while True:
        for batch in range(steps):
            if ((batch+1)*batchsize)<N:
                batch_data = train_sf[batch*batchsize:(batch+1)*batchsize]
            else:
                batch_data = train_sf[batch*batchsize:]
            images = np.zeros((batchsize,size[0],size[1],3),np.float32)
            labels = np.zeros((batchsize,size[0],size[1],n_cls),np.uint8)
            for i,path in enumerate(batch_data):
                img = cv2.imread(path)/255.0
                img = cv2.resize(img,(size[1],size[0]))
                lbl = read_label(path,(size[1],size[0]))
                #print (img.shape, lbl.shape)
                if (np.random.random() < 0.5) & (val==False):
                    n = np.random.randint(1,6)
                    img,lbl = augmentation(n,img,lbl)
                lbl = labeling_img(lbl)
                #lbl = np.reshape(lbl,(size[0]*size[1],-1))
                images[i,:,:,:] = img
                labels[i,:,:] = lbl #cv2.imread(os.path.join(l_path,path.split('/')[-1]),0)
            yield images,labels