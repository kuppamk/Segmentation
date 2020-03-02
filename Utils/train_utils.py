from model_parameters import EPOCHS
import numpy as np
import keras.backend as ktf
from data_parameters import w


k = 1/np.log(1.02+np.array(w))

def decay(epoch):
    base_lr = 0.05
    maxiter = EPOCHS
    power = 0.9
    lr = base_lr*(1 - (epoch/maxiter))**power
    return lr

def w_loss(y_true,y_pred):
    weights = np.array(k)
    loss = y_true * ktf.log(y_pred+1e-10) * weights
    loss = -ktf.sum(loss, -1)
    return ktf.mean(loss)

def mean_iou(y_true,y_pred):
    true_pixels = np.argmax(y_true, axis=-1)
    pred_pixels = np.argmax(y_pred, axis=-1)
    iou = []
    for i in range(19):
        true_indices = (true_pixels==i)
        pred_indices = (pred_pixels==i)
        true_labels = np.sum(true_indices, axis=-1)
        pred_labels = np.sum(pred_indices, axis=-1)
        inter = np.sum(true_indices & pred_indices, axis=-1)
        union = np.sum(true_indices | pred_indices, axis=-1)
        legal_batches = true_labels>0
        ious = inter/union
        iou.append(np.mean(ious[legal_batches]))
    iou = np.stack(iou)
    legal_labels = ~np.isnan(iou)
    iou =iou[legal_labels]
    return np.mean(iou)
