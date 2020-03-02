import keras
import numpy as np
import getopt
import sys
from Model import TinySSD_model
from Utils.generator import generator
from model_parameters import SIZE, BSIZE, EPOCHS
from Utils.train_utils import w_loss, mean_iou
from Utils.data_utils import generate_list_of_paths, generate_train_test_val_paths

def main(argv):
    opts, args = getopt.getopt(argv,"p:h",["dpath=", 'help'])
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('train.py -p <data path>')
            sys.exit()
        elif opt in ("-p", "--path"):
            root = arg
    #root = 'C:\\Users\\KKP3KOR\\Desktop\\karthik\\MSc\\data'
    train_home_dir = root+'\\leftImg8bit'
    label_home_dir = root+'\\gtFine'

    #generate the list of paths
    train_path, val_path, test_path = generate_train_test_val_paths(train_home_dir)
    ltrain_path, lval_path, ltest_path = generate_train_test_val_paths(label_home_dir)
    train = generate_list_of_paths(train_path)
    val = generate_list_of_paths(val_path)

    #define model
    model = TinySSD_model.architecture()
    model.summary()
    #keras.utils.plot_model(model, 'TinySSD.png', show_shapes=True)
    #optm = keras.optimizers.SGD(lr=0.05,momentum=0.9)
    optm = keras.optimizers.Adam(lr=0.001)
    #lrate = keras.callbacks.LearningRateScheduler(decay)
    #csv_logger = keras.callbacks.CSVLogger('./tinssd_log.out', append=True, separator=';')
    model.compile(loss=w_loss,optimizer=optm,metrics=['acc'])

    #load the generator
    train_gen=generator(train=train,size=SIZE)
    train_steps = int(np.ceil(len(train)/BSIZE))
    val_gen=generator(train=val,size=SIZE,val=True)
    val_steps = int(np.ceil(len(val)/BSIZE))
    print (train_steps)
    print (val_steps)

    train_model(model, train_gen, train_steps, val_gen, val_steps)

def train_model(model,train_gen, train_steps, val_gen, val_steps, saved_weights=None, start_epoch=0):
    if saved_weights != None:
        model.load_weights('tinyssd_weight341.hdf5')
    file = open('tinyssd_log.txt', 'w')
    file.write('loss;acc;miou\n')
    max_iou = 0
    for epoch in range(start_epoch, EPOCHS):
        val_iou = []
        print ("Begining of the Epoch:", epoch)
        model.fit_generator(train_gen,steps_per_epoch=train_steps,epochs=1,
                            use_multiprocessing=False,verbose=1)
        for i in range(val_steps):
            imgs,lbls = next(val_gen)
            predicts = model.predict(imgs)
            iou = mean_iou(lbls,predicts)
            val_iou.append(iou)
        miou = np.mean(val_iou)
        file.write(str(predicts[0])+';'+str(predicts[1])+';'+str(miou)+'\n')
        print ("miou:",miou)
        if miou > max_iou:
            name = 'tinyssd_weight'+str(epoch)+'.hdf5'
            model.save_weights(name)
            print ("Model saved as", name)
            max_iou = miou
    file.close()

if __name__ == '__main__':
    main(sys.argv[1:])


