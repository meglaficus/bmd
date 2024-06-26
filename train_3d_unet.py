from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras import backend as K # type: ignore
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
from keras.models import load_model
from tensorflow.keras.utils import plot_model # type: ignore
import numpy as np
import json
import tensorflow as tf
from misc.utils import PlotLogger

from generator.generator_for_3d_unet import MyGenerator
from build_models.unet_3d import build_unet

import argparse
import os
import glob
import datetime
import pytz



#############
## Metrics ##
#############

def dc(y_true, y_pred):
    y_true = K.flatten(tf.cast(y_true, tf.float64))
    y_pred = K.flatten(tf.cast(y_pred, tf.float64))
    intersection = K.sum(y_true * y_pred)
    return (2.0 * intersection + 1) / (K.sum(y_true) + K.sum(y_pred) + 1)

def se(y_true, y_pred):
    y_true = K.flatten(tf.cast(y_true, tf.float64))
    y_pred = K.flatten(tf.cast(y_pred, tf.float64))
    intersection = K.sum(y_true * y_pred)
    return (intersection + 1) / (K.sum(y_true) + 1)

def tversky_08(y_true, y_pred):
    y_true_pos = K.flatten(tf.cast(y_true, tf.float64))
    y_pred_pos = K.flatten(tf.cast(y_pred, tf.float64))
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.8
    return (true_pos + 1) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + 1)

def tversky_loss_08(y_true, y_pred):
    return 1 - tversky_08(y_true, y_pred)

def focal_tversky_loss_08(y_true, y_pred):
    pt_1 = tversky_08(y_true, y_pred)
    gamma = 0.75
    return K.pow((1 - pt_1), gamma)



##########################
## Training and Testing ##
##########################

def train(train_gen, val_gen, fold, args):

    LR = 5e-4
    E = 360
    B = args.batch_size

    input_shape = train_gen.get_input_shape()

    model = build_unet(input_shape)

    model.summary()
    
    path = os.path.join(args.base_dir, args.results_dir, f'fold_{fold}', "model.png")
    plot_model(model, to_file=path, show_shapes=True, show_layer_names=True)

    model.compile(loss=focal_tversky_loss_08, optimizer=Adam(learning_rate=LR), metrics=[dc, se])

    time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    print(time)
    
    cs = []
    csvlog_path = os.path.join(args.base_dir, args.results_dir, f'fold_{fold}', "log", time + ".csv")
    csv_logger = CSVLogger(csvlog_path)
    cs = cs + [csv_logger]
    
    pltlog_path = os.path.join(args.base_dir, args.results_dir, f'fold_{fold}', "learning_curves", time + ".png")
    plot_logger = PlotLogger(csvlog_path, pltlog_path)
    cs = cs + [plot_logger]

    
    model_path = args.base_dir + args.results_dir + f'/fold_{fold}' + "/models/" + time + ".keras"
    
    mc = ModelCheckpoint(filepath=model_path, monitor="val_loss", save_best_only=True)
    cs = cs + [mc]

    def lr_scheduler(epoch):
        initial_lr = LR  # Initial learning rate
        exponent = 0.9

        lr = initial_lr * (1 - epoch/ E) ** exponent
        return lr


    lrs = LearningRateScheduler(lr_scheduler)
    cs = cs + [lrs]
    print(len(train_gen))
    steps_per_epoch = len(train_gen)

    model.fit(train_gen, epochs=E, steps_per_epoch=steps_per_epoch,
                                       validation_data=val_gen, callbacks=cs, verbose=1)
    K.clear_session()

    model = load_model(model_path, custom_objects={"focal_tversky_loss_08": focal_tversky_loss_08, "dc": dc, "se": se})
    score = model.evaluate(val_gen)
    print(score)

    return



#########
## RUN ##
#########

def run(args):

    ####################################
    ## Prepare train_gen and test_gen ##
    ####################################

    dataset_path = os.path.join(args.base_dir, args.dataset_dir)
    B = args.batch_size
    
    json_path = os.path.join(args.base_dir, args.folds_path)
    
    fold = args.fold
    
    with open(json_path, 'r') as file:
        folds = json.load(file)
    
    
    train_ids = folds[str(fold)]['train']
    val_ids = folds[str(fold)]['val']
    
    train_ct_list = []
    train_lb_list = []
    
    for train_id in train_ids:
        train_ct_list += glob.glob(os.path.join(dataset_path,train_id, "*ct.npy"))
        train_lb_list += glob.glob(os.path.join(dataset_path,train_id, "*lb.npy"))
    
    train_ct_list.sort()
    train_lb_list.sort()
    
    val_ct_list = []
    val_lb_list = []
    for val_id in val_ids:
        val_ct_list += glob.glob(os.path.join(dataset_path,val_id, "*ct.npy"))
        val_lb_list += glob.glob(os.path.join(dataset_path,val_id, "*lb.npy"))
    
    val_ct_list.sort()
    val_lb_list.sort()


    train_gen = MyGenerator(train_ct_list, train_lb_list, batch_size=B, flip=1, alpha=0.2, beta=0.3, ws=[2000, 400, 600, 150])
    val_gen = MyGenerator(val_ct_list, val_lb_list, batch_size=B, flip=0, alpha=0, beta=0, ws=[2000, 400, 600, 150])

    d = os.path.join(args.base_dir, args.results_dir, f'fold_{fold}')
    l = [d, d + "/log", d + "/models", d + "/learning_curves", d + "/predicted_images"]
    for i in l:
        if not (os.path.exists(i)):
            os.makedirs(i)
    
    ###########
    ## Train ##
    ###########

    train(train_gen, val_gen, fold, args)

    print("Done")

    return



##############
## argparse ##
##############

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-bd", "--base_dir", type=str, default="files/UNet/")
    parser.add_argument("-dd", "--dataset_dir", type=str, default="scans/")
    parser.add_argument("-rd", "--results_dir", type=str, default="results/")
    parser.add_argument("-fp", "--folds_path", type=str, default="folds.json")
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("-f", "--fold", type=int, default=0)
    
    args = parser.parse_args()

    return args

##########
## Main ##
##########

if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if gpus: 
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
        )

    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    
    args = get_args()
    print(args)
    run(args)