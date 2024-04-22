from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils import plot_model, to_categorical
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
import tensorflow as tf

from misc.utils import pre, rec, F, PlotLogger
from generator.generator_for_3d_resnet import MyGenerator
from build_models.resnet_3d import build_resnet

import argparse
import datetime
import os
import pytz
import numpy as np
import glob
import random
import json


#################
## Preparation ##
#################

def create_generators(args, B):
    
    dataset_path = os.path.join(args.base_dir, args.dataset_dir)
    
    pos_path = os.path.join(dataset_path, 'pos')
    neg_path = os.path.join(dataset_path, 'neg')
    
    fold = args.fold
    json_path = os.path.join(args.base_dir, args.folds_path)
    
    with open(json_path, 'r') as file:
        folds = json.load(file)
    
    
    train_ids = folds[str(fold)]['train']
    val_ids = folds[str(fold)]['val']

    tr_pos_list = []
    tr_neg_list = []
    
    for train_id in train_ids:
        tr_pos_list += glob.glob(os.path.join(pos_path,train_id, "*ct.npy"))
        tr_neg_list += glob.glob(os.path.join(neg_path,train_id, "*ct.npy"))

    random.shuffle(tr_neg_list)
    tr_neg_list = tr_neg_list[:(len(tr_pos_list) * 50)]

    tr_pos_list.sort()
    tr_neg_list.sort()

    print("len(tr_pos_list)", len(tr_pos_list))
    print("len(tr_neg_list)", len(tr_neg_list))

    x_train = tr_pos_list + tr_neg_list

    y_train = np.zeros(len(x_train))
    y_train[0:len(tr_pos_list)] = 1
    
    # --------------
    val_pos_list = []
    val_neg_list = []
    
    for val_id in val_ids:
        val_pos_list += glob.glob(os.path.join(pos_path,val_id, "*ct.npy"))
        val_neg_list += glob.glob(os.path.join(neg_path,val_id, "*ct.npy"))

    random.shuffle(val_neg_list)
    val_neg_list = val_neg_list[:(len(val_pos_list) * 50)]

    val_pos_list.sort()
    val_neg_list.sort()

    print("len(val_pos_list)", len(val_pos_list))
    print("len(val_neg_list)", len(val_neg_list))

    x_val = val_pos_list + val_neg_list

    y_val = np.zeros(len(x_val))
    y_val[0:len(val_pos_list)] = 1
    
    
    # Create generators.
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)

    train_gen = MyGenerator(x_train, y_train, batch_size=B, flip=1, rs=[0, 0], alpha=0.2, beta=0, ws=[2000, 400, 600, 150])
    val_gen = MyGenerator(x_val, y_val, batch_size=B, ws=[2000, 400, 600, 150])

    print("len(train_gen)", len(train_gen))
    print("len(val_gen)", len(val_gen))
    print("steps_per_epoch", len(train_gen) / 120)

    return train_gen, val_gen


###########
## Train ##
###########

def run(args):

    LR = 0.004
    E = 360
    B = 6
    
    fold = args.fold

    ### Prepare Generators ###
    train_gen, val_gen = create_generators(args, B)

    ### Create Model ###
    input_shape = train_gen.get_input_shape()
    model = build_resnet(input_shape)
    model.compile(optimizer=Adam(lr=0.004), loss="categorical_crossentropy", metrics=[pre, rec, F])
    
    time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    print(time)

    model.summary()
    path = os.path.join(args.base_dir, args.results_dir, f'fold_{fold}', "model.png")

    plot_model(model, to_file=path, show_shapes=True, show_layer_names=True)
    print("Save model.png, done.")

    ### Set Callbacks ###
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
    
    steps_per_epoch = int( np.ceil(input_shape[0] / B) )

    ### Train ###
    print("Start training...")
    model.fit_generator(train_gen, validation_data=val_gen, epochs=E, steps_per_epoch=steps_per_epoch, callbacks=cs)

    ### Validate and Save ###
    model = load_model(model_path, custom_objects={"pre": pre, "rec": rec, "F": F})
    score = model.evaluate_generator(val_gen)
    print(score)

    K.clear_session()


####################
## ArgumentParser ##
####################

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-bd", "--base_dir", type=str, default="files/ResNet/")
    parser.add_argument("-dd", "--dataset_dir", type=str, default="scans/")
    parser.add_argument("-rd", "--results_dir", type=str, default="results/")
    parser.add_argument("-fp", "--folds_path", type=str, default="folds.json")
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("-f", "--fold", type=int, default=0)
    Args = parser.parse_args()

    Args.dt = datetime.datetime.now(pytz.timezone("Asia/Tokyo")).strftime('%Y%m%d_%H%M%S')

    d = Args.results_dir
    l = [d, d + "/log", d + "/models", d + "/learning_curves", d + "/csvs"]
    for i in l:
        os.makedirs(i, exist_ok=True)

    Args.train_pos_dir = Args.root_dir + Args.val_pos_dir
    Args.train_neg_dir = Args.root_dir + Args.val_neg_dir
    Args.val_pos_dir = Args.root_dir + Args.val_pos_dir
    Args.val_neg_dir = Args.root_dir + Args.val_neg_dir

    return Args


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