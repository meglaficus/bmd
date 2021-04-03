from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils import plot_model, to_categorical
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger

from misc.utils import pre, rec, F
from generator.generator_for_3d_resnet import MyGenerator
from build_models.resnet_3d import build_resnet

import argparse
import datetime
import os
import pytz
import numpy as np
import glob
import random


#################
## Preparation ##
#################

def create_generators(args, B):

    ### Prepare training data ###
    tr_pos_list = glob.glob(args.train_pos_dir + "/**/*.npy", recursive=True)
    tr_neg_list = glob.glob(args.train_neg_dir + "/**/*.npy", recursive=True)

    random.shuffle(tr_neg_list)
    tr_neg_list = tr_neg_list[:(len(tr_pos_list) * 50)]

    tr_pos_list.sort()
    tr_neg_list.sort()

    print("len(tr_pos_list)", len(tr_pos_list))
    print("len(tr_neg_list)", len(tr_neg_list))

    x_train = tr_pos_list + tr_neg_list

    y_train = np.zeros(len(x_train))
    y_train[0:len(tr_pos_list)] = 1

    ### Prepare validation data ###
    val_pos_list = glob.glob(args.val_pos_dir + "/**/*.npy", recursive=True)
    val_neg_list = glob.glob(args.val_neg_dir + "/**/*.npy", recursive=True)

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


def set_callbacks(args, E, LR):

    cs = []
    csvlog_path = args.results_dir + "/log/" + args.dt + ".csv"
    csv_logger = CSVLogger(csvlog_path)
    cs = cs + [csv_logger]

    model_path = args.results_dir + "/models/" + args.dt + ".h5"
    mc = ModelCheckpoint(filepath=model_path, monitor="val_loss", save_best_only=True)
    cs = cs + [mc]

    def step_decay(epoch):
        x = LR
        if epoch > E * 2 / 3:
            x = LR / 10
        if epoch > E * 8 / 9:
            x = LR / 100
        return x

    lrs = LearningRateScheduler(step_decay)
    cs = cs + [lrs]

    return cs


###########
## Train ##
###########

def run(args):

    LR = 0.004
    E = 180
    SPE = 1/120
    B = 128

    ### Prepare Generators ###
    train_gen, val_gen = create_generators(args, B)

    ### Create Model ###
    input_shape = train_gen.get_input_shape()
    model = build_resnet(input_shape)
    model.compile(optimizer=Adam(lr=0.004), loss="categorical_crossentropy", metrics=[pre, rec, F])

    model.summary()
    path = args.results_dir + "model.png"
    plot_model(model, to_file=path, show_shapes=True, show_layer_names=True)
    print("Save model.png, done.")

    ### Set Callbacks ###
    cs = set_callbacks(args, E, LR)

    ### Train ###
    print("Start training...")
    model.fit_generator(train_gen, validation_data=val_gen, epochs=E, steps_per_epoch=len(train_gen) * SPE, callbacks=cs)

    ### Validate and Save ###
    model_path = args.results_dir + "models/" + args.dt + ".h5"
    model = load_model(model_path, custom_objects={"pre": pre, "rec": rec, "F": F})
    score = model.evaluate_generator(val_gen)
    print(score)

    K.clear_session()


####################
## ArgumentParser ##
####################

def set_args():

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument("--root_dir", type=str, default="/xxx/dir2")
    parser.add_argument("--train_pos_dir", type=str, default="/train_pos")
    parser.add_argument("--train_neg_dir", type=str, default="/train_neg")
    parser.add_argument("--val_pos_dir", type=str, default="/val_pos")
    parser.add_argument("--val_neg_dir", type=str, default="/val_neg")
    parser.add_argument("--results_dir", type=str, default="./results_3d_resnet")
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

    Args = set_args()
    run(Args)