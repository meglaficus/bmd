from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras import backend as K # type: ignore

from misc.utils import PlotLogger
from generator.generator_for_3d_unet import MyGenerator
import os
import datetime
import numpy as np
import tensorflow as tf

# Metrics and Loss Functions
from train_3d_unet import focal_tversky_loss_08, dc, se
import glob
import json
import argparse


# Function to load data and continue training
def continue_training(train_gen, val_gen, fold, args):
    # Load the saved model
    model = load_model(args.saved_model_path, custom_objects={"focal_tversky_loss_08": focal_tversky_loss_08, "dc": dc, "se": se})
    
    # Define the optimizer and compile the model
    LR = 0.0018 # fix for future
    E = 600
    B = args.batch_size

    input_shape = train_gen.get_input_shape()
    
    model.compile(loss=focal_tversky_loss_08, optimizer=Adam(learning_rate=LR), metrics=[dc, se])
    
    # fix this for later use
    time = "20240421_130548"
    print(time)
    
    cs = []
    csvlog_path = os.path.join(args.base_dir, args.results_dir, f'fold_{fold}', "log", time + ".csv")
    csv_logger = CSVLogger(csvlog_path)
    cs = cs + [csv_logger]
    
    pltlog_path = os.path.join(args.base_dir, args.results_dir, f'fold_{fold}', "learning_curves", time + ".png")
    plot_logger = PlotLogger(csvlog_path, pltlog_path)
    cs = cs + [plot_logger]

    
    model_path = args.base_dir + args.results_dir + f'/fold_{fold}' + "/models/" + time + "_best" + ".keras"
    
    mc = ModelCheckpoint(filepath=model_path, monitor="val_loss", save_best_only=True)
    cs = cs + [mc]
    
    model_path_latest = args.base_dir + args.results_dir + f'/fold_{fold}' + "/models/" + time + "_latest" + ".keras"
    
    
    mc_latest = ModelCheckpoint(filepath=model_path_latest, monitor="val_loss", save_best_only=True)
    cs = cs + [mc_latest]

    def lr_scheduler(epoch):
        initial_lr = LR  # Initial learning rate
        exponent = 0.9

        lr = initial_lr * (1 - (epoch - args.initial_epoch)/ E) ** exponent
        return lr


    lrs = LearningRateScheduler(lr_scheduler)
    cs = cs + [lrs]
    
    steps_per_epoch = int( np.ceil(input_shape[0] / B) )

    model.fit(train_gen, epochs=E, steps_per_epoch=steps_per_epoch,
                                       validation_data=val_gen, callbacks=cs, verbose=1, initial_epoch=args.initial_epoch)
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

    continue_training(train_gen, val_gen, fold, args)

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
    parser.add_argument("-ie", "--initial_epoch", type=int, default=360)
    parser.add_argument("-mp", "--saved_model_path", type=str, default="")
    
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