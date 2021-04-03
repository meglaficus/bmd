from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
from keras.models import load_model
from keras.utils import plot_model

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
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return (2.0 * intersection + 1) / (K.sum(y_true) + K.sum(y_pred) + 1)

def se(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return (intersection + 1) / (K.sum(y_true) + 1)

def tversky_08(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
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

def train(train_gen, val_gen, args):

    LR = 0.0005
    E = 360
    SPE = 1/240

    input_shape = train_gen.get_input_shape()

    model = build_unet(input_shape)

    model.summary()
    path = args.results_dir + "model.png"
    plot_model(model, to_file=path, show_shapes=True, show_layer_names=True)
    print("Save model.png, done.")

    model.compile(loss=focal_tversky_loss_08, optimizer=Adam(lr=LR), metrics=[dc, se])

    cs = []
    csvlog_path = args.results_dir + "log/" + args.dt + ".csv"
    csv_logger = CSVLogger(csvlog_path)
    cs = cs + [csv_logger]

    mc = ModelCheckpoint(filepath=args.model_path, monitor="val_loss", save_best_only=True)
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

    model.fit_generator(train_gen, epochs=E, steps_per_epoch=len(train_gen) * SPE,
                                       validation_data=val_gen, callbacks=cs, verbose=1)
    K.clear_session()


    model = load_model(args.model_path, custom_objects={"focal_tversky_loss_08": focal_tversky_loss_08, "dc": dc, "se": se, })
    score = model.evaluate_generator(val_gen)
    print(score)

    return



#########
## RUN ##
#########

def run(args):

    ####################################
    ## Prepare train_gen and test_gen ##
    ####################################

    B = 6

    train_pos_dir = args.root_dir + args.train_dir
    val_pos_dir = args.root_dir + args.val_dir

    train_ct_list = glob.glob(train_pos_dir + "/**/*ct.npy", recursive=True)
    train_lb_list = glob.glob(train_pos_dir + "/**/*lb.npy", recursive=True)
    train_ct_list.sort()
    train_lb_list.sort()
    print(len(train_ct_list), len(train_lb_list))

    val_ct_list = glob.glob(val_pos_dir + "/**/*ct.npy", recursive=True)
    val_lb_list = glob.glob(val_pos_dir + "/**/*lb.npy", recursive=True)
    val_ct_list.sort()
    val_lb_list.sort()
    print(len(val_ct_list), len(val_lb_list))

    train_gen = MyGenerator(train_ct_list, train_lb_list, batch_size=B, flip=1, alpha=0.2, beta=0.3, ws=[2000, 400, 600, 150])
    val_gen = MyGenerator(val_ct_list, val_lb_list, batch_size=B, flip=0, alpha=0, beta=0, ws=[2000, 400, 600, 150])


    ###########
    ## Train ##
    ###########

    train(train_gen, val_gen, args)

    print("Done")

    return



##############
## argparse ##
##############

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="/xxx/dir1")
    parser.add_argument("--train_dir", type=str, default="/train")
    parser.add_argument("--val_dir", type=str, default="/val")
    parser.add_argument("--results_dir", type=str, default="./results_3d_unet")
    args = parser.parse_args()

    d = args.results_dir
    l = [d, d + "/log", d + "/models", d + "/learning_curves", d + "/predicted_images"]
    for i in l:
        if not (os.path.exists(i)):
            os.makedirs(i)

    args.dt = datetime.datetime.now(pytz.timezone("Asia/Tokyo")).strftime('%Y%m%d_%H%M%S')
    print(args.dt)

    args.model_path = args.results_dir + "/models" + args.dt + ".h5"

    return args



##########
## Main ##
##########

if __name__ == '__main__':

    args = get_args()
    print(args)
    run(args)

