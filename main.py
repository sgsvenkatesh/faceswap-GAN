import platform
import imageio
import glob
import os
import cv2
import time

import numpy as np
from keras.layers import *
import keras.backend as K
import tensorflow as tf
from tensorflow.python.client import device_lib
from preprocess import preprocess_video
from detector.face_detector import MTCNNFaceDetector
from pathlib import PurePath, Path
# from IPython.display import clear_output
from moviepy.editor import VideoFileClip

device_lib.list_local_devices()

fn_source_video = "videos/source.mp4"
fn_target_video = "videos/target.mp4"

print(platform.python_version())

global TOTAL_ITERS
TOTAL_ITERS = 25000

# imageio.plugins.ffmpeg.download()

fd = MTCNNFaceDetector(sess=K.get_session(), model_path="./mtcnn_weights/")

os.system("mkdir -p faceA/rgb faceA/binary_mask faceB/rgb faceB/binary_mask")

save_interval = 5 # perform face detection every {save_interval} frames
save_path = "./faceA/"
preprocess_video("videos/source.mp4", fd, save_interval, save_path)
save_path = "./faceB/"
preprocess_video("videos/target.mp4", fd, save_interval, save_path)

print(str(len(glob.glob("faceA/rgb/*.*"))) + " face(s) extracted from source video: " + fn_source_video + ".")
print(str(len(glob.glob("faceB/rgb/*.*"))) + " face(s) extracted from target video: " + fn_target_video + ".")

#####################
## Configuration
#####################

K.set_learning_phase(1)
# Number of CPU cores
num_cpus = os.cpu_count()

# Input/Output resolution
RESOLUTION = 64 # 64x64, 128x128, 256x256
assert (RESOLUTION % 64) == 0, "RESOLUTION should be 64, 128, or 256."

# Batch size
batchSize = 4

# Use motion blurs (data augmentation)
# set True if training data contains images extracted from videos
use_da_motion_blur = False

# Use eye-aware training
# require images generated from prep_binary_masks.ipynb
use_bm_eyes = True

# Probability of random color matching (data augmentation)
prob_random_color_match = 0.5

da_config = {
    "prob_random_color_match": prob_random_color_match,
    "use_da_motion_blur": use_da_motion_blur,
    "use_bm_eyes": use_bm_eyes
}

# Path to training images
img_dirA = './faceA/rgb'
img_dirB = './faceB/rgb'
img_dirA_bm_eyes = "./faceA/binary_mask"
img_dirB_bm_eyes = "./faceB/binary_mask"

# Path to saved model weights
models_dir = "./models"

# Architecture configuration
arch_config = {}
arch_config['IMAGE_SHAPE'] = (RESOLUTION, RESOLUTION, 3)
arch_config['use_self_attn'] = True
arch_config['norm'] = "hybrid" # instancenorm, batchnorm, layernorm, groupnorm, none
arch_config['model_capacity'] = "lite" # standard, lite

# Loss function weights configuration
loss_weights = {}
loss_weights['w_D'] = 0.1 # Discriminator
loss_weights['w_recon'] = 1. # L1 reconstruction loss
loss_weights['w_edge'] = 0.1 # edge loss
loss_weights['w_eyes'] = 30. # reconstruction and edge loss on eyes area
loss_weights['w_pl'] = (0.01, 0.1, 0.3, 0.1) # perceptual loss (0.003, 0.03, 0.3, 0.3)

# Init. loss config.
loss_config = {}
loss_config["gan_training"] = "mixup_LSGAN"
loss_config['use_PL'] = False
loss_config["PL_before_activ"] = True
loss_config['use_mask_hinge_loss'] = False
loss_config['m_mask'] = 0.
loss_config['lr_factor'] = 1.
loss_config['use_cyclic_loss'] = False

#####################
## Build the model
#####################

from networks.faceswap_gan_model import FaceswapGANModel
from data_loader.data_loader import DataLoader
from utils import showG, showG_mask, showG_eyes

model = FaceswapGANModel(**arch_config)

os.system("wget https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_notop_resnet50.h5")

#from keras_vggface.vggface import VGGFace

# VGGFace ResNet50
#vggface = VGGFace(include_top=False, model='resnet50', input_shape=(224, 224, 3))'

from colab_demo.vggface_models import RESNET50
vggface = RESNET50(include_top=False, weights=None, input_shape=(224, 224, 3))
vggface.load_weights("rcmalli_vggface_tf_notop_resnet50.h5")

#from keras.applications.resnet50 import ResNet50
#vggface = ResNet50(include_top=False, input_shape=(224, 224, 3))

#vggface.summary()

model.build_pl_model(vggface_model=vggface, before_activ=loss_config["PL_before_activ"])
model.build_train_functions(loss_weights=loss_weights, **loss_config)

#####################
## Training
#####################

# Create ./models directory
Path(f"models").mkdir(parents=True, exist_ok=True)

# Get filenames
train_A = glob.glob(img_dirA+"/*.*")
train_B = glob.glob(img_dirB+"/*.*")

train_AnB = train_A + train_B

assert len(train_A), "No image found in " + str(img_dirA)
assert len(train_B), "No image found in " + str(img_dirB)
print ("Number of images in folder A: " + str(len(train_A)))
print ("Number of images in folder B: " + str(len(train_B)))

def show_loss_config(loss_config):
    for config, value in loss_config.items():
        print(f"{config} = {value}")

def reset_session(save_path):
    global model, vggface
    global train_batchA, train_batchB
    model.save_weights(path=save_path)
    del model
    del vggface
    del train_batchA
    del train_batchB
    K.clear_session()
    model = FaceswapGANModel(**arch_config)
    model.load_weights(path=save_path)
    #vggface = VGGFace(include_top=False, model='resnet50', input_shape=(224, 224, 3))
    vggface = RESNET50(include_top=False, weights=None, input_shape=(224, 224, 3))
    vggface.load_weights("rcmalli_vggface_tf_notop_resnet50.h5")
    model.build_pl_model(vggface_model=vggface, before_activ=loss_config["PL_before_activ"])
    train_batchA = DataLoader(train_A, train_AnB, batchSize, img_dirA_bm_eyes,
                              RESOLUTION, num_cpus, K.get_session(), **da_config)
    train_batchB = DataLoader(train_B, train_AnB, batchSize, img_dirB_bm_eyes,
                              RESOLUTION, num_cpus, K.get_session(), **da_config)

## training starts here

# Start training
t0 = time.time()

# This try/except is meant to resume training if we disconnected from Colab
try:
    gen_iterations
    print(f"Resume training from iter {gen_iterations}.")
except:
    gen_iterations = 0

errGA_sum = errGB_sum = errDA_sum = errDB_sum = 0
errGAs = {}
errGBs = {}
# Dictionaries are ordered in Python 3.6
for k in ['ttl', 'adv', 'recon', 'edge', 'pl']:
    errGAs[k] = 0
    errGBs[k] = 0

display_iters = 2
# global TOTAL_ITERS

global train_batchA, train_batchB
train_batchA = DataLoader(train_A, train_AnB, batchSize, img_dirA_bm_eyes,
                          RESOLUTION, num_cpus, K.get_session(), **da_config)
train_batchB = DataLoader(train_B, train_AnB, batchSize, img_dirB_bm_eyes,
                          RESOLUTION, num_cpus, K.get_session(), **da_config)

Loss_DA = []
Loss_DB = []
Loss_GA = []
Loss_GB = []
Loss_GA_Adv = []
Loss_GB_Adv = []
Loss_GA_Recon = []
Loss_GB_Recon = []
Loss_GA_Pl = []
Loss_GB_Pl = []

while gen_iterations <= TOTAL_ITERS:

    # Loss function automation
    if gen_iterations == (TOTAL_ITERS // 5 - display_iters // 2):
        # clear_output()
        loss_config['use_PL'] = True
        loss_config['use_mask_hinge_loss'] = False
        loss_config['m_mask'] = 0.0
        reset_session(models_dir)
        print("Building new loss funcitons...")
        show_loss_config(loss_config)
        model.build_train_functions(loss_weights=loss_weights, **loss_config)
        print("Done.")
    elif gen_iterations == (TOTAL_ITERS // 5 + TOTAL_ITERS // 10 - display_iters // 2):
        # clear_output()
        loss_config['use_PL'] = True
        loss_config['use_mask_hinge_loss'] = True
        loss_config['m_mask'] = 0.5
        reset_session(models_dir)
        print("Building new loss funcitons...")
        show_loss_config(loss_config)
        model.build_train_functions(loss_weights=loss_weights, **loss_config)
        print("Complete.")
    elif gen_iterations == (2 * TOTAL_ITERS // 5 - display_iters // 2):
        # clear_output()
        loss_config['use_PL'] = True
        loss_config['use_mask_hinge_loss'] = True
        loss_config['m_mask'] = 0.2
        reset_session(models_dir)
        print("Building new loss funcitons...")
        show_loss_config(loss_config)
        model.build_train_functions(loss_weights=loss_weights, **loss_config)
        print("Done.")
    elif gen_iterations == (TOTAL_ITERS // 2 - display_iters // 2):
        # clear_output()
        loss_config['use_PL'] = True
        loss_config['use_mask_hinge_loss'] = True
        loss_config['m_mask'] = 0.4
        loss_config['lr_factor'] = 0.3
        reset_session(models_dir)
        print("Building new loss funcitons...")
        show_loss_config(loss_config)
        model.build_train_functions(loss_weights=loss_weights, **loss_config)
        print("Done.")
    elif gen_iterations == (2 * TOTAL_ITERS // 3 - display_iters // 2):
        # clear_output()
        model.decoder_A.load_weights("models/decoder_B.h5")  # swap decoders
        model.decoder_B.load_weights("models/decoder_A.h5")  # swap decoders
        loss_config['use_PL'] = True
        loss_config['use_mask_hinge_loss'] = True
        loss_config['m_mask'] = 0.5
        loss_config['lr_factor'] = 1
        reset_session(models_dir)
        print("Building new loss funcitons...")
        show_loss_config(loss_config)
        model.build_train_functions(loss_weights=loss_weights, **loss_config)
        print("Done.")
    elif gen_iterations == (8 * TOTAL_ITERS // 10 - display_iters // 2):
        # clear_output()
        loss_config['use_PL'] = True
        loss_config['use_mask_hinge_loss'] = True
        loss_config['m_mask'] = 0.1
        loss_config['lr_factor'] = 0.3
        reset_session(models_dir)
        print("Building new loss funcitons...")
        show_loss_config(loss_config)
        model.build_train_functions(loss_weights=loss_weights, **loss_config)
        print("Done.")
    elif gen_iterations == (9 * TOTAL_ITERS // 10 - display_iters // 2):
        # clear_output()
        loss_config['use_PL'] = True
        loss_config['use_mask_hinge_loss'] = False
        loss_config['m_mask'] = 0.0
        loss_config['lr_factor'] = 0.1
        reset_session(models_dir)
        print("Building new loss funcitons...")
        show_loss_config(loss_config)
        model.build_train_functions(loss_weights=loss_weights, **loss_config)
        print("Done.")

    if gen_iterations == 5:
        print("working.")

    # Train dicriminators for one batch
    data_A = train_batchA.get_next_batch()
    data_B = train_batchB.get_next_batch()
    errDA, errDB = model.train_one_batch_D(data_A=data_A, data_B=data_B)
    errDA_sum += errDA[0]
    errDB_sum += errDB[0]

    # Train generators for one batch
    data_A = train_batchA.get_next_batch()
    data_B = train_batchB.get_next_batch()
    errGA, errGB = model.train_one_batch_G(data_A=data_A, data_B=data_B)
    errGA_sum += errGA[0]
    errGB_sum += errGB[0]
    for i, k in enumerate(['ttl', 'adv', 'recon', 'edge', 'pl']):
        errGAs[k] += errGA[i]
        errGBs[k] += errGB[i]
    gen_iterations += 1

    # Visualization
    if gen_iterations % display_iters == 0:
        # clear_output()

        # Display loss information
        show_loss_config(loss_config)
        print("----------")
        print('[iter %d] Loss_DA: %f Loss_DB: %f Loss_GA: %f Loss_GB: %f time: %f'
              % (gen_iterations, errDA_sum / display_iters, errDB_sum / display_iters,
                 errGA_sum / display_iters, errGB_sum / display_iters, time.time() - t0))
        print("----------")
        print("Generator loss details:")
        print(f'[Adversarial loss]')
        print(f'GA: {errGAs["adv"] / display_iters:.4f} GB: {errGBs["adv"] / display_iters:.4f}')
        print(f'[Reconstruction loss]')
        print(f'GA: {errGAs["recon"] / display_iters:.4f} GB: {errGBs["recon"] / display_iters:.4f}')
        print(f'[Edge loss]')
        print(f'GA: {errGAs["edge"] / display_iters:.4f} GB: {errGBs["edge"] / display_iters:.4f}')
        if loss_config['use_PL'] == True:
            print(f'[Perceptual loss]')
            try:
                print(f'GA: {errGAs["pl"][0] / display_iters:.4f} GB: {errGBs["pl"][0] / display_iters:.4f}')
            except:
                print(f'GA: {errGAs["pl"] / display_iters:.4f} GB: {errGBs["pl"] / display_iters:.4f}')

        Loss_DA.append(errDA_sum / display_iters)
        Loss_DB.append(errDB_sum / display_iters)
        Loss_GA.append(errGA_sum / display_iters)
        Loss_GB.append(errGB_sum / display_iters)
        Loss_GA_Adv.append(errGAs["adv"] / display_iters)
        Loss_GB_Adv.append(errGBs["adv"] / display_iters)
        Loss_GA_Recon.append(errGAs["recon"] / display_iters)
        Loss_GB_Recon.append(errGBs["recon"] / display_iters)
        Loss_GA_Pl.append(errGAs["edge"] / display_iters)
        Loss_GB_Pl.append(errGBs["edge"] / display_iters)

        # Display images
        print("----------")
        wA, tA, _ = train_batchA.get_next_batch()
        wB, tB, _ = train_batchB.get_next_batch()
        print("Transformed (masked) results:")
        showG(tA, tB, model.path_A, model.path_B, batchSize)
        print("Masks:")
        showG_mask(tA, tB, model.path_mask_A, model.path_mask_B, batchSize)
        print("Reconstruction results:")
        showG(wA, wB, model.path_bgr_A, model.path_bgr_B, batchSize)
        errGA_sum = errGB_sum = errDA_sum = errDB_sum = 0
        for k in ['ttl', 'adv', 'recon', 'edge', 'pl']:
            errGAs[k] = 0
            errGBs[k] = 0

        # Save models
        model.save_weights(path=models_dir)

## losses calc start here

Loss = {}
keys = ["Loss_DA", "Loss_DB", "Loss_GA", "Loss_GB", "Loss_GA_Adv", "Loss_GB_Adv", "Loss_GA_Recon", "Loss_GB_Recon",
        "Loss_GA_Pl", "Loss_GB_Pl"]

Loss["Loss_DA"] = Loss_DA
Loss["Loss_DB"] = Loss_DB
Loss["Loss_GA"] = Loss_GA
Loss["Loss_GB"] = Loss_GB
Loss["Loss_GA_Adv"] = Loss_GA_Adv
Loss["Loss_GB_Adv"] = Loss_GB_Adv
Loss["Loss_GA_Recon"] = Loss_GA_Recon
Loss["Loss_GB_Recon"] = Loss_GB_Recon
Loss["Loss_GA_Pl"] = Loss_GA_Pl
Loss["Loss_GB_Pl"] = Loss_GB_Pl

import json
from datetime import datetime

now = datetime.now()
timestamp = datetime.timestamp(now)

filename = "Loss_" + str(timestamp) + ".txt"
print(filename)

with open(filename, 'w') as file:
    for key in keys:
        file.write(key + "\n")
        print(key)
        file.write(json.dumps(Loss[key]))
        file.write("\n")

x = list(range(len(Loss_DA)))

print(x)

for ind in range(len(x)):
    x[ind] = (x[ind] + 1) * display_iters

# plt.plot(x, Loss_DA, label="Loss_DA")
# plt.plot(x, Loss_DB, label="Loss_DB")
# plt.plot(x, Loss_GA, label="Loss_GA")
# plt.plot(x, Loss_GB, label="Loss_GB")
# plt.plot(x, Loss_GA_Adv, label="Loss_GA_Adv")
# plt.plot(x, Loss_GB_Adv, label="Loss_GB_Adv")
# plt.plot(x, Loss_GA_Recon, label="Loss_GA_Recon")
# plt.plot(x, Loss_GB_Recon, label="Loss_GB_Recon")
# plt.plot(x, Loss_GA_Pl, label="Loss_GA_Pl")
# plt.plot(x, Loss_GB_Pl, label="Loss_GB_Pl")
# plt.gca().legend()

# plt.show()
import pickle

output = open('data.pkl', 'wb')
pickle.dump(Loss, output)
#files.download('data.pkl')
output.close()

# video conversion starts here

from converter.video_converter import VideoConverter

global model, vggface
global train_batchA, train_batchB
del model
del vggface
del train_batchA
del train_batchB
tf.reset_default_graph()
K.clear_session()
model = FaceswapGANModel(**arch_config)
model.load_weights(path=models_dir)

fd = MTCNNFaceDetector(sess=K.get_session(), model_path="./mtcnn_weights/")
vc = VideoConverter()
vc.set_face_detector(fd)
vc.set_gan_model(model)

options = {
    # ===== Fixed =====
    "use_smoothed_bbox": True,
    "use_kalman_filter": True,
    "use_auto_downscaling": False,
    "bbox_moving_avg_coef": 0.65,
    "min_face_area": 35 * 35,
    "IMAGE_SHAPE": model.IMAGE_SHAPE,
    # ===== Tunable =====
    "kf_noise_coef": 1e-3,
    "use_color_correction": "hist_match",
    "detec_threshold": 0.8,
    "roi_coverage": 0.9,
    "enhance": 0.,
    "output_type": 3,
    "direction": "AtoB", # ==================== This line determines the transform direction ====================
}

if options["direction"] == "AtoB":
    input_fn = fn_source_video
    output_fn = "OUTPUT_VIDEO_AtoB.mp4"
elif options["direction"] == "BtoA":
    input_fn = fn_target_video
    output_fn = "OUTPUT_VIDEO_BtoA.mp4"

duration = None # None or a non-negative float tuple: (start_sec, end_sec). Duration of input video to be converted

vc.convert(input_fn=input_fn, output_fn=output_fn, options=options, duration=duration)

