"""
ST: 'cGOM'
Bachmann David, Hess Stephan & Julian Wolf (SV)
pdz, ETH Zürich
2018

This file contains functions to easily create mask based on COCO predictions.
"""

# Global imports
import os
import sys
import skimage
import cv2
import random
import warnings
import argparse
import numpy as np

# Local imports
import utils

# Import Mask RCNN from parent directory
sys.path.append('..')
from mrcnn.config import Config
from mrcnn import model as modellib, visualize
from mrcnn.utils import extract_bboxes

# Suppress warnings
warnings.filterwarnings('ignore', message='Anti-aliasing will be enabled by default in skimage 0.15 to')

# Import default_configs
parser = argparse.ArgumentParser(description='Make masks based on COCO prediction.')
parser.add_argument('--config', default='./default_configs/make_mask.yaml')
c = parser.parse_args()
args = utils.read_config(c.config)

# Derived config class
class PredictConfig(Config):

    # Name
    NAME = "predict"

    # Number of GPUs
    GPU_COUNT = 1

    # Number of images per GPU
    IMAGES_PER_GPU = 1

    # Detection confidence
    DETECTION_MIN_CONFIDENCE = args['detection_min_confidence']

    # Number of classes (enforce COCO)
    NUM_CLASSES = 81

# Propose masks
def create_mask(model, names):

    # Directories
    CLASS_DIR = os.path.join(args['data_set_dir'], names)
    TRAIN_DIR = os.path.join(args['output_dir'], names + '/train')
    VAL_DIR = os.path.join(args['output_dir'], names + '/val')
    MASK_DIR = os.path.join(args['output_dir'], names + '/masks')
    UNMASK_DIR = os.path.join(args['output_dir'], names + '/unmasks')
    utils.make_path(TRAIN_DIR)
    utils.make_path(VAL_DIR)
    utils.make_path(MASK_DIR)
    utils.make_path(UNMASK_DIR)

    # Dict to dump
    train_dict = utils.Dictset()
    val_dict = utils.Dictset()

    # Check if combined label
    labels = names.split('_')
    n = len(labels)

    # Hu moments
    valid_moments = [np.array([]) for _ in range(n)]

    # Colors
    colors = visualize.random_colors(n)

    # Predict masks for all files in the folder
    for image_path in os.listdir(CLASS_DIR):
        image = skimage.io.imread(os.path.join(CLASS_DIR, image_path))
        r = model.detect([image], verbose=1)[0]
        masks = r['masks']
        num_masks = masks.shape[-1]

        if num_masks > 0:
            # Get boxes
            bb = extract_bboxes(masks)
            bb_image = image.copy()
            for i in range(num_masks):
                (y1, x1, y2, x2) = bb[i]
                bb_image = visualize.draw_box(bb_image, (y1, x1, y2, x2), (255, 0, 0), bb_image.shape[0]/500.)
                cv2.putText(bb_image, 'mask' + str(i), (x1, y1), cv2.FONT_HERSHEY_PLAIN, bb_image.shape[0]/500., (255, 0, 0))

            # Get Hu moments
            moments = np.zeros((num_masks, 7))
            for i in range(num_masks):
                moments[i] = (cv2.HuMoments(cv2.moments(masks[:, :, i].astype('uint8'))).flatten())

            # Calculate the closest resemblance
            track_val = np.zeros(num_masks)
            for i in range(n):
                valid_moment = valid_moments[i]
                if valid_moment.size > 0:
                    avg_moment = np.average(valid_moment, axis=0)
                    dist = np.linalg.norm(avg_moment - moments, axis=1)

                    # Multiple predictions can be made
                    for _ in range(args['num_preds']):
                        min = np.argmin(dist)
                        track_val[min] = i + 1
                        dist[min] = np.Inf
                else:
                    pass

            # Create Window
            cv2.namedWindow('masks', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('masks', args['res'][0], args['res'][1])
            cv2.moveWindow('masks', 500, 500)

            # Create sliders
            for i in range(num_masks):
                cv2.createTrackbar('mask' + str(i), 'masks', int(track_val[i]), n, lambda x: None)
            # Create save slider
            cv2.createTrackbar('save mask','masks', 0, 1, lambda x: None)

            while True:
                # Extract valid masks
                temp_masks = np.compress(track_val, masks, axis=2)
                temp_track_val = np.compress(track_val, track_val, axis=0)

                # Create mask image
                masked_img = bb_image.copy()
                for i in range(temp_masks.shape[-1]):
                    color = colors[int(temp_track_val[i]) - 1]
                    masked_img = visualize.apply_mask(masked_img, temp_masks[:, :, i], color)

                # Image, note that cv displays as BGR
                cv2.imshow('masks', np.flip(masked_img, axis=2))

                # Get trackbar positions
                for i in range(num_masks):
                    track_val[i] = cv2.getTrackbarPos('mask' + str(i), 'masks')
                save_flag = cv2.getTrackbarPos('save mask', 'masks')

                # Break when flag is enabled
                cv2.waitKey(1)
                if save_flag == 1:
                    break

            cv2.destroyAllWindows()

            # Hu moments for later
            for i in range(n):
                slice = np.compress(track_val == i + 1, moments, axis=0)
                valid_moments[i] = np.concatenate([valid_moments[i], slice], axis=0) if valid_moments[i].size else slice

            # Encode masks and decode labels
            en_masks = utils.encode_masks(temp_masks)
            dec_labels = utils.decode_labels(temp_track_val, labels)

            # Store the image and mask if we have at least one mask
            if temp_masks.shape[-1] > 0:
                skimage.io.imsave(os.path.join(MASK_DIR, image_path), masked_img)
                # With a probability add the data to the training set
                if random.random() > args['val_prob']:
                    skimage.io.imsave(os.path.join(TRAIN_DIR, image_path), image)
                    train_dict.add_entry(image_path, en_masks, dec_labels)
                # Else to the validation set
                else:
                    skimage.io.imsave(os.path.join(VAL_DIR, image_path), image)
                    val_dict.add_entry(image_path, en_masks, dec_labels)
            else:
                # If no mask was selected it is moved to the unmask dir, from where it can be manually processed
                skimage.io.imsave(os.path.join(UNMASK_DIR, image_path), image)
        else:
            # If no mask was found it is moved to the unmask dir, from where it can be manually processed
            skimage.io.imsave(os.path.join(UNMASK_DIR, image_path), image)

    # Dump the dictionaries to the corresponding folders
    assert train_dict.size() > 0 and val_dict.size() > 0, "Train or val set are empty. Add more images."
    train_dict.dump(os.path.join(TRAIN_DIR, args['dump_name']))
    val_dict.dump(os.path.join(VAL_DIR, args['dump_name']))

if __name__ == '__main__':

    # Configs for model
    config = PredictConfig()
    config.display()

    # Load the model
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args['log_dir'])
    assert os.path.basename(args['weights_path']) == "mask_rcnn_coco.h5", 'Use COCO weights.'
    model.load_weights(args['weights_path'], by_name=True)

    # Extract the label names
    names = [name for name in os.listdir(os.path.join(args['data_set_dir'], '.'))]

    # Create masks
    for name in names:
        create_mask(model, name)
