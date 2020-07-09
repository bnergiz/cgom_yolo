"""
ST: 'cGOM'
Bachmann David, Hess Stephan & Julian Wolf (SV)
pdz, ETH Zürich
2018

This file contains all functions required to train the neural network.
"""

# Global imports
import os
import sys
import skimage
import warnings
import argparse
from imgaug import augmenters as aug
import numpy as np

# Local imports
import utils

# Import Mask RCNN from parent directory
sys.path.append('..')
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn.utils import Dataset

# Suppress warnings
warnings.filterwarnings('ignore', message='Anti-aliasing will be enabled by default in skimage 0.15 to')

# Import default_configs
parser = argparse.ArgumentParser(description='Train mask RCNN.')
parser.add_argument('--config', default='./default_configs/make_train.yaml')
c = parser.parse_args()
args = utils.read_config(c.config)

# Derived config class
class TrainConfig(Config):

    # Name
    NAME = "train"

    # Number of GPUs
    GPU_COUNT = 1

    # Number of images per GPU
    IMAGES_PER_GPU = 2

    # Number of training steps per epoch
    STEPS_PER_EPOCH = args['steps_per_epoch']

    # Validation Steps
    VALIDATION_STEPS = args['val_steps']

class TrainDataset(Dataset):

    def load_train(self, names, subset):
        """
        Load the data from a folder 'names' with a subfolder 'train' or 'val'.
        :param names: Name of the folder, corresponds to the contained labels in the folder, ie. cup_pen_phone.
        :param subset: Name of subset folder, ie. 'train' or 'val'
        :return:
        """
        # Directory
        CLASS_DIR = os.path.join(args['data_set_dir'], names)
        SUBSET_DIR = os.path.join(CLASS_DIR, subset)

        # Check if combined label
        labels = names.split('_')

        # Add classes
        for label in labels:
            # If we have number labels we pass them, ie. cup and cup_1 and cup_2
            # Note that we can only have 6 folders of the same label
            # If more are required expand the array or generalize
            if label in ['1', '2', '3', '4', '5']:
                pass
            else:
                class_id = len(self.class_info)
                self.add_class('train', class_id, label)

        # Load annotations
        anns = utils.load(os.path.join(SUBSET_DIR, "via_region_data.json"))

        # Add images
        for key, value in anns.items():

            # Get the polygons (masks) and labels
            regions = value['regions'].values()
            polygons = [region['shape_attributes'] for region in regions]
            labels = [region['region_attributes'] for region in regions]

            # Encode the labels
            en_labels = np.zeros(len(labels))
            for i, label in enumerate(labels):
                en_labels[i] = utils.encode_labels(self.class_info, label['label'])

            # Unfortunately we have to load the image in order to get the height and width
            file_name = key.split('.')[0] + '.JPG'
            IMAGE_DIR = os.path.join(SUBSET_DIR, file_name)
            image = skimage.io.imread(IMAGE_DIR)
            height, width = image.shape[:2]

            # Create a unique image ID
            image_id = len(self.image_info)

            # Add the image
            self.add_image(
                "train",
                image_id=image_id,
                path=IMAGE_DIR,
                width=width, height=height,
                polygons=polygons, labels=en_labels)

    def load_mask(self, image_id):
        """
        Loads the masks of an image with image ID.
        :param image_id: ID of image.
        :return: The masks as bool and the labels of each masks.
        """
        # Get info
        info = self.image_info[image_id]

        # Convert polygons to a bitmap mask of shape
        mask = np.zeros((info['height'], info['width'], len(info['polygons'])), dtype=np.uint8)
        for i, polygon in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(polygon['all_points_y'], polygon['all_points_x'])
            mask[rr, cc, i] = 1

        # Get the labels
        labels = info['labels']

        return mask.astype(np.bool), labels

    def image_reference(self, image_id):
        """
        Maps the image ID to the info entry.
        :param image_id: ID of image.
        :return: The path of the image.
        """
        info = self.image_info[image_id]
        return info['path']

    def dump_labels(self, path):
        """
        Dumps the labels as a json file.
        :param path: Path to dump to.
        :return:
        """
        labels_dict = dict(zip(self.class_names, self.class_ids.tolist()))
        utils.dump(labels_dict, path)


if __name__ == '__main__':

    # Extract the label names
    names = [name for name in os.listdir(os.path.join(args['data_set_dir'], '.'))]

    # Create training data set
    dataset_train = TrainDataset()
    dataset_val = TrainDataset()

    # Fill the datasets
    for name in names:
        dataset_train.load_train(name, 'train')
        dataset_val.load_train(name, 'val')

    # Prepare the datasets
    dataset_train.prepare()
    dataset_val.prepare()

    # Dump the actual class ID and labels
    LABEL_PATH = os.path.join(args['label_dir'], 'labels.json')
    utils.make_path(args['label_dir'])
    dataset_train.dump_labels(LABEL_PATH)

    # Configs for model
    class TrainConfig(TrainConfig):
        NUM_CLASSES = len(dataset_train.class_info)
    config = TrainConfig()
    config.display()

    # Load the model
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=args['log_dir'])

    # Load weights
    if args['weights_path'] == 'last':
        model.load_weights(model.find_last(), by_name=True)
    else:
        if os.path.basename(args['weights_path']) == "mask_rcnn_coco.h5":
            # Exclude the last layers because they require a matching number of classes
            model.load_weights(args['weights_path'], by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"])
        else:
            model.load_weights(args['weights_dir'], by_name=True)

    # Augmentation
    augmentation = aug.SomeOf((0, 2), [
        aug.Fliplr(0.5),
        aug.Flipud(0.5),
        aug.OneOf([aug.Affine(rotate=90), aug.Affine(rotate=180), aug.Affine(rotate=270)]),
        aug.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
    ])

    # Train the network
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=args['num_epochs'][0],
                layers='heads', augmentation=augmentation)
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=args['num_epochs'][1],
                layers='4+', augmentation=augmentation)
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10.,
                epochs=args['num_epochs'][2],
                layers='all', augmentation=augmentation)
