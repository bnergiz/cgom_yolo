import yaml
import os
import json
import csv
from skimage.measure import find_contours, approximate_polygon

def read_config(config_file_name):
    """
    Read a config file.
    :param config_file_name: Name of the config file, default is config.yaml.
    :return: Config file.
    """
    with open(config_file_name, 'r') as file_pointer:
#        args = yaml.load(file_pointer, Loader=yaml.FullLoader)
        args = yaml.unsafe_load(file_pointer)
    return args

def make_path(dir):
    """
    Make a directory.
    :param dir: Directory to create.
    :return:
    """
    if not os.path.isdir(dir):
        os.makedirs(dir)

def read_gaze(path, max_res):
    """
    Read a GazaData file.
    :param path: Path to the GazaData file.
    :return: Loaded file.
    """
    file = []
    with open(path, 'r') as file_pointer:
        reader = csv.reader(file_pointer, delimiter='\t')
        header = next(reader)
        assert header[0] == 'Event Start Trial Time [ms]' \
               and header[1] == 'Event End Trial Time [ms]' \
               and header[3] == 'Visual Intake Position X [px]' \
               and header[4] == 'Visual Intake Position Y [px]',\
            "Make sure the gaze file follows the format in 'read_gaze()'"
        for row in reader:

            # Only use fixations and values inside the cam resolution
            if row[3] != '-' and row[4] != '-' \
                    and float(row[3]) < max_res[1] and float(row[4]) < max_res[0]:

                # Convert time string to seconds
                t_start = float(row[0])/1000.
                t_end = float(row[1])/1000.

                # Round to the nearest pixel
                x = int(float(row[3]))
                y = int(float(row[4]))

                # Create entry
                file.append({'start_time': t_start, 'end_time': t_end, 'x': x, 'y': y})

            else:
                pass
    return file

def write_gaze(file, path):
    """
    Create a gaze output file - of any format basically.
    :param file: File to write.
    :param path: Path to write to.
    :return:
    """
    with open(path, 'w') as file_pointer:
        writer = csv.writer(file_pointer, delimiter='\t')
        writer.writerow(list(file[0].keys()))
        for row in file:
            writer.writerow(list(row.values()))

def load(path):
    """
    Load a json file.
    :param path: Where from which to load the file.
    :return: Loaded file.
    """
    with open(path, 'r') as file_pointer:
        file = json.load(file_pointer)
    return file

def dump(file, path):
    """
    Dump a json file.
    :param file: File to dump.
    :param path: Path to dump to.
    :return:
    """
    with open(path, 'w') as file_pointer:
        json.dump(file, file_pointer)

def encode_masks(masks):
    """
    Encode the dense mask to a sparse mask by polygon approximation.
    :param mask: Dense mask.
    :return: Sparse mask of dim [region id, (number of points, x, y)]
    """
    en_masks = []
    for i in range(masks.shape[-1]):
        coords = []
        for contour in find_contours(masks[:, :, i], 0):
            coords.append(approximate_polygon(contour, tolerance=0.5))
        en_masks.append(coords)
    return en_masks

def decode_labels(ids, labels):
    """
    Decode the labels.
    :param id: ID array of dim [N].
    :param labels: Label array of dim [number of labels].
    :return: Array of dim [N].
    """
    return [labels[int(ids[i]) - 1] for i in range(ids.size)]

def encode_labels(ids, label):
    """
    Encode single label.
    :param ids: Dict of dim [number of labels] containing the map from 'name' to 'id'
    :param label: The label to encode.
    :return: ID of the label.
    """
    for id in ids:
        if id['name'] == label:
            return id['id']
        else:
            pass
    raise ValueError('Label not in class_info.')

class Dictset(object):
    """
    Create a dict in the VIA format (http://www.robots.ox.ac.uk/~vgg/software/via/).
    """
    def __init__(self):
        self.dict = {}

    def add_entry(self, name, masks, labels):
        """
        Add an entry to the dict.
        :param name: Image name.
        :param masks: Encoded masks belonging to the image.
        :param labels: Labels of each mask.
        :return:
        """
        regions = {}
        i = 0
        for mask, label in zip(masks, labels):
            for coords in mask:
                x_points = [coords[j, 1] for j in range(coords.shape[0])]
                y_points = [coords[j, 0] for j in range(coords.shape[0])]
                regions[str(i)] = {"shape_attributes": {"name":"polygon", "all_points_x": x_points, "all_points_y": y_points}, "region_attributes":{"label": label}}
                i += 1

        self.dict[name] = {"fileref": "", "size": "", "filename": "", "base64_img_data": "", "file_attributes": {}, "regions": regions}

    def size(self):
        """
        Gives the size of the dict.
        :return: Number of images in the dict.
        """
        return len(self.dict)

    def dump(self, path):
        """
        Dump the dict
        :param path: The path to dump to.
        :return:
        """
        dump(self.dict, path)
