import csv
import json

def read_gaze(path, max_res):
    """
    Read a GazaData file. COPY from toolbox/utils
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

def load(path):
    """
    Load a json file.
    :param path: Where from which to load the file.
    :return: Loaded file.
    """
    with open(path, 'r') as file_pointer:
        file = json.load(file_pointer)
    return file