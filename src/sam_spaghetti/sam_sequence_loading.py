import logging
import os
import re

import pandas as pd
import numpy as np

from time import time as current_time

from imageio import imread as imread2d
from timagetk.io import imread
from timagetk.io import read_trsf

from cellcomplex.property_topomesh.io import read_ply_property_topomesh

from sam_spaghetti.utils.signal_luts import vector_signals, tensor_signals

max_time = 100


def load_sequence_filenames(sequence_name, image_dirname, verbose=False, debug=False, loglevel=0):
    sequence_filenames = []
    for time in range(max_time):
        filename = sequence_name+"_t"+str(time).zfill(2)
        if os.path.exists(image_dirname+"/"+sequence_name+"/"+filename):
            sequence_filenames += [filename]
    return sequence_filenames


def load_sequence_signal_images(sequence_name, image_dirname, signal_names=None, raw=True, registered=False, verbose=False, debug=False, loglevel=0):

    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)

    signal_images = {}

    sequence_filenames = load_sequence_filenames(sequence_name, image_dirname=image_dirname)

    if len(sequence_filenames)>0:
        logging.info("".join(["  " for l in range(loglevel)])+"--> Loading sequence images "+sequence_name+" : "+str([f[-3:] for f in sequence_filenames]))

        for i_f,filename in enumerate(sequence_filenames):

            if signal_names is None:
                data_filename = image_dirname+"/"+sequence_name+"/"+filename+"/"+filename+"_signal_data.csv"
                if os.path.exists(data_filename):
                    file_df = pd.read_csv(data_filename)
                else:
                    data_filename = image_dirname+"/"+sequence_name+"/"+filename+"/"+filename+"_cell_data.csv"
                    file_df = pd.read_csv(data_filename)
                file_signals = [c for c in file_df.columns if (not "center" in c) and (not "layer" in c) and (not 'curvature' in c) and (not 'Unnamed' in c) and (not 'label' in c)]
            else:
                file_signals = [c for c in signal_names if os.path.exists(image_dirname+"/"+sequence_name+"/"+filename+"/"+filename+"_"+c+".inr.gz")]

            for signal_name in file_signals:

                start_time = current_time()
                logging.info("".join(["  " for l in range(loglevel)])+"  --> Loading : "+filename+" "+signal_name)

                if registered and i_f>0:
                    signal_image_file = image_dirname+"/"+sequence_name+"/"+filename+"/"+filename+"_to_"+sequence_filenames[0][-3:]+"_"+signal_name+".inr.gz"
                else:
                    raw_signal_image_file = image_dirname+"/"+sequence_name+"/"+filename+"/"+filename+"_"+signal_name+"_raw.inr.gz"
                    if not raw or not os.path.exists(raw_signal_image_file):
                        signal_image_file = image_dirname+"/"+sequence_name+"/"+filename+"/"+filename+"_"+signal_name+".inr.gz"
                    else:
                        signal_image_file = raw_signal_image_file
                if os.path.exists(signal_image_file):
                    img = imread(signal_image_file)
                    if not signal_name in signal_images:
                        signal_images[signal_name] = {}
                    signal_images[signal_name][filename] = img
                else:
                    signal_image_file = image_dirname+"/"+sequence_name+"/"+filename+"/"+filename+"_"+signal_name+".inr"
                    if os.path.exists(signal_image_file):
                        signal_images[signal_name][filename] = imread(signal_image_file)
                    else:
                        logging.warn("".join(["  " for l in range(loglevel)])+"  --> Unable to find : "+filename+" "+signal_name)
                logging.info("".join(["  " for l in range(loglevel)])+"  <-- Loading : "+filename+" "+signal_name+" ["+str(current_time() - start_time)+" s]")

    return signal_images


def load_sequence_segmented_images(sequence_name, image_dirname, membrane_name='PI', registered=False, verbose=False, debug=False, loglevel=0):

    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)

    segmented_images = {}

    sequence_filenames = load_sequence_filenames(sequence_name, image_dirname=image_dirname)

    if len(sequence_filenames)>0:
        logging.info("".join(["  " for l in range(loglevel)])+"--> Loading sequence segmented images "+sequence_name+" : "+str([f[-3:] for f in sequence_filenames]))

        for i_f, filename in enumerate(sequence_filenames):
            start_time = current_time()
            logging.info("".join(["  " for l in range(loglevel)])+"  --> Loading : "+filename+" "+membrane_name+" segmented")
            if registered and i_f>0:
                segmented_image_file = image_dirname+"/"+sequence_name+"/"+filename+"/"+filename+"_to_"+sequence_filenames[0][-3:]+"_"+membrane_name+"_seg.inr.gz"
            else:
                segmented_image_file = image_dirname+"/"+sequence_name+"/"+filename+"/"+filename+"_"+membrane_name+"_seg.inr.gz"

            if os.path.exists(segmented_image_file):
                img = imread(segmented_image_file)
                segmented_images[filename] = img
            else:
                logging.warn("".join(["  " for l in range(loglevel)])+"  --> Unable to find : "+filename+" "+membrane_name+" segmented")
            logging.info("".join(["  " for l in range(loglevel)])+"  <-- Loading : "+filename+" "+membrane_name+" segmented  ["+str(current_time() - start_time)+" s]")

    return segmented_images


def load_sequence_signal_image_slices(sequence_name, image_dirname, signal_names=None, projection_type='max_intensity', aligned=False, verbose=False, debug=False, loglevel=0):

    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)

    signal_image_slices = {}

    sequence_filenames = []
    for time in range(max_time):
        filename = sequence_name+"_t"+str(time).zfill(2)
        if os.path.exists(image_dirname+"/"+sequence_name+"/"+filename+"/"+filename+"_signal_data.csv"):
            sequence_filenames += [filename]

    if len(sequence_filenames)>0:
        logging.info("".join(["  " for l in range(loglevel)])+"--> Loading sequence 2D signal images "+sequence_name+" : "+str([f[-3:] for f in sequence_filenames]))

        for filename in sequence_filenames:
            data_filename = image_dirname+"/"+sequence_name+"/"+filename+"/"+filename+"_signal_data.csv"
            file_df = pd.read_csv(data_filename)

            if signal_names is None:
                file_signals = [c for c in file_df.columns if (not "center" in c) and (not "layer" in c) and (not 'curvature' in c) and (not 'Unnamed' in c) and (not 'label' in c)]
            else:
                file_signals = [c for c in signal_names if c in file_df.columns]

            for signal_name in file_signals:
                signal_image_file = image_dirname+"/"+sequence_name+"/"+filename+"/"+filename+("_aligned" if aligned else "_")+projection_type+"_"+signal_name+"_projection.tif"
                # signal_image_file = image_dirname+"/"+sequence_name+"/"+filename+"/"+filename+("_aligned_" if aligned else "_")+projection_type+"_"+signal_name+"_slice.tif"
                # signal_image_file = image_dirname+"/"+sequence_name+"/"+filename+"/"+filename+("_aligned_" if aligned else "_")+projection_type+"_"+signal_name+"_projection.tif"
                if os.path.exists(signal_image_file):
                    if not signal_name in signal_image_slices:
                        signal_image_slices[signal_name] = {}
                    img = imread2d(signal_image_file)
                    if img.dtype == np.uint8:
                        img = (img.astype(np.uint16))*256
                    signal_image_slices[signal_name][filename] = img
                else:
                    logging.warn("".join(["  " for l in range(loglevel)])+"  --> Unable to find : "+filename+" "+signal_name)
    return signal_image_slices


def array_from_printed_string(string):
    if not pd.isnull(string):
        s = re.sub("[ ]+", " ", string)
        s = re.sub("\[ ", "[", s)
        s = re.sub("\n", "", s)
        s = re.sub(" ", ",", s)
        return np.array(eval(s))
    else:
        return string


def load_sequence_signal_data(sequence_name, image_dirname, nuclei=True, normalized=False, aligned=False, verbose=False, debug=False, loglevel=0):

    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)

    signal_data = {}

    sequence_filenames = []
    for time in range(max_time):
        filename = sequence_name+"_t"+str(time).zfill(2)
        data_filename = "".join([image_dirname+"/"+sequence_name+"/"+filename+"/"+filename,"_aligned_L1" if aligned else "","_normalized" if normalized else "","_","signal" if nuclei else "cell","_data.csv"])
        if os.path.exists(data_filename):
            sequence_filenames += [filename]

    if len(sequence_filenames)>0:
        logging.info("".join(["  " for l in range(loglevel)])+"--> Loading sequence data "+sequence_name+" : "+str([f[-3:] for f in sequence_filenames]))
        for filename in sequence_filenames:
            data_filename = "".join([image_dirname+"/"+sequence_name+"/"+filename+"/"+filename,"_aligned_L1" if aligned else "","_normalized" if normalized else "","_","signal" if nuclei else "cell","_data.csv"])
            df = pd.read_csv(data_filename)

            for column in vector_signals+tensor_signals:
                if column in df.columns:
                    df[column] = map(array_from_printed_string, df[column].values)

            signal_data[filename] = df

    return signal_data


def load_sequence_primordia_data(sequence_name, image_dirname, verbose=False, debug=False, loglevel=0):

    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)

    primordia_data = {}

    sequence_filenames = []
    for time in range(max_time):
        filename = sequence_name+"_t"+str(time).zfill(2)
        primordia_data_filename = "".join([image_dirname+"/"+sequence_name+"/"+filename+"/"+filename,"_aligned_primordia_extrema.csv"])
        if os.path.exists(primordia_data_filename):
            sequence_filenames += [filename]

    if len(sequence_filenames)>0:
        logging.info("".join(["  " for l in range(loglevel)])+"--> Loading sequence primordia data "+sequence_name+" : "+str([f[-3:] for f in sequence_filenames]))
        for filename in sequence_filenames:
            primordia_data_filename = "".join([image_dirname+"/"+sequence_name+"/"+filename+"/"+filename,"_aligned_primordia_extrema.csv"])
            df = pd.read_csv(primordia_data_filename)
            primordia_data[filename] = df

    return primordia_data
                

def load_sequence_rigid_transformations(sequence_name, image_dirname, verbose=False, debug=False, loglevel=0):

    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)

    rigid_transformations = {}

    sequence_filenames = []
    for time in range(max_time):
        filename = sequence_name+"_t"+str(time).zfill(2)
        if os.path.exists(image_dirname+"/"+sequence_name+"/"+filename+"/"+filename+"_signal_data.csv"):
            sequence_filenames += [filename]
        elif os.path.exists(image_dirname+"/"+sequence_name+"/"+filename+"/"+filename+"_cell_data.csv"):
            sequence_filenames += [filename]

    for i_file,(reference_filename,floating_filename) in enumerate(zip(sequence_filenames[:-1],sequence_filenames[1:])):
        logging.info("".join(["  " for l in range(loglevel)])+"--> Loading rigid transforms "+reference_filename+" <--> "+floating_filename)
        reference_to_floating_transform_file = image_dirname+"/"+sequence_name+"/"+reference_filename+"/"+reference_filename+"_to_"+floating_filename[-3:]+"_rigid_transform.csv"
        floating_to_reference_transform_file = image_dirname+"/"+sequence_name+"/"+floating_filename+"/"+floating_filename+"_to_"+reference_filename[-3:]+"_rigid_transform.csv"
            
        rigid_transformations[(reference_filename,floating_filename)] = np.loadtxt(reference_to_floating_transform_file,delimiter=";")
        rigid_transformations[(floating_filename,reference_filename)] = np.loadtxt(floating_to_reference_transform_file,delimiter=";")    

    return rigid_transformations


def load_sequence_vectorfield_transformations(sequence_name, image_dirname, verbose=False, debug=False, loglevel=0):

    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)

    vectorfield_transformations = {}

    sequence_filenames = []
    for time in range(max_time):
        filename = sequence_name+"_t"+str(time).zfill(2)
        if os.path.exists(image_dirname+"/"+sequence_name+"/"+filename+"/"+filename+"_signal_data.csv"):
            sequence_filenames += [filename]
        elif os.path.exists(image_dirname+"/"+sequence_name+"/"+filename+"/"+filename+"_cell_data.csv"):
            sequence_filenames += [filename]

    for i_file,(reference_filename,floating_filename) in enumerate(zip(sequence_filenames[:-1],sequence_filenames[1:])):
        start_time = current_time()
        logging.info("".join(["  " for l in range(loglevel)])+"--> Loading vectorfield transform "+reference_filename+" <-- "+floating_filename)
        vector_field_file = image_dirname+"/"+sequence_name+"/"+floating_filename+"/"+floating_filename+"_to_"+reference_filename[-3:]+"_vector_field.inr.gz"
        vectorfield_transformations[(floating_filename,reference_filename)] = read_trsf(vector_field_file)
        logging.info("".join(["  " for l in range(loglevel)])+"<-- Loading vectorfield transform "+reference_filename+" <-- "+floating_filename+" ["+str(current_time()-start_time)+" s]")

        start_time = current_time()
        logging.info("".join(["  " for l in range(loglevel)])+"--> Loading vectorfield transform "+reference_filename+" --> "+floating_filename)
        invert_vector_field_file = image_dirname+"/"+sequence_name+"/"+reference_filename+"/"+reference_filename+"_to_"+floating_filename[-3:]+"_vector_field.inr.gz"
        vectorfield_transformations[(reference_filename,floating_filename)] = read_trsf(invert_vector_field_file)
        logging.info("".join(["  " for l in range(loglevel)])+"<-- Loading vectorfield transform "+reference_filename+" --> "+floating_filename+" ["+str(current_time()-start_time)+" s]")

    return vectorfield_transformations


def load_sequence_wall_meshes(sequence_name, image_dirname, loglevel=0):
    
    wall_topomeshes = {}
    
    sequence_filenames = []
    for time in range(max_time):
        filename = sequence_name+"_t"+str(time).zfill(2)
        if os.path.exists(image_dirname+"/"+sequence_name+"/"+filename+"/"+filename+"_walls.ply"):
            sequence_filenames += [filename]

    for i_file, filename in enumerate(sequence_filenames):
        start_time = current_time()
        logging.info("".join(["  " for l in range(loglevel)]) + "--> Loading wall mesh " + filename)
        wall_filename = image_dirname+"/"+sequence_name+"/"+filename+"/"+filename+"_walls.ply"
        wall_topomeshes[filename] = read_ply_property_topomesh(wall_filename)
        logging.info("".join(["  " for l in range(loglevel)]) + "<-- Loading wall mesh " + filename + " [" + str(current_time() - start_time) + " s]")

    return wall_topomeshes

def load_sequence_signal_wall_data(sequence_name, image_dirname, loglevel=0):

    wall_data = {}

    sequence_filenames = []
    for time in range(max_time):
        filename = sequence_name+"_t"+str(time).zfill(2)
        wall_data_filename = "".join([image_dirname+"/"+sequence_name+"/"+filename+"/"+filename,"_walls.csv"])
        if os.path.exists(wall_data_filename):
            sequence_filenames += [filename]

    if len(sequence_filenames)>0:
        logging.info("".join(["  " for l in range(loglevel)])+"--> Loading sequence wall data "+sequence_name+" : "+str([f[-3:] for f in sequence_filenames]))
        for filename in sequence_filenames:
            wall_data_filename = "".join([image_dirname+"/"+sequence_name+"/"+filename+"/"+filename,"_walls.csv"])
            df = pd.read_csv(wall_data_filename)
            wall_data[filename] = df

    return wall_data
