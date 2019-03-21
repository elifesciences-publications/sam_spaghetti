import pandas as pd
import numpy as np

from time import time as current_time

from scipy.misc import imread as imread2d

from vplants.image.serial.all import imread, imsave
from vplants.image.spatial_image import SpatialImage

import logging
import os

max_time = 100

def load_sequence_signal_images(sequence_name, image_dirname, signal_names=None, raw=True, verbose=False, debug=False, loglevel=0):

    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)

    signal_images = {}

    sequence_filenames = []
    for time in xrange(max_time):
        filename = sequence_name+"_t"+str(time).zfill(2)
        if os.path.exists(image_dirname+"/"+sequence_name+"/"+filename+"/"+filename+"_signal_data.csv"):
            sequence_filenames += [filename]

    if len(sequence_filenames)>0:
        logging.info("".join(["  " for l in xrange(loglevel)])+"--> Loading sequence images "+sequence_name+" : "+str([f[-3:] for f in sequence_filenames]))

        for filename in sequence_filenames:
            data_filename = image_dirname+"/"+sequence_name+"/"+filename+"/"+filename+"_signal_data.csv"
            file_df = pd.read_csv(data_filename)

            if signal_names is None:
                file_signals = [c for c in file_df.columns if (not "center" in c) and (not "layer" in c) and (not 'curvature' in c) and (not 'Unnamed' in c) and (not 'label' in c)]
            else:
                file_signals = [c for c in signal_names if c in file_df.columns]

            for signal_name in file_signals:

                start_time = current_time()
                logging.info("".join(["  " for l in xrange(loglevel)])+"  --> Loading : "+filename+" "+signal_name)
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
                        logging.warn("".join(["  " for l in xrange(loglevel)])+"  --> Unable to find : "+filename+" "+signal_name)
                logging.info("".join(["  " for l in xrange(loglevel)])+"  <-- Loading : "+filename+" "+signal_name+" ["+str(current_time() - start_time)+" s]")

    return signal_images


def load_sequence_signal_image_slices(sequence_name, image_dirname, signal_names=None, projection_type='max_intensity', aligned=False, verbose=False, debug=False, loglevel=0):

    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)

    signal_image_slices = {}

    sequence_filenames = []
    for time in xrange(max_time):
        filename = sequence_name+"_t"+str(time).zfill(2)
        if os.path.exists(image_dirname+"/"+sequence_name+"/"+filename+"/"+filename+"_signal_data.csv"):
            sequence_filenames += [filename]

    if len(sequence_filenames)>0:
        logging.info("".join(["  " for l in xrange(loglevel)])+"--> Loading sequence 2D signal images "+sequence_name+" : "+str([f[-3:] for f in sequence_filenames]))

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
                    logging.warn("".join(["  " for l in xrange(loglevel)])+"  --> Unable to find : "+filename+" "+signal_name)
    return signal_image_slices


def load_sequence_signal_data(sequence_name, image_dirname, normalized=False, aligned=False, verbose=False, debug=False, loglevel=0):

    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)

    signal_data = {}

    sequence_filenames = []
    for time in xrange(max_time):
        filename = sequence_name+"_t"+str(time).zfill(2)
        # data_filename = "".join([image_dirname+"/"+filename+"/"+filename,"_aligned_L1" if aligned else "","_normalized" if normalized else "","_signal_data.csv"])
        # data_filename = image_dirname+"/"+sequence_name+"/"+filename+"/"+filename+"_signal_data.csv"
        data_filename = "".join([image_dirname+"/"+sequence_name+"/"+filename+"/"+filename,"_aligned_L1" if aligned else "","_normalized" if normalized else "","_signal_data.csv"])
        if os.path.exists(data_filename):
            sequence_filenames += [filename]

    if len(sequence_filenames)>0:
        logging.info("".join(["  " for l in xrange(loglevel)])+"--> Loading sequence data "+sequence_name+" : "+str([f[-3:] for f in sequence_filenames]))
        for filename in sequence_filenames:
            data_filename = "".join([image_dirname+"/"+sequence_name+"/"+filename+"/"+filename,"_aligned_L1" if aligned else "","_normalized" if normalized else "","_signal_data.csv"])
            df = pd.read_csv(data_filename)
            # if ('DIIV' in df.columns)&('TagBFP' in df.columns):
            #     df['qDII'] = deepcopy(df['DIIV'].values)
            #     df['DIIV'] = deepcopy(df['DIIV'].values*df['TagBFP'].values)
            signal_data[filename] = df

    return signal_data


def load_sequence_primordia_data(sequence_name, image_dirname, verbose=False, debug=False, loglevel=0):

    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)

    primordia_data = {}

    sequence_filenames = []
    for time in xrange(max_time):
        filename = sequence_name+"_t"+str(time).zfill(2)
        primordia_data_filename = "".join([image_dirname+"/"+sequence_name+"/"+filename+"/"+filename,"_aligned_primordia_extrema.csv"])
        if os.path.exists(primordia_data_filename):
            sequence_filenames += [filename]

    if len(sequence_filenames)>0:
        logging.info("".join(["  " for l in xrange(loglevel)])+"--> Loading sequence primordia data "+sequence_name+" : "+str([f[-3:] for f in sequence_filenames]))
        for filename in sequence_filenames:
            primordia_data_filename = "".join([image_dirname+"/"+sequence_name+"/"+filename+"/"+filename,"_aligned_primordia_extrema.csv"])
            df = pd.read_csv(primordia_data_filename)
            primordia_data[filename] = df

    return primordia_data
                

def load_sequence_rigid_transformations(sequence_name, image_dirname, verbose=False, debug=False, loglevel=0):

    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)

    rigid_transformations = {}

    sequence_filenames = []
    for time in xrange(max_time):
        filename = sequence_name+"_t"+str(time).zfill(2)
        if os.path.exists(image_dirname+"/"+sequence_name+"/"+filename+"/"+filename+"_signal_data.csv"):
            sequence_filenames += [filename]

    for i_file,(reference_filename,floating_filename) in enumerate(zip(sequence_filenames[:-1],sequence_filenames[1:])):
        logging.info("".join(["  " for l in xrange(loglevel)])+"--> Loading rigid transforms "+reference_filename+" <--> "+floating_filename)
        reference_to_floating_transform_file = image_dirname+"/"+sequence_name+"/"+reference_filename+"/"+reference_filename+"_to_"+floating_filename[-3:]+"_rigid_transform.csv"
        floating_to_reference_transform_file = image_dirname+"/"+sequence_name+"/"+floating_filename+"/"+floating_filename+"_to_"+reference_filename[-3:]+"_rigid_transform.csv"
            
        rigid_transformations[(reference_filename,floating_filename)] = np.loadtxt(reference_to_floating_transform_file,delimiter=";")
        rigid_transformations[(floating_filename,reference_filename)] = np.loadtxt(floating_to_reference_transform_file,delimiter=";")    

    return rigid_transformations


def load_sequence_vectorfield_transformations(sequence_name, image_dirname, verbose=False, debug=False, loglevel=0):

    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)

    vectorfield_transformations = {}

    sequence_filenames = []
    for time in xrange(max_time):
        filename = sequence_name+"_t"+str(time).zfill(2)
        if os.path.exists(image_dirname+"/"+sequence_name+"/"+filename+"/"+filename+"_signal_data.csv"):
            sequence_filenames += [filename]

    for i_file,(reference_filename,floating_filename) in enumerate(zip(sequence_filenames[:-1],sequence_filenames[1:])):
        start_time = current_time()
        logging.info("".join(["  " for l in xrange(loglevel)])+"--> Loading vectorfield transform "+reference_filename+" <-- "+floating_filename)
        vector_field_file = image_dirname+"/"+sequence_name+"/"+floating_filename+"/"+floating_filename+"_to_"+reference_filename[-3:]+"_vector_field.inr.gz"
        vectorfield_transformations[(floating_filename,reference_filename)] = imread(vector_field_file)
        logging.info("".join(["  " for l in xrange(loglevel)])+"<-- Loading vectorfield transform "+reference_filename+" <-- "+floating_filename+" ["+str(current_time()-start_time)+" s]")

        start_time = current_time()
        logging.info("".join(["  " for l in xrange(loglevel)])+"--> Loading vectorfield transform "+reference_filename+" --> "+floating_filename)
        invert_vector_field_file = image_dirname+"/"+sequence_name+"/"+reference_filename+"/"+reference_filename+"_to_"+floating_filename[-3:]+"_vector_field.inr.gz"
        vectorfield_transformations[(reference_filename,floating_filename)] = imread(invert_vector_field_file)
        logging.info("".join(["  " for l in xrange(loglevel)])+"<-- Loading vectorfield transform "+reference_filename+" --> "+floating_filename+" ["+str(current_time()-start_time)+" s]")

    return vectorfield_transformations

