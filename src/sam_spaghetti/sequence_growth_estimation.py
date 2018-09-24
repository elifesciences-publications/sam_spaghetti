import numpy as np
from scipy import ndimage as nd
import pandas as pd

from sam_spaghetti.sam_sequence_loading import load_sequence_signal_images, load_sequence_signal_data, load_sequence_rigid_transformations, load_sequence_vectorfield_transformations
from sam_spaghetti.utils.signal_luts import quantified_signals

from vplants.tissue_nukem_3d.growth_estimation import surfacic_growth_estimation

from copy import deepcopy
from time import time as current_time

import os
import logging


def compute_surfacic_growth(sequence_name, image_dirname, save_files=True, maximal_length=15., microscope_orientation=-1, verbose=False, debug=False, loglevel=0):
    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)

    sequence_data = load_sequence_signal_data(sequence_name, image_dirname, normalized=True, aligned=False, verbose=verbose, debug=debug, loglevel=loglevel)
    sequence_rigid_transforms = load_sequence_rigid_transformations(sequence_name, image_dirname, verbose=verbose, debug=debug, loglevel=loglevel)
    sequence_vectorfield_transforms = load_sequence_vectorfield_transformations(sequence_name, image_dirname, verbose=verbose, debug=debug, loglevel=loglevel)

    sequence_data = surfacic_growth_estimation(sequence_data,sequence_rigid_transforms,sequence_vectorfield_transforms,maximal_length=maximal_length,microscope_orientation=microscope_orientation,quantified_signals=quantified_signals, verbose=verbose, debug=debug, loglevel=loglevel)

    if save_files:
        for filename in sequence_data:
            normalized=True
            aligned=False
            data_filename = "".join([image_dirname+"/"+sequence_name+"/"+filename+"/"+filename,"_aligned_L1" if aligned else "","_normalized" if normalized else "","_signal_data.csv"])
            sequence_data[filename].to_csv(data_filename,index=False)

    return sequence_data
        
            