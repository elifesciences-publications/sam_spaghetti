import numpy as np
from scipy import ndimage as nd
import pandas as pd

from sam_spaghetti.sam_sequence_loading import load_sequence_signal_images, load_sequence_signal_data, load_sequence_rigid_transformations, load_sequence_vectorfield_transformations
from sam_spaghetti.utils.signal_luts import quantified_signals

from cellcomplex.property_topomesh.io import save_ply_property_topomesh
from tissue_nukem_3d.growth_estimation import surfacic_growth_estimation, volumetric_growth_estimation

from copy import deepcopy
from time import time as current_time

import os
import logging


def compute_growth(sequence_name, image_dirname, save_files=True, maximal_length=18., microscope_orientation=-1, growth_type='surfacic', verbose=False, debug=False, loglevel=0):
    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)

    sequence_data = load_sequence_signal_data(sequence_name, image_dirname, normalized=True, aligned=False, verbose=verbose, debug=debug, loglevel=loglevel)
    if len(sequence_data) == 0:
        sequence_data = load_sequence_signal_data(sequence_name, image_dirname, nuclei=False, aligned=False, verbose=verbose, debug=debug, loglevel=loglevel)
    sequence_rigid_transforms = load_sequence_rigid_transformations(sequence_name, image_dirname, verbose=verbose, debug=debug, loglevel=loglevel)
    sequence_vectorfield_transforms = load_sequence_vectorfield_transformations(sequence_name, image_dirname, verbose=verbose, debug=debug, loglevel=loglevel)

    if growth_type == 'surfacic':
        sequence_data = surfacic_growth_estimation(sequence_data,sequence_rigid_transforms,sequence_vectorfield_transforms,maximal_length=maximal_length,microscope_orientation=microscope_orientation,quantified_signals=quantified_signals, verbose=verbose, debug=debug, loglevel=loglevel)
    elif growth_type == 'volumetric':
        sequence_data, sequence_triangulations = volumetric_growth_estimation(sequence_data,sequence_rigid_transforms,sequence_vectorfield_transforms,maximal_length=maximal_length,microscope_orientation=microscope_orientation,quantified_signals=quantified_signals, verbose=verbose, debug=debug, loglevel=loglevel)

    if save_files:
        for filename in sequence_data:
            normalized=True
            aligned=False
            data_filename = "".join([image_dirname+"/"+sequence_name+"/"+filename+"/"+filename,"_aligned_L1" if aligned else "","_normalized" if normalized else "","_signal_data.csv"])
            sequence_data[filename].to_csv(data_filename,index=False)
            if growth_type == 'volumetric':
                triangulation_filename = "".join([image_dirname + "/" + sequence_name + "/" + filename + "/" + filename, "_registered_tetrahedrization.ply"])
                properties_to_save = {d:[] for d in range(4)}
                properties_to_save[0] += ['layer']
                for direction in ['previous','next']:
                    for prop in ['strain_tensor','stretch_tensor','volumetric_growth','volumetric_growth_anisotropy','main_growth_direction']:
                        properties_to_save[0] += [direction + "_"+ prop]
                        properties_to_save[3] += [direction + "_"+ prop]
                save_ply_property_topomesh(sequence_triangulations[filename],triangulation_filename,properties_to_save=properties_to_save)

    return sequence_data
