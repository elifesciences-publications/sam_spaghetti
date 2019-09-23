import logging
from time import time as current_time
from copy import deepcopy
import os
import argparse

import numpy as np
import scipy.ndimage as nd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


from sam_spaghetti.sam_sequence_loading import load_sequence_signal_images, load_sequence_rigid_transformations, load_sequence_segmented_images, load_sequence_signal_data, load_sequence_wall_meshes

from timagetk.io import imread, imsave
from timagetk.components import SpatialImage
from timagetk.algorithms import watershed
from timagetk.algorithms import resample_isotropic

from tissue_nukem_3d.microscopy_images.read_microscopy_image import read_czi_image
from tissue_nukem_3d.epidermal_maps import compute_local_2d_signal

from cellcomplex.property_topomesh.io import save_ply_property_topomesh, read_ply_property_topomesh
from cellcomplex.property_topomesh.analysis import compute_topomesh_property, compute_topomesh_cell_property_from_faces

from cellcomplex.property_topomesh.utils.matplotlib_tools import mpl_draw_topomesh
from cellcomplex.property_topomesh.utils.pandas_tools import topomesh_to_dataframe

from tissue_paredes.wall_extraction import extract_wall_meshes
from tissue_paredes.wall_analysis import compute_wall_property, compute_wall_angle, estimate_cell_centers
from tissue_paredes.wall_signal_quantification import quantify_wall_signals, quantify_wall_membrane_signals
from tissue_paredes.polarity_analysis import compute_wall_signal_polarities, cell_signal_polarity_topomesh, cell_signal_polarity_histograms

from tissue_paredes.utils.matplotlib_tools import mpl_draw_wall_lines, mpl_draw_wall_signal_polarity_vectors, plot_cell_polarity_histograms

logging.getLogger().setLevel(logging.INFO)


def extract_sequence_walls(sequence_name, save_files=True, image_dirname=None, membrane_name='PI', wall_types = ['anticlinal_L1'], resampling_voxelsize=0.5, target_edge_length= 2., microscope_orientation=-1, verbose=True, debug=False, loglevel=0):
    """
    """

    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)

    if save_files and (image_dirname is None):
        logging.error("No output directory provided, results will NOT be saved!")
        save_files = False

    reference_images = load_sequence_signal_images(sequence_name,image_dirname,signal_names=[membrane_name])[membrane_name]
    segmented_images = load_sequence_segmented_images(sequence_name,image_dirname)
    filenames = np.sort(segmented_images.keys())

    wall_topomeshes = {}
    for i_file, filename in enumerate(filenames):
        img = reference_images[filename]
        seg_img = segmented_images[filename]

        wall_filename = image_dirname + "/" + sequence_name + "/" + filename + "/" + filename + "_walls.ply"

        all_wall_topomesh = extract_wall_meshes(seg_img,wall_types=wall_types,resampling_voxelsize=resampling_voxelsize,smoothing=True,target_edge_length=target_edge_length)
        logging.info("".join(["  " for l in range(loglevel)]) + "--> " + filename + " : " + str(all_wall_topomesh.nb_wisps(3)) + " wall meshes extracted")
        wall_topomeshes[filename] = all_wall_topomesh

        if save_files:
            save_ply_property_topomesh(all_wall_topomesh,wall_filename,properties_to_save={0:['wall_label'],1:[],2:[],3:['cell_labels']})

    result = (wall_topomeshes,)
    return result


def compute_sequence_wall_polarities(sequence_name, save_files=True, image_dirname=None, membrane_name='PI', signal_names=['PIN1'], loglevel=0):

    if not membrane_name in signal_names:
        signal_names = signal_names + [membrane_name]

    signal_images = load_sequence_signal_images(sequence_name, image_dirname, signal_names=signal_names, loglevel=loglevel+1)
    sequence_cell_data = load_sequence_signal_data(sequence_name, image_dirname, normalized=False, aligned=False, nuclei=False, loglevel=loglevel+1)
    wall_topomeshes = load_sequence_wall_meshes(sequence_name, image_dirname, loglevel=loglevel+1)

    filenames = np.sort(wall_topomeshes.keys())

    sequence_wall_data = {}

    for i_file, filename in enumerate(filenames):

        all_wall_topomesh = wall_topomeshes[filename]
        img_dict = dict([(signal_name,signal_images[signal_name][filename]) for signal_name in signal_names])

        logging.info(str(all_wall_topomesh.nb_wisps(3))+" Walls")

        wall_filename = image_dirname + "/" + sequence_name + "/" + filename + "/" + filename + "_walls.ply"

        # compute_wall_property(all_wall_topomesh,'area')
        # compute_wall_property(all_wall_topomesh,'barycenter')
        # compute_wall_property(all_wall_topomesh,'normal')
        # compute_wall_angle(all_wall_topomesh,direction=[1,0,0],signed=False)

        # quantify_wall_signals(all_wall_topomesh,img_dict,wall_sigma=0.6)
        quantify_wall_membrane_signals(all_wall_topomesh,img_dict,membrane_channel=membrane_name,channel_names=signal_names,exclude_contours=False)

        for channel_name in signal_names:
            logging.info("".join(["  " for l in range(loglevel)]) + "  --> Quantfying " + channel_name + " wall polarity " )
            compute_wall_signal_polarities(all_wall_topomesh,channel_name)

        properties_to_save={0:['wall_label'],1:[],2:[],3:['cell_labels']}
        for channel_name in signal_names:
            properties_to_save[0] += [channel_name, "left_"+channel_name, "right_"+channel_name, "normal"]
            properties_to_save[3] += [channel_name, "left_"+channel_name, "right_"+channel_name, channel_name+"_polarity", channel_name+"_polarity_vector", "barycenter", "normal"]
        save_ply_property_topomesh(all_wall_topomesh,wall_filename,properties_to_save=properties_to_save)

        all_wall_topomesh = read_ply_property_topomesh(wall_filename)

        cell_topomesh = cell_signal_polarity_topomesh(all_wall_topomesh,signal_names)

        wall_data = topomesh_to_dataframe(all_wall_topomesh,3)
        wall_data['left_label'] = all_wall_topomesh.wisp_property('cell_labels',3).values()[:,0]
        wall_data['right_label'] = all_wall_topomesh.wisp_property('cell_labels',3).values()[:,1]
        for k,dim in enumerate(['x','y','z']):
            wall_data['normal_'+dim] = all_wall_topomesh.wisp_property('normal',3).values()[:,k]
            for signal_name in signal_names:
                wall_data[signal_name+'_polarity_vector_'+dim] = all_wall_topomesh.wisp_property(signal_name+'_polarity_vector',3).values()[:,k]

        wall_data_filename = image_dirname + "/" + sequence_name + "/" + filename + "/" + filename + "_walls.csv"
        wall_data.to_csv(wall_data_filename,index=False)

        sequence_wall_data[filename] = wall_data

        cell_data = sequence_cell_data[filename]
        cell_polarity_data = topomesh_to_dataframe(cell_topomesh,0)
        for k,dim in enumerate(['x','y','z']):
            for signal_name in signal_names:
                cell_polarity_data[signal_name+'_polarity_vector_'+dim] = cell_topomesh.wisp_property(signal_name+'_polarity_vector',0).values()[:,k]

        for column in cell_polarity_data.columns:
            if not column in cell_data.columns:
                cell_data[column] = np.nan
        cell_data.set_index('label',inplace=True)
        cell_data.update(cell_polarity_data.set_index('label'))
        cell_data.reset_index(inplace=True)
        cell_data_filename = image_dirname + "/" + sequence_name + "/" + filename + "/" + filename + "_cell_data.csv"
        cell_data.to_csv(cell_data_filename,index=False)

    result = (sequence_wall_data,)
    return result
