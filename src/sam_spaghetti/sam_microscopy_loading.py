import numpy as np

from time import time as current_time

from timagetk.io import imread, imsave
from timagetk.components import SpatialImage
from timagetk.plugins.resampling import isometric_resampling

from tissue_nukem_3d.microscopy_images.read_microscopy_image import read_czi_image, read_lsm_image, read_tiff_image

import logging
import os


def load_image_from_microscopy(microscopy_file, no_organ_file=None, nomenclature_name=None, image_dirname=None, channel_names=None, save_images=True, reference_name='TagBFP', resampling_voxelsize=None, verbose=True, debug=False, loglevel=0):
    """
    """

    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)

    filename = os.path.split(microscopy_file)[1]

    if image_dirname is None:
        image_dirname = os.path.split(microscopy_file)[0] + "/../nuclei_images/"

    if channel_names is None:
        if "qDII-CLV3-DR5-PIN1-PI" in filename:
            channel_names = ['DIIV', 'DR5', 'PIN1', 'PI', 'TagBFP', 'CLV3']
        elif "qDII-CLV3-DR5-PI" in filename:
            channel_names = ['DIIV', 'PI', 'DR5', 'TagBFP', 'CLV3']
        elif "qDII-CLV3-DR5" in filename:
            channel_names = ['DIIV', 'DR5', 'TagBFP', 'CLV3']
        elif "qDII-CLV-pAHP6" in filename:
            channel_names = ['DIIV', 'AHP6', 'TagBFP', 'CLV3']
        elif "qDII-CLV3-PIN1-PI" in filename:
            channel_names = ['DIIV', 'PIN1', 'PI', 'TagBFP', 'CLV3']
        else:
            channel_names = ['DIIV', 'TagBFP', 'CLV3']

    start_time = current_time()
    logging.info("".join(["  " for l in range(loglevel)]) + "--> Loading microscopy image")
    if os.path.splitext(microscopy_file)[1] == ".czi":
        img_dict = read_czi_image(microscopy_file, channel_names=channel_names)
    elif os.path.splitext(microscopy_file)[1] == ".lsm":
        img_dict = read_lsm_image(microscopy_file, channel_names=channel_names)
    elif os.path.splitext(microscopy_file)[1] == ".tif":
        img_dict = read_tiff_image(microscopy_file, channel_names=channel_names)
    else:
        img_dict = None
        logging.error("".join(["  " for l in range(loglevel)]) + "--> Could not read image format! " + str(os.path.splitext(microscopy_file)[1]) + " not supported!")
    if not isinstance(img_dict, dict): # Single channel images might be returned as SpatialImage
        img_dict = {channel_names[0]: img_dict}
    logging.info("".join(["  " for l in range(loglevel)]) + "<-- Loading microscopy image [" + str(current_time() - start_time) + " s]")

    if resampling_voxelsize is not None:
        for channel in img_dict.keys():
            img_dict[channel] = isometric_resampling(img_dict[channel],method=resampling_voxelsize)

    if nomenclature_name is None:
        nomenclature_name = filename
    sequence_name = nomenclature_name[:-4]

    reference_img = list(img_dict.values())[0]

    if no_organ_file is None:
        no_organ_file = os.path.split(microscopy_file)[0] + "/../TIF-No-organs/" + filename[:-4] + "-No-organs.tif"

    if os.path.exists(no_organ_file):
        logging.info("".join(["  " for l in range(loglevel)]) + "  --> Loading cropped image")
        no_organ_dict = read_tiff_image(no_organ_file, channel_names=channel_names)
        voxelsize = reference_img.voxelsize
        for channel in channel_names:
            no_organ_dict[channel] = SpatialImage(no_organ_dict[channel], voxelsize=voxelsize)
            if resampling_voxelsize is not None:
                no_organ_dict[channel] = isometric_resampling(no_organ_dict[channel],method=resampling_voxelsize)
    else:
        no_organ_dict = {}

    n_channels = len(img_dict)

    if save_images:
        logging.info("".join(["  " for l in range(loglevel)]) + "--> Saving image channels")

    for i_channel, channel_name in enumerate(channel_names):
        if save_images:
            raw_img_file = image_dirname + "/" + sequence_name + "/" + nomenclature_name + "/" + nomenclature_name + "_" + channel_name + "_raw.inr.gz"
            img_file = image_dirname + "/" + sequence_name + "/" + nomenclature_name + "/" + nomenclature_name + "_" + channel_name + ".inr.gz"
            no_organ_img_file = image_dirname + "/" + sequence_name + "/" + nomenclature_name + "/" + nomenclature_name + "_" + channel_name + "_no_organ.inr.gz"

        if (channel_name != 'DIIV') or (not 'PIN1' in channel_names):
            # if True:
            if channel_name in no_organ_dict:
                no_organ_img = no_organ_dict[channel_name]

                if save_images:
                    start_time = current_time()
                    logging.info("".join(["  " for l in range(loglevel)]) + "  --> Saving raw " + channel_name + " image")
                    imsave(raw_img_file, img_dict[channel_name])
                    logging.info("".join(["  " for l in range(loglevel)]) + "  <-- Saving raw " + channel_name + " image [" + str(current_time() - start_time) + " s]")
                    start_time = current_time()
                    logging.info("".join(["  " for l in range(loglevel)]) + "  --> Saving cropped " + channel_name + " image")
                    imsave(img_file, no_organ_img)
                    logging.info("".join(["  " for l in range(loglevel)]) + "  <-- Saving cropped " + channel_name + " image [" + str(current_time() - start_time) + " s]")

                img_dict[channel_name] = no_organ_img
            else:
                if save_images:
                    start_time = current_time()
                    logging.info("".join(["  " for l in range(loglevel)]) + "  --> Saving " + channel_name + " image")
                    imsave(img_file, img_dict[channel_name])
                    logging.info("".join(["  " for l in range(loglevel)]) + "  <-- Saving " + channel_name + " image [" + str(current_time() - start_time) + " s]")
        elif channel_name == 'DIIV':
            voxelsize = img_dict[channel_name].voxelsize

            if channel_name in no_organ_dict:
                no_organ_img = no_organ_dict[channel_name]
                substracted_img = SpatialImage(np.maximum(0, (no_organ_dict['DIIV'].astype(np.int32) - no_organ_dict['PIN1'].astype(np.int32))).astype(np.uint16), voxelsize=voxelsize)

                if save_images:
                    logging.info("".join(["  " for l in range(loglevel)]) + "  --> Saving raw " + channel_name + " image")
                    imsave(raw_img_file, img_dict[channel_name])
                    logging.info("".join(["  " for l in range(loglevel)]) + "  --> Saving cropped " + channel_name + " image")
                    imsave(no_organ_img_file, no_organ_img)
                    logging.info("".join(["  " for l in range(loglevel)]) + "  --> Saving cropped substracted " + channel_name + " image")
                    imsave(img_file, substracted_img)
            else:
                substracted_img = SpatialImage(np.maximum(0, (img_dict['DIIV'].get_array().astype(np.int32) - img_dict['PIN1'].get_array().astype(np.int32))).astype(np.uint16),voxelsize=voxelsize)

                if save_images:
                    logging.info("".join(["  " for l in range(loglevel)]) + "  --> Saving raw " + channel_name + " image")
                    imsave(raw_img_file, img_dict[channel_name])
                    logging.info("".join(["  " for l in range(loglevel)]) + "  --> Saving substracted " + channel_name + " image")
                    imsave(img_file, substracted_img)

            img_dict[channel_name] = substracted_img
        else:
            if save_images:
                imsave(img_file, img_dict[channel_name])

    return img_dict