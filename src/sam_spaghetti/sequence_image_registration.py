import numpy as np
from scipy import ndimage as nd
import pandas as pd

from timagetk.components import SpatialImage
from timagetk.io import imsave, save_trsf

from tissue_nukem_3d.growth_estimation import image_sequence_rigid_vectorfield_registration, apply_sequence_point_registration

from sam_spaghetti.sam_sequence_loading import load_sequence_signal_images, load_sequence_rigid_transformations, load_sequence_segmented_images, load_sequence_signal_data

from timagetk.wrapping.bal_trsf import TRSF_TYPE_DICT, TRSF_UNIT_DICT
from timagetk.algorithms.trsf import allocate_c_bal_matrix, apply_trsf, create_trsf, compose_trsf

import os
import logging
from time import time as current_time

max_time = 100


def register_sequence_images(sequence_name, pyramid_lowest_level=1, compute_vectorfield=True, save_files=True, image_dirname=None, reference_name='TagBFP', microscope_orientation=-1, verbose=True, debug=False, loglevel=0):
    """
    """

    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)

    if save_files and (image_dirname is None):
        logging.error("No output directory provided, results will NOT be saved!")
        save_files = False

    reference_images = load_sequence_signal_images(sequence_name,image_dirname,signal_names=[reference_name])[reference_name]
    filenames = np.sort(list(reference_images.keys()))

    transformed_images, rigid_transformations, vectorfield_transformations = image_sequence_rigid_vectorfield_registration(reference_images,microscope_orientation=microscope_orientation,pyramid_lowest_level=pyramid_lowest_level,compute_vectorfield=compute_vectorfield,verbose=verbose,debug=debug,loglevel=loglevel)

    if save_files:

        for i_file,(reference_filename,floating_filename) in enumerate(zip(filenames[:-1],filenames[1:])):
            transformed_image_file = image_dirname+"/"+sequence_name+"/"+floating_filename+"/"+floating_filename+"_to_"+filenames[0][-3:]+"_"+reference_name+".inr.gz"
            imsave(transformed_image_file,SpatialImage(transformed_images[floating_filename],voxelsize=transformed_images[floating_filename].voxelsize))
        
            reference_to_floating_transform_file = image_dirname+"/"+sequence_name+"/"+reference_filename+"/"+reference_filename+"_to_"+floating_filename[-3:]+"_rigid_transform.csv"
            np.savetxt(reference_to_floating_transform_file, rigid_transformations[(reference_filename,floating_filename)], delimiter=";")
        
            floating_to_reference_transform_file = image_dirname+"/"+sequence_name+"/"+floating_filename+"/"+floating_filename+"_to_"+reference_filename[-3:]+"_rigid_transform.csv"
            np.savetxt(floating_to_reference_transform_file, rigid_transformations[(floating_filename,reference_filename)], delimiter=";")

            if compute_vectorfield:
                start_time = current_time()
                logging.info("".join(["  " for l in range(loglevel)])+"  --> Saving vectorfield transformations")
                vector_field_file = image_dirname+"/"+sequence_name+"/"+floating_filename+"/"+floating_filename+"_to_"+reference_filename[-3:]+"_vector_field.inr.gz"
                # imsave(vector_field_file,SpatialImage(vectorfield_transformations[(floating_filename,reference_filename)],voxelsize=transformed_images[floating_filename].voxelsize))
                save_trsf(vectorfield_transformations[(floating_filename,reference_filename)],vector_field_file)

                invert_vector_field_file = image_dirname+"/"+sequence_name+"/"+reference_filename+"/"+reference_filename+"_to_"+floating_filename[-3:]+"_vector_field.inr.gz"
                # imsave(invert_vector_field_file,SpatialImage(vectorfield_transformations[(reference_filename,floating_filename)],voxelsize=transformed_images[floating_filename].voxelsize))
                save_trsf(vectorfield_transformations[(reference_filename,floating_filename)], invert_vector_field_file)

                logging.info("".join(["  " for l in range(loglevel)])+"  <-- Saving vectorfield transformations ["+str(current_time()-start_time)+" s]")

    result = (transformed_images, rigid_transformations, vectorfield_transformations)
    return result


def apply_sequence_registration(sequence_name, save_files=True, image_dirname=None, reference_name='TagBFP', membrane_name='PI', microscope_orientation=-1, raw=True, verbose=True, debug=False, loglevel=0):

    if save_files and (image_dirname is None):
        logging.error("No output directory provided, results will NOT be saved!")
        save_files = False

    signal_images = load_sequence_signal_images(sequence_name,image_dirname, raw=raw, verbose=verbose, debug=debug, loglevel=loglevel+1)
    segmented_images = load_sequence_segmented_images(sequence_name, image_dirname, membrane_name=membrane_name, verbose=verbose, debug=debug, loglevel=loglevel+1)
    if len(segmented_images)>0:
        signal_images[membrane_name+"_seg"] = segmented_images
        
    filenames = np.sort(list(signal_images[reference_name].keys()))

    registered_images = {}
    for signal_name in signal_images.keys():
        registered_images[signal_name] = {}
        registered_images[signal_name][filenames[0]] = signal_images[signal_name][filenames[0]]

    reference_img = signal_images[reference_name][filenames[0]]

    sequence_rigid_transforms = load_sequence_rigid_transformations(sequence_name, image_dirname, verbose=verbose, debug=debug, loglevel=loglevel)

    sequence_rigid_trsf = {}
    sequence_rigid_trsf[filenames[0]] = create_trsf(param_str_2='-identity', trsf_type=TRSF_TYPE_DICT['RIGID_3D'], trsf_unit=TRSF_UNIT_DICT['REAL_UNIT'])

    for i_file, (reference_filename, floating_filename) in enumerate(zip(filenames[:-1], filenames[1:])):
        # rigid_transformation = sequence_rigid_transforms[(floating_filename,reference_filename)]
        rigid_transformation = sequence_rigid_transforms[(reference_filename, floating_filename)]

        rigid_trsf = create_trsf(param_str_2='-identity', trsf_type=TRSF_TYPE_DICT['RIGID_3D'], trsf_unit=TRSF_UNIT_DICT['REAL_UNIT'])
        allocate_c_bal_matrix(rigid_trsf.mat.c_struct, rigid_transformation)

        # sequence_rigid_trsf[floating_filename] = compose_trsf([sequence_rigid_trsf[reference_filename],rigid_trsf])
        sequence_rigid_trsf[floating_filename] = rigid_trsf

        for signal_name in signal_images.keys():
            signal_img = signal_images[signal_name][floating_filename]
            if not "_seg" in signal_name:
                # registered_signal_img = apply_trsf(signal_img, sequence_rigid_trsf[floating_filename], template_img=reference_img, param_str_2='-interpolation linear')
                registered_signal_img = apply_trsf(signal_img, sequence_rigid_trsf[floating_filename], template_img=reference_img, param_str_2='-interpolation nearest')
            else:
                registered_signal_img = apply_trsf(signal_img, sequence_rigid_trsf[floating_filename], template_img=reference_img, param_str_2='-interpolation nearest')
                registered_signal_img[registered_signal_img==0] = 1
            registered_images[signal_name][floating_filename] = registered_signal_img

            if save_files:
                registered_image_file = image_dirname + "/" + sequence_name + "/" + floating_filename + "/" + floating_filename + "_to_" + filenames[0][-3:] + "_" + signal_name + ".inr.gz"
                imsave(registered_image_file, SpatialImage(registered_signal_img , voxelsize=registered_signal_img .voxelsize))

    sequence_data = load_sequence_signal_data(sequence_name,image_dirname, normalized=True, aligned=False, verbose=verbose, debug=debug, loglevel=loglevel+1)
    if len(sequence_data)==0:
        sequence_data = load_sequence_signal_data(sequence_name, image_dirname, nuclei=False, aligned=False, verbose=verbose, debug=debug, loglevel=loglevel + 1)
        nuclei = False
    else:
        nuclei = True
    apply_sequence_point_registration(sequence_data,sequence_rigid_transforms, verbose=verbose, debug=debug, loglevel=loglevel+1)

    if save_files:
        for i_f, filename in enumerate(filenames):
            file_data = sequence_data[filename]
            if "Normalized_"+reference_name in file_data.columns:
                data_filename = image_dirname + "/" + sequence_name + "/" + filename + "/" + filename + "_normalized_"+("signal" if nuclei else "cell")+"_data.csv"
            else:
                data_filename = image_dirname + "/" + sequence_name + "/" + filename + "/" + filename + "_"+("signal" if nuclei else "cell")+"_data.csv"
            file_data.to_csv(data_filename)

    return registered_images
