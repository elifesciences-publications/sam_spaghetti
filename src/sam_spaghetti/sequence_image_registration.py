import numpy as np
from scipy import ndimage as nd
import pandas as pd

from vplants.image.serial.all import imread, imsave
from vplants.image.spatial_image import SpatialImage

from vplants.container import array_dict

from vplants.tissue_nukem_3d.growth_estimation import image_sequence_rigid_vectorfield_registration

from sam_spaghetti.sam_sequence_loading import load_sequence_signal_images

import os
import logging
from time import time as current_time

max_time = 100


def register_sequence_images(sequence_name, save_files=True, image_dirname=None, reference_name='TagBFP', microscope_orientation=-1, verbose=True, debug=False, loglevel=0):
    """
    """

    logging.getLogger().setLevel(logging.INFO if verbose else logging.DEBUG if debug else logging.ERROR)

    if save_files and (image_dirname is None):
        logging.error("No output directory provided, results will NOT be saved!")
        save_files = False

    reference_images = load_sequence_signal_images(sequence_name,image_dirname,signal_names=[reference_name])[reference_name]
    filenames = np.sort(reference_images.keys())

    transformed_images, rigid_transformations, vectorfield_transformations = image_sequence_rigid_vectorfield_registration(reference_images,microscope_orientation=microscope_orientation,verbose=verbose,debug=debug,loglevel=loglevel)

    if save_files:

        for i_file,(reference_filename,floating_filename) in enumerate(zip(filenames[:-1],filenames[1:])):
            transformed_image_file = image_dirname+"/"+sequence_name+"/"+floating_filename+"/"+floating_filename+"_to_"+filenames[0][-3:]+"_"+reference_name+".inr.gz"
            imsave(transformed_image_file,SpatialImage(transformed_images[floating_filename],voxelsize=transformed_images[floating_filename].voxelsize))
        
            reference_to_floating_transform_file = image_dirname+"/"+sequence_name+"/"+reference_filename+"/"+reference_filename+"_to_"+floating_filename[-3:]+"_rigid_transform.csv"
            np.savetxt(reference_to_floating_transform_file, rigid_transformations[(reference_filename,floating_filename)], delimiter=";")
        
            floating_to_reference_transform_file = image_dirname+"/"+sequence_name+"/"+floating_filename+"/"+floating_filename+"_to_"+reference_filename[-3:]+"_rigid_transform.csv"
            np.savetxt(floating_to_reference_transform_file, rigid_transformations[(floating_filename,reference_filename)], delimiter=";")

            start_time = current_time()
            logging.info("".join(["  " for l in xrange(loglevel)])+"  --> Saving vectorfield transformations")
            vector_field_file = image_dirname+"/"+sequence_name+"/"+floating_filename+"/"+floating_filename+"_to_"+reference_filename[-3:]+"_vector_field.inr.gz"
            imsave(vector_field_file,SpatialImage(vectorfield_transformations[(floating_filename,reference_filename)],voxelsize=transformed_images[floating_filename].voxelsize))
            
            invert_vector_field_file = image_dirname+"/"+sequence_name+"/"+reference_filename+"/"+reference_filename+"_to_"+floating_filename[-3:]+"_vector_field.inr.gz"
            imsave(invert_vector_field_file,SpatialImage(vectorfield_transformations[(reference_filename,floating_filename)],voxelsize=transformed_images[floating_filename].voxelsize))
            logging.info("".join(["  " for l in xrange(loglevel)])+"  <-- Saving vectorfield transformations ["+str(current_time()-start_time)+" s]")
 

    result = (transformed_images, rigid_transformations, vectorfield_transformations)
    return result
            