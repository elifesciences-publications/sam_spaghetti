import numpy as np 
import pandas as pd
import scipy.ndimage as nd

from imageio import imsave as imsave2d

from timagetk.components import SpatialImage
from timagetk.io import imsave

from timagetk.algorithms.trsf import allocate_c_bal_matrix, apply_trsf, create_trsf
from timagetk.algorithms.reconstruction import pts2transfo

from timagetk.wrapping.bal_trsf import TRSF_TYPE_DICT
from timagetk.wrapping.bal_trsf import TRSF_UNIT_DICT

from tissue_nukem_3d.epidermal_maps import compute_local_2d_signal, nuclei_density_function

from sam_spaghetti.sam_sequence_loading import load_sequence_signal_images, load_sequence_signal_data, load_sequence_primordia_data, load_sequence_segmented_images
from sam_spaghetti.sam_sequence_primordia_alignment import golden_angle

from time import time as current_time
from copy import deepcopy

import logging

def sequence_aligned_signal_images(sequence_name, image_dirname, save_files=False, signal_names=None, filenames=None,microscope_orientation=-1, verbose=False, debug=False, loglevel=0):

    signal_images = load_sequence_signal_images(sequence_name, image_dirname, verbose=verbose, debug=debug, loglevel=loglevel + 1)
    signal_data = load_sequence_signal_data(sequence_name, image_dirname, normalized=True, aligned=True, verbose=verbose, debug=debug, loglevel=loglevel + 1)

    if signal_names is None:
        signal_names = list(signal_images.keys())

    logging.info("".join(["  " for l in range(loglevel)]) + "--> Computing aligned signal images " + str(signal_names))

    if filenames is None:
        filenames = np.sort(list(signal_images[signal_names[0]].keys()))

    if len(filenames) > 0:
        file_times = np.array([int(f[-2:]) for f in filenames])

        reflections = {}
        alignment_transformations = {}

        aligned_images = {}
        for signal_name in signal_names:
            aligned_images[signal_name] = {}

        for i_time, (time, filename) in enumerate(zip(file_times, filenames)):
            file_data = signal_data[filename]
            file_data = file_data[file_data['layer'] == 1]

            X = file_data['aligned_x'].values
            Y = file_data['aligned_y'].values
            Z = file_data['aligned_z'].values

            img_X = file_data['center_x'].values
            img_Y = file_data['center_y'].values
            img_Z = file_data['center_z'].values

            img_points = np.transpose([img_X, img_Y, img_Z])
            aligned_points = np.transpose([X, Y, Z])

            reference_img = signal_images[signal_names[0]][filename]
            img_center = (np.array(reference_img.shape) * np.array(reference_img.voxelsize)) / 2.
            img_center[2] = reference_img.shape[2] * reference_img.voxelsize[2] / 8.

            alignment_transformation = pts2transfo(img_center + microscope_orientation * aligned_points, microscope_orientation * img_points)

            rotation_angle = ((180. * np.arctan2(alignment_transformation[1, 0], alignment_transformation[0, 0]) / np.pi) + 180) % 360 - 180
            reflection = np.sign(alignment_transformation[0, 0] * alignment_transformation[1, 1]) == -1

            if reflection:
                img_points = np.transpose([img_X, microscope_orientation * reference_img.shape[1] * reference_img.voxelsize[1] - img_Y, img_Z])
                alignment_transformation = pts2transfo(img_center + microscope_orientation * aligned_points, microscope_orientation * img_points)

            reflections[filename] = reflection
            alignment_transformations[filename] = alignment_transformation

            alignment_trsf = create_trsf(param_str_2='-identity', trsf_type=TRSF_TYPE_DICT['RIGID_3D'], trsf_unit=TRSF_UNIT_DICT['REAL_UNIT'])
            allocate_c_bal_matrix(alignment_trsf.mat.c_struct, alignment_transformations[filename])

            for i_signal, signal_name in enumerate(signal_names):
                start_time = current_time()
                logging.info("".join(["  " for l in range(loglevel)]) + "  --> Aligning : " + filename + " " + signal_name)

                if filename in signal_images[signal_name].keys():
                    if reflections[filename]:
                        reflected_image = deepcopy(signal_images[signal_name][filename])
                        reflected_image[:, :] = signal_images[signal_name][filename][:, ::-1, :]
                        aligned_images[signal_name][filename] = apply_trsf(SpatialImage(reflected_image.astype(reference_img.dtype), voxelsize=reference_img.voxelsize), alignment_trsf, param_str_2='-interpolation nearest')
                    else:
                        aligned_images[signal_name][filename] = apply_trsf(SpatialImage(deepcopy(signal_images[signal_name][filename]).astype(reference_img.dtype), voxelsize=reference_img.voxelsize), alignment_trsf, param_str_2='-interpolation nearest')
                    if 'timagetk' in aligned_images[signal_name][filename].metadata.keys():
                        del aligned_images[signal_name][filename].metadata['timagetk']
                logging.info("".join(["  " for l in range(loglevel)]) + "  <-- Aligning : " + filename + " " + signal_name + " [" + str(current_time() - start_time) + " s]")

            if save_files:
                logging.info("".join(["  " for l in range(loglevel)]) + "--> Saving aligned signal images : " + filename + " " + str(signal_names))
                for i_signal, signal_name in enumerate(signal_names):
                    image_filename = image_dirname + "/" + sequence_name + "/" + filename + "/" + filename + "_aligned_" + signal_name + ".inr.gz"
                    imsave(image_filename, aligned_images[signal_name][filename])

        return aligned_images


def sequence_signal_image_slices(sequence_name, image_dirname, save_files=False, signal_names=None, filenames=None, registered=False, aligned=False, filtering=False, projection_type="L1_slice", reference_name='TagBFP', membrane_name='PI', resolution=None, r_max=120., microscope_orientation=-1, verbose=False, debug=False, loglevel=0):

    signal_images = load_sequence_signal_images(sequence_name, image_dirname, registered=registered, verbose=verbose, debug=debug, loglevel=loglevel+1)
    signal_data = load_sequence_signal_data(sequence_name, image_dirname, normalized=aligned or registered, aligned=aligned, verbose=verbose, debug=debug, loglevel=loglevel+1)
    if len(signal_data)==0:
        signal_data = load_sequence_signal_data(sequence_name, image_dirname, nuclei=False, aligned=aligned, verbose=verbose, debug=debug, loglevel=loglevel + 1)

    segmented_images = load_sequence_segmented_images(sequence_name, image_dirname, membrane_name=membrane_name, registered=registered, verbose=verbose, debug=debug, loglevel=loglevel+1)
    if len(segmented_images)>0:
        signal_images[membrane_name+"_seg"] = segmented_images

    if signal_names is None:
        signal_names = list(signal_images.keys())

    logging.info("".join(["  " for l in range(loglevel)])+"--> Computing 2D signal images "+str(signal_names))

    assert reference_name in signal_names

    if filenames is None:
        filenames = np.sort(list(signal_images[reference_name].keys()))
    

    if len(filenames)>0:
        file_times = np.array([int(f[-2:]) for f in filenames])

        filtered_signal_images = {}
        for signal_name in signal_names:
            filtered_signal_images[signal_name] = {}

        for filename in filenames:
            for signal_name in signal_names:
                signal_img = signal_images[signal_name][filename].get_array().astype(float)
                filtered_img = signal_img
                if filtering:
                    start_time = current_time()
                    logging.info("".join(["  " for l in range(loglevel)])+"  --> Filtering : "+filename+" "+signal_name)
                    filtered_img = gaussian_filter(filtered_img,sigma=nuclei_sigma/np.array(signal_img.voxelsize),order=0)
                    logging.info("".join(["  " for l in range(loglevel)])+"  <-- Filtering : "+filename+" "+signal_name+" ["+str(current_time() - start_time)+" s]")
                filtered_signal_images[signal_name][filename] = filtered_img.astype(signal_images[signal_name][filename].dtype)    

        if aligned:
            aligned_images = sequence_aligned_signal_images(sequence_name, image_dirname, save_files=save_files, signal_names=signal_names,microscope_orientation=microscope_orientation, verbose=verbose, debug=debug, loglevel=loglevel+1)

            image_centers = {}
            for i_time, (time, filename) in enumerate(zip(file_times, filenames)):
                reference_img = signal_images[signal_names[0]][filename]
                img_center = (np.array(reference_img.shape) * np.array(reference_img.voxelsize)) / 2.
                img_center[2] = reference_img.shape[2] * reference_img.voxelsize[2] / 8.
                image_centers[filename] = img_center

        slice_coords = {}
        image_slices = {}
        for signal_name in signal_names:
            image_slices[signal_name] = {}

        for i_time, (time, filename) in enumerate(zip(file_times,filenames)):
            file_data = signal_data[filename]
            file_data = file_data[file_data['layer']==1]

            if aligned:
                X = file_data['aligned_x'].values
                Y = file_data['aligned_y'].values
                Z = file_data['aligned_z'].values
            elif registered:
                X = file_data['registered_x'].values
                Y = file_data['registered_y'].values
                Z = file_data['registered_z'].values
            else:
                X = file_data['center_x'].values
                Y = file_data['center_y'].values
                Z = file_data['center_z'].values

            reference_img = signal_images[reference_name][filename]
            size = np.array(reference_img.shape)
            voxelsize = microscope_orientation*np.array(reference_img.voxelsize)

            if resolution is None:
                resolution = np.abs(voxelsize)[0]

            if aligned:
                n_points = int(np.round((2*r_max)/resolution+1))
                xx,yy = np.meshgrid(np.linspace(-r_max,r_max,n_points),np.linspace(-r_max,r_max,n_points))
            else:
                n_points = int(np.round(((size-1)*np.abs(voxelsize))[0]/resolution+1))
                xx,yy = np.meshgrid(np.linspace(0,((size-1)*voxelsize)[0],n_points),np.linspace(0,((size-1)*voxelsize)[1],n_points))

            # print(signal_images[signal_names[0]][filename].shape, xx.shape)
            # extent = xx.max(),xx.min(),yy.min(),yy.max()
            extent = xx.min(),xx.max(),yy.max(),yy.min()

            if projection_type == "L1_slice":
                start_time = current_time()
                logging.info("".join(["  " for l in range(loglevel)])+"  --> Computing Z-map : "+filename)

                zz = compute_local_2d_signal(np.transpose([X,Y]),np.transpose([xx,yy],(1,2,0)),Z)
                if aligned:
                    img_center = image_centers[filename]
                    coords = (img_center+microscope_orientation*np.transpose([xx,yy,zz],(1,2,0)))/np.array(reference_img.voxelsize)
                else:
                    coords = (microscope_orientation*np.transpose([xx,yy,zz],(1,2,0)))/np.array(reference_img.voxelsize)

                extra_mask = np.any(coords > (np.array(reference_img.shape) - 1),axis=-1)
                # extra_mask = np.any(coords > (np.array(reference_img.shape) - 1), axis=1).reshape(xx.shape)
                coords = np.maximum(np.minimum(coords, np.array(reference_img.shape) - 1), 0)
                coords[np.isnan(coords)]=0
                coords = coords.astype(int)
                coords = tuple(np.transpose(np.concatenate(coords)))

                slice_coords[filename] = coords

                logging.info("".join(["  " for l in range(loglevel)])+"  <-- Computing Z-map : "+filename+"   ["+str(current_time() - start_time)+" s]")

                for i_signal, signal_name in enumerate(signal_names):
                    start_time = current_time()
                    logging.info("".join(["  " for l in range(loglevel)])+"  --> Slicing : "+filename+" "+signal_name)

                    if aligned:
                        image_slices[signal_name][filename] = aligned_images[signal_name][filename][slice_coords[filename]].reshape(xx.shape).T[:,::-1]
                    else:
                        image_slices[signal_name][filename] = filtered_signal_images[signal_name][filename][slice_coords[filename]].reshape(xx.shape).T[:,::-1]
                    image_slices[signal_name][filename][extra_mask] = 0

                    if "_seg" in signal_name:
                        image_slices[signal_name][filename][image_slices[signal_name][filename]==0] = 1

                    logging.info("".join(["  " for l in range(loglevel)])+"  <-- Slicing : "+filename+" "+signal_name+" ["+str(current_time() - start_time)+" s]")

            elif projection_type == "max_intensity":
                if aligned:
                    img_center = image_centers[filename]
                    coords = (img_center + microscope_orientation*np.transpose([xx,yy,np.zeros_like(xx)],(1,2,0)))/np.array(reference_img.voxelsize)
                else:
                    coords = (microscope_orientation*np.transpose([xx,yy,np.zeros_like(xx)],(1,2,0)))/np.array(reference_img.voxelsize)

                extra_mask = np.any(coords > (np.array(reference_img.shape) - 1),axis=-1)
                coords = np.maximum(np.minimum(coords, np.array(reference_img.shape) - 1), 0)
                coords[np.isnan(coords)]=0
                coords = coords.astype(int)
                coords = tuple(np.transpose(np.concatenate(coords)))

                for i_signal, signal_name in enumerate(signal_names):
                    if not '_seg' in signal_name:
                        start_time = current_time()
                        logging.info("".join(["  " for l in range(loglevel)])+"  --> Projecting : "+filename+" "+signal_name)
                        # depth = (np.arange(size[2])/float(size[2]-1))[np.newaxis,np.newaxis]*np.ones_like(xx)[:,:,np.newaxis]
                        # depth = np.ones_like(depth)

                        if aligned:
                            # max_projection = (depth * (aligned_images[signal_name][filename].get_array()[coords[:2]].reshape(xx.shape + (reference_img.shape[2],)))).max(axis=2)
                            max_projection = (aligned_images[signal_name][filename].get_array()[coords[:2]].reshape(xx.shape + (reference_img.shape[2],))).max(axis=2)
                            # max_projection = np.transpose(max_projection)[::-1,::-1]
                        else:
                            # max_projection = (depth * (filtered_signal_images[signal_name][filename][coords[:2]].reshape(xx.shape + (reference_img.shape[2],)))).max(axis=2)
                            max_projection = (filtered_signal_images[signal_name][filename][coords[:2]].reshape(xx.shape + (reference_img.shape[2],))).max(axis=2)
                        max_projection[extra_mask] = 0

                        image_slices[signal_name][filename] = max_projection.T[:,::-1]
                        logging.info("".join(["  " for l in range(loglevel)])+"  <-- Projecting : "+filename+" "+signal_name+" ["+str(current_time() - start_time)+" s]")
                    else:
                        start_time = current_time()
                        logging.info("".join(["  " for l in range(loglevel)])+"  --> Projecting : "+filename+" segmented " + membrane_name)
                        projection = labelled_image_projection(filtered_signal_images[signal_name][filename],direction=microscope_orientation)
                        image_slices[signal_name][filename] = projection.T[:,::-1]
                        image_slices[signal_name][filename][image_slices[signal_name][filename]==0] = 1
                        logging.info("".join(["  " for l in range(loglevel)]) + "  <-- Projecting : " + filename + " segmented " + membrane_name + " [" + str(current_time() - start_time) + " s]")

            if save_files and projection_type in ['L1_slice']:
                logging.info("".join(["  " for l in range(loglevel)])+"--> Saving 2D signal images : "+filename+" "+str(signal_names))
                for i_signal, signal_name in enumerate(signal_names):
                    image_filename = image_dirname+"/"+sequence_name+"/"+filename+"/"+filename+("_aligned_" if aligned else "_")+projection_type+"_"+signal_name+"_projection.tif"
                    imsave2d(image_filename,image_slices[signal_name][filename])
        return image_slices


def labelled_image_projection(seg_img, axis=2, direction=1, background_label=1):
    if "get_array" in dir(seg_img):
        seg_img =seg_img.get_array()

    xxx, yyy, zzz = np.mgrid[0:seg_img.shape[0], 0:seg_img.shape[1], 0:seg_img.shape[2]].astype(float)

    if axis == 0:
        y = np.arange(seg_img.shape[1])
        z = np.arange(seg_img.shape[2])
        yy,zz = map(np.transpose,np.meshgrid(y,z))
        proj = xxx * (seg_img != background_label)
    elif axis == 1:
        x = np.arange(seg_img.shape[0])
        z = np.arange(seg_img.shape[2])
        xx,zz = map(np.transpose,np.meshgrid(x,z))
        proj = yyy * (seg_img != background_label)
    elif axis == 2:
        x = np.arange(seg_img.shape[0])
        y = np.arange(seg_img.shape[1])
        xx,yy = map(np.transpose,np.meshgrid(x,y))
        proj = zzz * (seg_img != background_label)

    proj[proj == 0] = np.nan
    if direction == 1:
        proj = np.nanmax(proj, axis=axis)
        proj[np.isnan(proj)] = seg_img.shape[axis] - 1
    elif direction == -1:
        proj = np.nanmin(proj, axis=axis)
        proj[np.isnan(proj)] = 0

    if axis == 0:
        xx = proj
    elif axis == 1:
        yy = proj
    elif axis == 2:
        zz = proj

    # coords = tuple(np.transpose(np.concatenate(np.transpose([xx, yy, zz], (1, 2, 0)).astype(int))))
    coords = tuple(np.transpose(np.concatenate(np.transpose([xx, yy, zz], (1, 2, 0)).astype(int))))
    projected_img = np.transpose(seg_img[coords].reshape(xx.shape))

    return projected_img


def image_angular_slice(img, theta=0., resolution=None, extent=None, width=0.):
    img_center = (np.array(img.shape) * np.array(img.voxelsize)) / 2.

    if resolution is None:
        resolution = img.voxelsize[0]

    if extent is None:
        image_x = np.arange(img.shape[0])*img.voxelsize[0] - img_center[0]
        extent = (np.min(image_x),np.max(image_x))

    radial_distances = np.linspace(extent[0],extent[1],1+(extent[1]-extent[0])/resolution)

    if width>0:
        orthoradial_distances = np.linspace(-width,width,2*width/resolution)
    else:
        orthoradial_distances = np.array([0.])

    slice_images = []
    for d in orthoradial_distances:
        radial_x = -d*np.sin(np.radians(theta)) + radial_distances*np.cos(np.radians(theta))
        radial_y = d*np.cos(np.radians(theta)) + radial_distances*np.sin(np.radians(theta))

        image_z = np.arange(img.shape[2]) * img.voxelsize[2] - img_center[2]
        xx,zz = np.meshgrid(radial_x,image_z)
        yy,zz = np.meshgrid(radial_y,image_z)

        coords = np.concatenate(np.transpose([xx, yy, zz], (1, 2, 0)))
        coords = (img_center + coords)/np.array(img.voxelsize)
        extra_mask = np.any(coords>(np.array(img.shape)-1),axis=1).reshape(xx.shape)
        coords = np.maximum(np.minimum(coords,np.array(img.shape)-1),0)
        coords = tuple(np.transpose(coords.astype(int)))
        slice_img = img.get_array()[coords].reshape(xx.shape)
        slice_img[extra_mask] = 0
        slice_images += [slice_img]

    return np.max(slice_images,axis=0)


def sequence_image_primordium_slices(sequence_name, image_dirname, save_files=False, signal_names=None, filenames=None, primordia_range=range(-3,6), reference_name='TagBFP', resolution=None, r_max=120., microscope_orientation=-1, verbose=False, debug=False, loglevel=0):

    aligned_images = sequence_aligned_signal_images(sequence_name, image_dirname, save_files=False, signal_names=signal_names, microscope_orientation=microscope_orientation, verbose=verbose, debug=debug, loglevel=loglevel + 1)
    primordia_data = load_sequence_primordia_data(sequence_name, image_dirname, verbose=verbose, debug=debug, loglevel=loglevel+1)

    if signal_names is None:
        signal_names = list(aligned_images.keys())

    image_slices = {}
    for signal_name in signal_names:
        image_slices[signal_name] = {}
        for primordium in primordia_range:
            image_slices[signal_name][primordium] = {}

    if filenames is None:
        filenames = np.sort(list(aligned_images[reference_name].keys()))

    if len(filenames) > 0:
        file_times = np.array([int(f[-2:]) for f in filenames])

        image_centers = {}
        for i_time, (time, filename) in enumerate(zip(file_times, filenames)):
            reference_img = aligned_images[signal_names[0]][filename]
            img_center = (np.array(reference_img.shape) * np.array(reference_img.voxelsize)) / 2.
            # img_center[2] = reference_img.shape[2] * reference_img.voxelsize[2] / 8.
            image_centers[filename] = img_center

        for i_time, (time, filename) in enumerate(zip(file_times, filenames)):

            reference_img = aligned_images[reference_name][filename]
            size = np.array(reference_img.shape)
            voxelsize = microscope_orientation * np.array(reference_img.voxelsize)

            if resolution is None:
                resolution = np.abs(voxelsize)[0]

            img_z = np.arange(size[2]) * voxelsize[2] - img_center[2]
            img_r = np.arange(r_max/resolution) * resolution
            rr, zz = map(np.transpose,np.meshgrid(img_r,img_z))
            extent = rr.min(), rr.max(), zz.max(), zz.min()

            for primordium in primordia_range:
                primordium_data = pd.concat([primordia_data[f][primordia_data[f]['primordium'] == primordium] for f in filenames])
                if len(primordium_data) > 0:
                    primordium_theta = (primordium * golden_angle + 180) % 360 - 180
                    primordium_theta = primordium_theta + np.mean(primordium_data['aligned_theta'].values - primordium_theta)
                    primordium_theta = (primordium_theta + 180) % 360 - 180
                    print(primordium,primordium_theta)

                    for i_signal, signal_name in enumerate(signal_names):
                        start_time = current_time()
                        logging.info("".join(["  " for l in range(loglevel)]) + "  --> Slicing P"+str(primordium)+" : " + filename + " " + signal_name)

                        # image_theta = primordium_theta # identity
                        # image_theta = -primordium_theta # flip X
                        # image_theta = 180 - primordium_theta # flip Y
                        image_theta = 180 + primordium_theta # flip X + flip Y
                        # image_theta = 90 - primordium_theta # transpose
                        # image_theta = primordium_theta - 90 # transpose + flip X
                        # image_theta = primordium_theta + 90 # transpose + flip Y

                        slice_img = image_angular_slice(aligned_images[signal_name][filename],theta=image_theta,extent=(0,r_max),width=0. if signal_name in ['PI','PIN1'] else 2.)

                        image_slices[signal_name][primordium][filename] = SpatialImage(np.transpose(slice_img),voxelsize=(resolution,reference_img.voxelsize[2]))

                        logging.info("".join(["  " for l in range(loglevel)]) + "  <-- Slicing P"+str(primordium)+" : " + filename + " " + signal_name + " [" + str(current_time() - start_time) + " s]")

            if save_files:
                logging.info("".join(["  " for l in range(loglevel)])+"--> Saving primordium signal images : "+filename+" "+str(signal_names))
                for i_signal, signal_name in enumerate(signal_names):
                    for primordium in primordia_range:
                        if filename in image_slices[signal_name][primordium].keys():
                            image_filename = image_dirname+"/"+sequence_name+"/"+filename+"/"+filename+"_P"+str(primordium)+"_"+signal_name+"_slice.tif"
                            imsave2d(image_filename,image_slices[signal_name][primordium][filename])

    return image_slices


def sequence_signal_data_primordium_slices(sequence_name, image_dirname, filenames=None, primordia_range=range(-3,6), width=2., microscope_orientation=-1, verbose=False, debug=False, loglevel=0):

    signal_images = load_sequence_signal_images(sequence_name, image_dirname, signal_names=['TagBFP'], verbose=verbose, debug=debug, loglevel=loglevel + 1)
    aligned_signal_data = load_sequence_signal_data(sequence_name, image_dirname, normalized=True, aligned=True, verbose=verbose, debug=debug, loglevel=loglevel + 1)
    signal_data = load_sequence_signal_data(sequence_name, image_dirname, normalized=True, aligned=False, verbose=verbose, debug=debug, loglevel=loglevel + 1)
    primordia_data = load_sequence_primordia_data(sequence_name, image_dirname, verbose=verbose, debug=debug, loglevel=loglevel + 1)

    if filenames is None:
        filenames = np.sort(list(signal_data.keys()))

    if len(filenames) > 0:
        file_times = np.array([int(f[-2:]) for f in filenames])

        signal_data_slices = {}
        for primordium in primordia_range:
            signal_data_slices[primordium] = {}

        alignment_transformations = {}

        for i_time, (time, filename) in enumerate(zip(file_times, filenames)):

            reference_img = list(signal_images.values())[0][filename]

            file_data = aligned_signal_data[filename]
            file_data = file_data[file_data['layer'] == 1]

            img_points = file_data[['center_'+dim for dim in ['x','y','z']]].values
            aligned_points = file_data[['aligned_'+dim for dim in ['x','y','z']]].values

            alignment_transformation = pts2transfo(microscope_orientation * img_points, microscope_orientation * aligned_points)

            reflection = np.sign(alignment_transformation[0, 0] * alignment_transformation[1, 1]) == -1
            if reflection:
                img_points[:,1] =  microscope_orientation * reference_img.shape[1] * reference_img.voxelsize[1] - img_points[:,1]
                alignment_transformation = pts2transfo(microscope_orientation * img_points, microscope_orientation * aligned_points)

            alignment_transformations[filename] = alignment_transformation

            file_data = signal_data[filename]

            image_points = file_data[['center_'+dim for dim in ['x','y','z']]].values
            if reflection:
                image_points[:, 1] = microscope_orientation * reference_img.shape[1] * reference_img.voxelsize[1] - image_points[:, 1]

            homogeneous_points = np.concatenate([microscope_orientation * image_points,np.ones((len(file_data),1))],axis=1)
            aligned_homogeneous_points = np.einsum("...ij,...j->...i",alignment_transformation,homogeneous_points)

            file_data['aligned_x'] = microscope_orientation * aligned_homogeneous_points[:,0]
            file_data['aligned_y'] = microscope_orientation * aligned_homogeneous_points[:,1]
            file_data['aligned_z'] = microscope_orientation * aligned_homogeneous_points[:,2]

            file_data['radial_distance'] = np.linalg.norm([file_data['aligned_x'], file_data['aligned_y']], axis=0)
            file_data['aligned_theta'] = 180. / np.pi * np.sign(file_data['aligned_y']) * np.arccos(file_data['aligned_x'] / file_data['radial_distance'])

            aligned_points = file_data[['aligned_'+dim for dim in ['x','y','z']]].values

            for primordium in primordia_range:
                primordium_data = pd.concat([primordia_data[f][primordia_data[f]['primordium'] == primordium] for f in filenames])

                primordium_theta = (primordium * golden_angle + 180) % 360 - 180
                if len(primordium_data) > 0:
                    primordium_theta = primordium_theta + np.mean(primordium_data['aligned_theta'].values - primordium_theta)
                    primordium_theta = (primordium_theta + 180) % 360 - 180

                primordium_plane_normal = np.array([-np.sin(np.radians(primordium_theta)),np.cos(np.radians(primordium_theta)),0])
                primordium_plane_dot_products = np.einsum("...ij,...j->...i",aligned_points,primordium_plane_normal)

                primordium_vector = np.array([np.cos(np.radians(primordium_theta)), np.sin(np.radians(primordium_theta)), 0])
                primordium_dot_products = np.einsum("...ij,...j->...i",aligned_points,primordium_vector)

                file_primordium_data = file_data[(np.abs(primordium_plane_dot_products)<width)&(primordium_dot_products>-width)]
                # file_primordium_data = file_data[(np.abs(primordium_plane_dot_products)<width)&(primordium_dot_products>0)]

                signal_data_slices[primordium][filename] = file_primordium_data

        return signal_data_slices
