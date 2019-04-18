# -*- python -*-
# -*- coding: utf-8 -*-

import logging
from os.path import splitext

import numpy as np
import scipy.ndimage as nd

from timagetk.io import imsave
from timagetk.components import SpatialImage
from timagetk.plugins import morphology
from timagetk.plugins import h_transform
from timagetk.plugins import region_labeling
from timagetk.plugins import linear_filtering
from timagetk.plugins import segmentation
from timagetk.algorithms.resample import isometric_resampling, resample
from timagetk.algorithms.exposure import z_slice_contrast_stretch, z_slice_equalize_adapthist

from vplants.tissue_analysis.spatial_image_analysis import SpatialImageAnalysis

def splitext_zip(fname):
    """
    Returns filename and extension of fname
    Unsensitive to 'gz' or 'zip' extensions
    """
    base_fname, ext = splitext(fname)
    if ext == '.gz' or ext == '.zip':
        base_fname, ext2 = splitext(base_fname)
        ext = ''.join([ext2, ext])
    return base_fname, ext

def segmentation_fname(img2seg_fname, h_min, iso, equalize, stretch):
    """
    Generate the segmentation filename using some of the pipeline steps.

    Parameters
    ----------
    img2seg_fname : str
        filename of the image to segment.
    h_min : int
        h-minima used with the h-transform function
    iso : bool
        indicate if isometric resampling was performed by the pipeline
    equalize : bool
        indicate if adaptative equalization of intensity was performed
    stretch : bool
        indicate if intensity histogram stretching was performed
    """
    suffix = '_seg'
    suffix += '-iso' if iso else ''
    suffix += '-adpat_eq' if equalize else ''
    suffix += '-hist_stretch' if stretch else ''
    suffix += '-h_min{}'.format(h_min)
    seg_im_fname = splitext_zip(img2seg_fname)[0] + suffix + '.inr.gz'
    return seg_im_fname


def signal_subtraction(img2seg, img2sub):
    """
    Performs SpatialImage subtraction.

    Parameters
    ----------
    img2seg : str
        image to segment.
    img2sub : str, optional
        image to subtract to the image to segment.
    """
    vxs = img2seg.voxelsize
    ori = img2seg.origin()
    md = img2seg.metadata

    try:
        assert np.allclose(img2seg.shape, img2sub.shape)
    except AssertionError:
        raise ValueError("Input images does not have the same shape!")
    # img2sub = morphology(img2sub, method='erosion', radius=3.)
    tmp_im = img2seg - img2sub
    tmp_im[img2seg <= img2sub] = 0
    img2seg = SpatialImage(tmp_im, voxelsize=vxs, origin=ori, metadata_dict=md)

    return img2seg


def membrane_image_segmentation(img2seg, h_min, img2sub=None, iso=False, equalize=True, stretch=False, std_dev=1.0, min_cell_volume=20., max_cell_volume=2000., back_id=1, to_8bits=False):
    """
    Define the segmentation pipeline

    Parameters
    ----------
    img2seg : str
        image to segment.
    h_min : int
        h-minima used with the h-transform function
    img2sub : str, optional
        image to subtract to the image to segment.
    iso : bool, optional
        if True (default), isometric resampling is performed after h-minima
        detection and before watershed segmentation
    equalize : bool, optional
        if True (default), intensity adaptative equalization is performed before
        h-minima detection
    stretch : bool, optional
        if True (default, False), intensity histogram stretching is performed
        before h-minima detection
    std_dev : float, optional
        standard deviation used for Gaussian smoothing of the image to segment
    min_cell_volume : float, optional
        minimal volume accepted in the segmented image
    back_id : int, optional
        the background label
    to_8bits : bool, optional
        transform the image to segment as an unsigned 8 bits image for the h-transform
        and seed-labelleing steps

    Returns
    -------
    seg_im : SpatialImage
        the labelled image obtained by seeded-watershed

    Notes
    -----
      * Both 'equalize' & 'stretch' can not be True at the same time since they work
        on the intensity of the pixels;
      * Signal subtraction is performed after intensity rescaling (if any);
      * Linear filtering (Gaussian smoothing) is performed before h-minima transform
        for local minima detection;
      * Gaussian smoothing should be performed on isometric images, if the provided
        image is not isometric, we resample it before smoothing, then go back to
        original voxelsize;
      * In any case H-Transfrom is performed on the image with its native resolution
        to speed upd seed detection;
      * Same goes for connexe components detection (seed labelling);
      * Segmentation will be performed on the isometric images if iso is True, in
        such case we resample the image of detected seeds and use the isometric
        smoothed intensity image;
    """
    # - Check we have only one intensity rescaling method called:
    try:
        assert equalize + stretch < 2
    except AssertionError:
        raise ValueError("Both 'equalize' & 'stretch' can not be True at once!")
    # - Check the standard deviation value for Gaussian smoothing is valid:
    try:
        assert std_dev <= 1.
    except AssertionError:
        raise ValueError("Standard deviation for Gaussian smoothing should be superior or equal to 1!")

    ori_vxs = img2seg.voxelsize
    ori_shape = img2seg.shape
    if equalize:
        print "\n - Performing z-slices adaptative histogram equalisation on the intensity image to segment..."
        img2seg = z_slice_equalize_adapthist(img2seg)
    if stretch:
        print "\n - Performing z-slices histogram contrast stretching on the intensity image to segment..."
        img2seg = z_slice_contrast_stretch(img2seg)
    if img2sub is not None:
        print "\n - Performing signal substraction..."
        img2seg = signal_subtraction(img2seg, img2sub)

    print "\n - Automatic seed detection...".format(h_min)
    # morpho_radius = 1.0
    # asf_img = morphology(img2seg, max_radius=morpho_radius, method='co_alternate_sequential_filter')
    # ext_img = h_transform(asf_img, h=h_min, method='h_transform_min')

    voxelsize = np.array(ori_vxs)
    print " -- Gaussian smoothing with sigma={}...".format(std_dev / voxelsize)
    smooth_image = nd.gaussian_filter(img2seg.get_array(), sigma=std_dev / voxelsize).astype(img2seg.get_array().dtype)
    smooth_img = SpatialImage(smooth_image, voxelsize=img2seg.voxelsize)

    if iso:
        print " -- Isometric resampling..."
        iso_img = isometric_resampling(img2seg)
        iso_smooth_img = isometric_resampling(smooth_img)

    print " -- H-minima transform with h-min={}...".format(h_min)
    if to_8bits:
        ext_img = h_transform(smooth_img.to_8bits(), h=h_min, method='h_transform_min')
    else:
        ext_img = h_transform(smooth_img, h=h_min, method='h_transform_min')
    if iso:
        smooth_img = iso_smooth_img  # no need to keep both images after this step!

    print " -- Region labelling: connexe components detection..."
    seed_img = region_labeling(ext_img, low_threshold=1, high_threshold=h_min, method='connected_components')
    print "Detected {} seeds!".format(len(np.unique(seed_img)) - 1)  # '0' is in the list!
    del ext_img  # no need to keep this image after this step!

    print "\n - Performing seeded watershed segmentation..."
    if iso:
        seed_img = isometric_resampling(seed_img, option='label')
    if to_8bits:
        seg_im = segmentation(smooth_img.to_8bits(), seed_img, method='seeded_watershed', try_plugin=False)
    else:
        seg_im = segmentation(smooth_img, seed_img, method='seeded_watershed', try_plugin=False)
    # seg_im[seg_im == 0] = back_id
    print "Detected {} labels!".format(len(np.unique(seg_im)))

    if min_cell_volume > 0.:
        print "\n - Performing cell volume filtering..."
        spia = SpatialImageAnalysis(seg_im, background=None)
        vol = spia.volume()
        too_small_labels = [k for k, v in vol.items() if v < min_cell_volume and k != 0]
        if too_small_labels != []:
            print "Detected {} labels with a volume < {}µm2".format(len(too_small_labels), min_cell_volume)
            print " -- Removing seeds leading to small cells..."
            spia = SpatialImageAnalysis(seed_img, background=None)
            seed_img = spia.get_image_without_labels(too_small_labels)
            print " -- Performing final seeded watershed segmentation..."
            seg_im = segmentation(smooth_img, seed_img, method='seeded_watershed', try_plugin=False)
            # seg_im[seg_im == 0] = back_id
            print "Detected {} labels!".format(len(np.unique(seg_im)))

    if max_cell_volume is not None:
        seg_volumes = dict(zip(np.arange(seg_im.max()) + 1, nd.sum(np.prod(voxelsize) * np.ones_like(seg_im), seg_im, np.arange(seg_im.max()) + 1)))
        labels_to_remove = np.array(seg_volumes.keys())[np.array(seg_volumes.values()) > max_cell_volume]
        for l in labels_to_remove:
            seg_im[seg_im == l] = back_id

    return seg_im


def segment_and_quantify(img_dict, membrane_name='PI', signal_names=None, save_files=True, image_dirname=None, nomenclature_name=None, microscope_orientation=-1, h_min=None, verbose=True, debug=False, loglevel=0):
    """

    :param img_dict:
    :param membrane_name:
    :param signal_names:
    :param save_files:
    :param image_dirname:
    :param nomenclature_name:
    :param microscope_orientation:
    :param verbose:
    :param debug:
    :param loglevel:
    :return:
    """

    sequence_name = nomenclature_name[:-4]

    membrane_img = img_dict[membrane_name]

    if h_min is None:
        h_min = 200 if membrane_img.dtype == np.uint16 else 2
    iso = False
    equalize = True
    stretch = False
    std_dev = 0.8

    seg_img = membrane_image_segmentation(membrane_img, h_min, iso=iso, equalize=equalize, stretch=stretch, std_dev=std_dev, min_cell_volume=0., back_id=1, to_8bits=False)

    if save_files:
        seg_name = segmentation_fname(nomenclature_name + "_" + membrane_name, h_min, iso, equalize, stretch)
        seg_file = image_dirname+"/"+sequence_name+"/"+nomenclature_name+"/" + seg_name
        imsave(seg_file, seg_img)
        seg_file = image_dirname+"/"+sequence_name+"/"+nomenclature_name+"/" + nomenclature_name + "_" + membrane_name + "_seg.inr.gz"
        imsave(seg_file, seg_img)

    results = (seg_img,)

    return results
