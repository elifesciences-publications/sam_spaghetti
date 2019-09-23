import numpy as np
import pandas as pd

import sam_spaghetti
from sam_spaghetti.sam_microscopy_loading import load_image_from_microscopy
from sam_spaghetti.sam_sequence_info import get_experiment_name, get_experiment_microscopy, get_nomenclature_name, get_experiment_channels, get_experiment_signals, get_experiment_reference, \
    get_sequence_orientation, get_experiment_microscope_orientation
from sam_spaghetti.detection_quantification import detect_and_quantify
from sam_spaghetti.sam_sequence_loading import load_sequence_signal_images, load_sequence_signal_image_slices, load_sequence_signal_data, load_sequence_signal_wall_data, load_sequence_wall_meshes
from sam_spaghetti.segmentation_quantification import segment_and_quantify
from sam_spaghetti.polarity_quantification import extract_sequence_walls, compute_sequence_wall_polarities
from sam_spaghetti.signal_image_slices import sequence_signal_image_slices, sequence_image_primordium_slices, sequence_signal_data_primordium_slices
from sam_spaghetti.signal_image_plot import signal_image_plot, signal_nuclei_plot, signal_map_plot, signal_image_all_primordia_plot, signal_nuclei_all_primordia_plot, signal_map_all_primordia_plot, \
    signal_wall_plot
from sam_spaghetti.signal_map_computation import compute_signal_maps, compute_primordia_signal_maps, compute_average_signal_maps, compute_average_primordia_signal_maps
from sam_spaghetti.sequence_image_registration import register_sequence_images, apply_sequence_registration
from sam_spaghetti.signal_data_compilation import compile_signal_data, compile_primordia_data
from sam_spaghetti.sequence_growth_estimation import compute_growth
from sam_spaghetti.sam_sequence_primordia_alignment import align_sam_sequence, detect_organ_primordia

from tissue_nukem_3d.signal_map import save_signal_map

import logging
import argparse
import os

from timagetk.algorithms.reconstruction import pts2transfo

guillaume_dirname = "/Users/gcerutti/Data/"
calculus_dirname = "/projects/SamMaps/"
sam_spaghetti_dirname = sam_spaghetti.__path__[0] + "/../../share/data"

# dirname = guillaume_dirname
# dirname = calculus_dirname
dirname = sam_spaghetti_dirname

max_sam_id = 100
max_time = 100

plot_choices = ['sequence_raw', 'sequence_registered', 'sequence_aligned', 'sequence_primordia', 'experiment_aligned', 'experiment_primordia', 'all_aligned', 'all_primordia']


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiments', help='List of experiment identifiers', nargs='+', required=True)
    parser.add_argument('-dir', '--data-directory', help='Path to SAM sequence data files directory (nomenclature, orientation...)', default=dirname)
    parser.add_argument('-Mdir', '--microscopy-directory', help='Path to CZI image directory [default : data_directory/microscopy]', default=None)
    parser.add_argument('-o', '--output-directory', help='Path to segmented image directory [default : data_directory/membrane_images]', default=None)
    parser.add_argument('-S', '--segmentation', default=False, action='store_true', help='Run membrane segmentation on all experiments')
    parser.add_argument('-s', '--save-channels', default=False, action='store_true', help='Save INR image files for each microscopy image channel')
    parser.add_argument('-W', '--wall-extraction', default=False, action='store_true', help='Run sequence wall mesh extraction on all experiments')
    parser.add_argument('-P', '--polarity-quantification', default=False, action='store_true', help='Run sequence wall polarity computation on all experiments')
    parser.add_argument('-R', '--registration', default=False, action='store_true', help='Run sequence image registration on all experiments')
    parser.add_argument('-i', '--image-plot', default=[], nargs='+', help='List of image projections types to plot', choices=plot_choices)
    parser.add_argument('-c', '--cell-plot', default=[], nargs='+', help='List of signal map types to plot', choices=plot_choices)
    parser.add_argument('-w', '--wall-plot', default=[], nargs='+', help='List of wall types to plot', choices=plot_choices)
    parser.add_argument('-p', '--projection-type', default='max_intensity', help='Projection type for the image plots', choices=['max_intensity', 'L1_slice'])
    parser.add_argument('-N', '--normalized', default=False, action='store_true', help='Display normalized signals when plotting')

    parser.add_argument('-v', '--verbose', default=False, action='store_true', help='Verbose')
    parser.add_argument('-d', '--debug', default=False, action='store_true', help='Debug')

    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO if args.verbose else logging.DEBUG if args.debug else logging.ERROR)

    data_dirname = args.data_directory

    microscopy_dirname = args.microscopy_directory if args.microscopy_directory is not None else data_dirname+"/microscopy"
    if not os.path.exists(microscopy_dirname):
        logging.warning(microscopy_dirname+" does not exist!")
        logging.warning("Microscopy directory not found! No detection will be performed.")

    experiments = args.experiments
    image_dirname = args.output_directory if args.output_directory is not None else data_dirname+"/membrane_images"

    for exp in experiments:
        experiment_name = get_experiment_name(exp,data_dirname)
        if experiment_name == "":
            logging.error("Experiment identifier \""+exp+"\" not recognized (consider adding it to the experiment_info.csv file in the data directory)")
            experiments.remove(exp)
        else:
            if args.segmentation and (microscopy_dirname is not None):
                experiment_dirname = microscopy_dirname+"/"+get_experiment_microscopy(exp,data_dirname)
                if os.path.exists(experiment_dirname+"/RAW"):
                    experiment_dirname += "/RAW"

                if not os.path.exists(experiment_dirname):
                    logging.warning("Microscopy directory not found for "+exp+", no segmentation will be performed.")
                else:
                    microscopy_filenames = [experiment_dirname+"/"+f for f in os.listdir(experiment_dirname) if np.any([ext in f for ext in ['.czi','.lsm']])]
                    nomenclature_names = [get_nomenclature_name(microscopy_filename,data_dirname) for microscopy_filename in microscopy_filenames]
                    nomenclature_names = [n for n in nomenclature_names if n is not None]

                    is_not_processed = dict(zip(nomenclature_names, [False for f in nomenclature_names]))
                    if args.segmentation:
                        is_not_processed = dict(zip(nomenclature_names,[not os.path.exists(image_dirname+"/"+filename[:-4]+"/"+filename+"/"+filename+"_cell_data.csv") for filename in nomenclature_names]))

                    channel_names = get_experiment_channels(exp, data_dirname)
                    reference_name = get_experiment_reference(exp, data_dirname)

                    if not os.path.exists(image_dirname):
                        os.makedirs(image_dirname)

                    for microscopy_filename in microscopy_filenames:
                        nomenclature_name = get_nomenclature_name(microscopy_filename,data_dirname)

                        if nomenclature_name is not None:
                            sequence_name = nomenclature_name[:-4]
                            if not os.path.exists(image_dirname+"/"+sequence_name):
                                os.makedirs(image_dirname+"/"+sequence_name)
                            if not os.path.exists(image_dirname+"/"+sequence_name+"/"+nomenclature_name):
                                os.makedirs(image_dirname+"/"+sequence_name+"/"+nomenclature_name)

                            if args.segmentation or is_not_processed[nomenclature_name]:
                                logging.error("--> Running segmentation on "+nomenclature_name)
                                img_dict = load_image_from_microscopy(microscopy_filename, save_images=args.save_channels, image_dirname=image_dirname, nomenclature_name=nomenclature_name, channel_names=channel_names, verbose=args.verbose, debug=args.debug, loglevel=1)
                                segment_and_quantify(img_dict, image_dirname=image_dirname, nomenclature_name=nomenclature_name, membrane_name=reference_name, verbose=args.verbose, debug=args.debug, loglevel=1)
                            else:
                                logging.info("--> Found segmentation output for "+nomenclature_name)
                        else:
                            logging.warning("--> No nomenclature found for " + microscopy_filename + ", skipping...")

    if not os.path.exists(image_dirname):
        logging.error("Result output directory not found, nothing left to do!")
    else:
        sequence_names = {}
        for exp in experiments:
            experiment_name = get_experiment_name(exp,data_dirname)
            reference_name = get_experiment_reference(exp, data_dirname)
            logging.info("--> Loading sequences for experiment "+str(exp))

            sequence_names[exp] = []
            for sam_id in range(max_sam_id):
                sequence_name = experiment_name+"_sam"+str(sam_id).zfill(2)
                logging.debug("--> Trying to load sequence "+str(sequence_name))
                signal_data = load_sequence_signal_data(sequence_name, image_dirname, normalized=False, aligned=False, verbose=args.verbose, debug=args.debug, loglevel=1)
                if len(signal_data)>0:
                    sequence_names[exp] += [sequence_name]
                    logging.debug("--> Loaded sequence "+str(sequence_name)+"!")
                else:
                    signal_images = load_sequence_signal_images(sequence_name, image_dirname, signal_names=[reference_name],verbose=args.verbose, debug=args.debug, loglevel=1)
                    if len(signal_images) > 0:
                        sequence_names[exp] += [sequence_name]
                        logging.debug("--> Loaded sequence "+str(sequence_name)+"!")

        for exp in experiments:
            reference_name = get_experiment_reference(exp, data_dirname)
            microscope_orientation = get_experiment_microscope_orientation(exp, data_dirname)
            for sequence_name in sequence_names[exp]:
                if 'sequence_raw' in args.image_plot:
                    logging.info("--> Plotting signal images "+sequence_name)
                    # signal_images = load_sequence_signal_images(sequence_name, image_dirname, verbose=args.verbose, debug=args.debug, loglevel=1)
                    # signal_data = load_sequence_signal_data(sequence_name, image_dirname, normalized=False, aligned=False, verbose=args.verbose, debug=args.debug, loglevel=1)
                    signal_image_slices = load_sequence_signal_image_slices(sequence_name, image_dirname, projection_type=args.projection_type, aligned=False, verbose=args.verbose, debug=args.debug, loglevel=1)
                    if len(signal_image_slices)==0:
                        signal_image_slices = sequence_signal_image_slices(sequence_name, image_dirname, reference_name=reference_name, microscope_orientation=microscope_orientation, projection_type=args.projection_type, resolution=None, aligned=False, save_files=True, verbose=args.verbose, debug=args.debug, loglevel=1)
                    figure = signal_image_plot(signal_image_slices, reference_name=reference_name, projection_type=args.projection_type, resolution=0.25, aligned=False, verbose=args.verbose, debug=args.debug, loglevel=1)
                    figure.savefig(image_dirname+"/"+sequence_name+"/"+sequence_name+"_"+args.projection_type+"_signals.png")

        for exp in experiments:
            reference_name = get_experiment_reference(exp, data_dirname)
            signal_names = get_experiment_signals(exp, data_dirname)
            microscope_orientation = get_experiment_microscope_orientation(exp, data_dirname)
            for sequence_name in sequence_names[exp]:
                if args.wall_extraction:
                    logging.info("--> Sequence wall mesh extraction "+sequence_name)
                    reference_name = get_experiment_reference(exp, data_dirname)
                    extract_sequence_walls(sequence_name, save_files=True, image_dirname=image_dirname, membrane_name=reference_name, microscope_orientation=microscope_orientation, resampling_voxelsize=0.5, target_edge_length=2., verbose=args.verbose, debug=args.debug, loglevel=1)

                if args.polarity_quantification:
                    logging.info("--> Sequence "+str(signal_names)+" wall polarity computation "+sequence_name)
                    compute_sequence_wall_polarities(sequence_name, save_files=True, image_dirname=image_dirname, membrane_name=reference_name, signal_names=signal_names, loglevel=1)

                if 'sequence_raw' in args.wall_plot:
                    wall_topomeshes = load_sequence_wall_meshes(sequence_name, image_dirname, loglevel=1)
                    # wall_data = load_sequence_signal_wall_data(sequence_name, image_dirname, loglevel=1)
                    signal_images = load_sequence_signal_images(sequence_name, image_dirname, signal_names=[reference_name], verbose=args.verbose, debug=args.debug, loglevel=1)
                    r_max = signal_images[reference_name].values()[0].shape[0] * signal_images[reference_name].values()[0].voxelsize[0] / 2.
                    logging.info("--> Plotting wall signals "+sequence_name)
                    signals_to_plot = signal_names + [s+"_polarity_vector" for s in signal_names]
                    figure = signal_wall_plot(wall_topomeshes, reference_name=reference_name, signal_names=signals_to_plot, r_max=r_max, linewidth=3, aligned=False, verbose=args.verbose, debug=args.debug, loglevel=1)
                    figure.savefig(image_dirname+"/"+sequence_name+"/"+sequence_name+"_walls.png")

        for exp in experiments:
            reference_name = get_experiment_reference(exp, data_dirname)
            signal_names = get_experiment_signals(exp, data_dirname)
            microscope_orientation = get_experiment_microscope_orientation(exp, data_dirname)
            for sequence_name in sequence_names[exp]:
                if 'sequence_raw' in args.cell_plot:
                    signal_data = load_sequence_signal_data(sequence_name, image_dirname, normalized=False, aligned=False, nuclei=False, verbose=args.verbose, debug=args.debug, loglevel=1)
                    signal_images = load_sequence_signal_images(sequence_name, image_dirname, signal_names=[reference_name], verbose=args.verbose, debug=args.debug, loglevel=1)
                    r_max = signal_images[reference_name].values()[0].shape[0]*signal_images[reference_name].values()[0].voxelsize[0]/2.
                    logging.info("--> Plotting detected cell signals "+sequence_name)
                    signals_to_plot = signal_names + [s+"_polarity_vector" for s in signal_names]
                    figure = signal_nuclei_plot(signal_data, r_max=r_max, normalized=args.normalized, signal_names=signals_to_plot, verbose=args.verbose, debug=args.debug, loglevel=1)
                    figure.savefig(image_dirname+"/"+sequence_name+"/"+sequence_name+"_L1_cell_signals.png")
