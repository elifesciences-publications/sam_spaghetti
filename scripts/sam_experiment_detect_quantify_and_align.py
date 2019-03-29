import numpy as np
import pandas as pd

import sam_spaghetti
from sam_spaghetti.sam_sequence_info import get_experiment_name, get_experiment_microscopy, get_nomenclature_name, get_experiment_channels, get_experiment_reference, get_sequence_orientation
from sam_spaghetti.detection_quantification import detect_from_czi
from sam_spaghetti.sam_sequence_loading import load_sequence_signal_images, load_sequence_signal_image_slices, load_sequence_signal_data
from sam_spaghetti.signal_image_slices import sequence_signal_image_slices, sequence_image_primordium_slices, sequence_signal_data_primordium_slices
from sam_spaghetti.signal_image_plot import signal_image_plot, signal_nuclei_plot, signal_map_plot, signal_image_all_primordia_plot, signal_nuclei_all_primordia_plot, signal_map_all_primordia_plot
from sam_spaghetti.signal_map_computation import compute_signal_maps, compute_primordia_signal_maps, compute_average_signal_maps, compute_average_primordia_signal_maps
from sam_spaghetti.sequence_image_registration import register_sequence_images
from sam_spaghetti.signal_data_compilation import compile_signal_data, compile_primordia_data
from sam_spaghetti.sequence_growth_estimation import compute_surfacic_growth
from sam_spaghetti.sam_sequence_primordia_alignment import align_sam_sequence, detect_organ_primordia

import logging
import argparse
import os

from timagetk.algorithms.reconstruction import pts2transfo

guillaume_dirname = "/Users/gcerutti/Data/"
calculus_dirname = "/projects/SamMaps/"
sam_spaghetti_dirname = sam_spaghetti.__path__[0]+"/../../share/data"

# dirname = guillaume_dirname
# dirname = calculus_dirname
dirname = sam_spaghetti_dirname

max_sam_id = 100
max_time = 100

plot_choices = ['sequence_raw', 'sequence_aligned', 'sequence_primordia', 'experiment_aligned', 'experiment_primordia', 'all_aligned', 'all_primordia']

def main():
    """

    Returns:

    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiments', help='List of experiment identifiers', nargs='+', required=True)
    parser.add_argument('-dir', '--data-directory', help='Path to SAM sequence data files directory (nomenclature, orientation...)', default=dirname)
    parser.add_argument('-Mdir', '--microscopy-directory', help='Path to CZI image directory [default : data_directory/microscopy]', default=None)
    parser.add_argument('-Ndir', '--nuclei-directory', help='Path to detected nuclei directory [default : data_directory/nuclei_images]', default=None)
    parser.add_argument('-D', '--detection', default=False, action='store_true', help='Run nuclei detection on all experiments')
    parser.add_argument('-s', '--save-channels', default=False, action='store_true', help='Save INR image files for each microscopy image channel')
    parser.add_argument('-R', '--registration', default=False, action='store_true', help='Run sequence image registration on all experiments')
    parser.add_argument('-i', '--image-plot', default=[], nargs='+', help='List of image projections types to plot',choices=plot_choices)
    parser.add_argument('-p', '--projection-type', default='max_intensity', help='Projection type for the image plots',choices=['max_intensity', 'L1_slice'])
    parser.add_argument('-n', '--nuclei-plot', default=[], nargs='+', help='List of signal map types to plot',choices=plot_choices)
    parser.add_argument('-m', '--map-plot', default=[], nargs='+', help='List of signal map types to plot',choices=plot_choices)
    parser.add_argument('-N', '--normalized', default=False, action='store_true', help='Display normalized signals when plotting')
    parser.add_argument('-G', '--growth-estimation', default=False, action='store_true', help='Estimate surfacic growth on all experiments')
    parser.add_argument('-P', '--primordia-alignment', default=False, action='store_true', help='Align sequences of all experiments based on the detection of CZ and P0')
    parser.add_argument('-C', '--data-compilation', default=False, action='store_true', help='Compile all the data from the experiments into .csv files in the data directory')
    
    parser.add_argument('-v', '--verbose', default=False, action='store_true', help='Verbose')
    parser.add_argument('-d', '--debug', default=False, action='store_true', help='Debug')


    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO if args.verbose else logging.DEBUG if args.debug else logging.ERROR)

    # mesh_to_cvt_image(input=args.input, output=args.output, method=args.method, verbose=args.verbose, debug=args.debug,
    #     save=not(args.no_save), voxelsize=args.voxelsize, nbcells=args.nbcells, max_step=args.step, res=args.resolution)
    data_dirname = args.data_directory

    microscopy_dirname = args.microscopy_directory if args.microscopy_directory is not None else data_dirname+"/microscopy"
    if not os.path.exists(microscopy_dirname):
        logging.warning(microscopy_dirname+" does not exist!")
        logging.warning("Microscopy directory not found! No detection will be performed.")

    experiments = args.experiments
    image_dirname = args.nuclei_directory if args.nuclei_directory is not None else data_dirname+"/nuclei_images"

    for exp in experiments:
        experiment_name = get_experiment_name(exp,data_dirname)
        if experiment_name == "":
            logging.error("Experiment identifier \""+exp+"\" not recognized (consider adding it to the experiment data file in the data directory)")
            experiments.remove(exp)
        else:
            if args.detection and (microscopy_dirname is not None):
                experiment_dirname = microscopy_dirname+"/"+get_experiment_microscopy(exp,data_dirname)
                if os.path.exists(experiment_dirname+"/RAW"):
                    experiment_dirname += "/RAW"

                if not os.path.exists(experiment_dirname):
                    logging.warning("Microscopy directory not found for "+exp+", no detection will be performed.")
                else:                    
                    czi_filenames = [experiment_dirname+"/"+f for f in os.listdir(experiment_dirname) if '.czi' in f]
                    nomenclature_names = [get_nomenclature_name(czi_filename,data_dirname) for czi_filename in czi_filenames]
                    nomenclature_names = [n for n in nomenclature_names if n is not None]
                    is_not_detected = dict(zip(nomenclature_names,[not os.path.exists(image_dirname+"/"+filename[:-4]+"/"+filename+"/"+filename+"_signal_data.csv") for filename in nomenclature_names]))

                    channel_names = get_experiment_channels(exp, data_dirname)
                    reference_name = get_experiment_reference(exp, data_dirname)

                    # if args.detection or np.any(is_not_detected.values()):
                    if not os.path.exists(image_dirname):
                        os.makedirs(image_dirname)

                    for czi_filename in czi_filenames:
                        nomenclature_name = get_nomenclature_name(czi_filename,data_dirname)

                        if nomenclature_name is not None:
                            sequence_name = nomenclature_name[:-4]

                            if not os.path.exists(image_dirname+"/"+sequence_name):
                                os.makedirs(image_dirname+"/"+sequence_name)

                            if not os.path.exists(image_dirname+"/"+sequence_name+"/"+nomenclature_name):
                                os.makedirs(image_dirname+"/"+sequence_name+"/"+nomenclature_name)

                            if args.detection or is_not_detected[nomenclature_name]:
                                logging.info("--> Running detection on "+nomenclature_name)
                                detect_from_czi(czi_filename, save_files=True, save_images=args.save_channels, image_dirname=image_dirname, nomenclature_name=nomenclature_name, channel_names=channel_names, reference_name=reference_name, verbose=args.verbose, debug=args.debug, loglevel=1)
                            else:
                                logging.info("--> Found detection output for "+nomenclature_name)
                        else:
                            logging.warning("--> No nomenclature found for " + czi_filename + ", skipping...")

    if not os.path.exists(image_dirname):
        logging.error("Result output directory not found, nothing left to do!")
    else:
        sequence_signal_data = {}
        for exp in experiments:
            experiment_name = get_experiment_name(exp,data_dirname)

            sequence_signal_data[exp] = {}
            for sam_id in xrange(max_sam_id):
                sequence_name = experiment_name+"_sam"+str(sam_id).zfill(2)
                logging.debug("-> Trying to load sequence "+str(sequence_name))
                signal_data = load_sequence_signal_data(sequence_name, image_dirname, normalized=False, aligned=False, verbose=args.verbose, debug=args.debug, loglevel=1)
                if len(signal_data)>0:
                    sequence_signal_data[exp][sequence_name] = signal_data
                    logging.debug("-> Loaded sequence "+str(sequence_name)+"!")

        for exp in experiments:
            for sequence_name in sequence_signal_data[exp]:
                if 'sequence_raw' in args.nuclei_plot:
                    signal_data = load_sequence_signal_data(sequence_name, image_dirname, normalized=False, aligned=False, verbose=args.verbose, debug=args.debug, loglevel=1)
                    logging.info("--> Plotting detected nuclei signals "+sequence_name)
                    figure = signal_nuclei_plot(signal_data, normalized=args.normalized, verbose=args.verbose, debug=args.debug, loglevel=1)
                    figure.savefig(image_dirname+"/"+sequence_name+"/"+sequence_name+"_L1_nuclei_signals.png")

                if 'sequence_raw' in args.image_plot:
                    logging.info("--> Plotting signal images "+sequence_name)
                    # signal_images = load_sequence_signal_images(sequence_name, image_dirname, verbose=args.verbose, debug=args.debug, loglevel=1)
                    # signal_data = load_sequence_signal_data(sequence_name, image_dirname, normalized=False, aligned=False, verbose=args.verbose, debug=args.debug, loglevel=1)
                    signal_image_slices = load_sequence_signal_image_slices(sequence_name, image_dirname, projection_type=args.projection_type, aligned=False, verbose=args.verbose, debug=args.debug, loglevel=1)
                    if len(signal_image_slices)==0:
                        signal_image_slices = sequence_signal_image_slices(sequence_name, image_dirname, projection_type=args.projection_type, resolution=0.25, aligned=False, save_files=True, verbose=args.verbose, debug=args.debug, loglevel=1)
                    figure = signal_image_plot(signal_image_slices, projection_type=args.projection_type, resolution=0.25, aligned=False, verbose=args.verbose, debug=args.debug, loglevel=1)
                    figure.savefig(image_dirname+"/"+sequence_name+"/"+sequence_name+"_"+args.projection_type+"_signals.png")

        for exp in experiments:
            for sequence_name in sequence_signal_data[exp]:
                if args.registration:
                    logging.info("--> Sequence image registration "+sequence_name)
                    register_sequence_images(sequence_name, save_files=True, image_dirname=image_dirname, verbose=args.verbose, debug=args.debug, loglevel=1)

        if args.data_compilation:
            logging.info("--> Compiling signal data from all experiments "+str(experiments))
            compile_signal_data(experiments,save_files=True, image_dirname=image_dirname, data_dirname=data_dirname, verbose=args.verbose, debug=args.debug, loglevel=1)
                            
        for exp in experiments:
            for sequence_name in sequence_signal_data[exp]:
                if args.growth_estimation:
                    logging.info("--> Computing sequence surfacic growth "+sequence_name)
                    compute_surfacic_growth(sequence_name, image_dirname, save_files=True, verbose=args.verbose, debug=args.debug, loglevel=1)

                    signal_normalized_data = load_sequence_signal_data(sequence_name, image_dirname, normalized=True, aligned=False, verbose=args.verbose, debug=args.debug, loglevel=1)  
                    logging.info("--> Plotting nuclei growth "+sequence_name)
                    figure = signal_nuclei_plot(signal_normalized_data, normalized=args.normalized, signal_names=['next_relative_surfacic_growth','previous_relative_surfacic_growth'], registered=True, verbose=args.verbose, debug=args.debug, loglevel=1)
                    figure.savefig(image_dirname+"/"+sequence_name+"/"+sequence_name+"_L1_registered_nuclei_growth.png")  

                if 'sequence_raw' in args.map_plot:
                    signal_normalized_data = load_sequence_signal_data(sequence_name, image_dirname, normalized=True, aligned=False, verbose=args.verbose, debug=args.debug, loglevel=1)  
                    signal_maps = compute_signal_maps(signal_normalized_data, normalized=args.normalized, verbose=args.verbose, debug=args.debug, loglevel=1)
                    logging.info("--> Plotting maps "+sequence_name)
                    figure = signal_map_plot(signal_maps, verbose=args.verbose, debug=args.debug, loglevel=1)
                    figure.savefig(image_dirname+"/"+sequence_name+"/"+sequence_name+"_L1_signal_maps.png")  

        sequence_aligned_signal_data = {}
        sequence_primordia_signal_data = {}
        sequence_aligned_signal_maps = {}
        sequence_primordia_signal_maps = {}
        for exp in experiments:
            sequence_aligned_signal_data[exp] = {}
            sequence_primordia_signal_data[exp] = {}
            sequence_aligned_signal_maps[exp] = {}
            sequence_primordia_signal_maps[exp] = {}
            for sequence_name in sequence_signal_data[exp]:
                if args.primordia_alignment:
                    logging.info("--> Sequence primordia alignment "+sequence_name)
                    sam_orientation = get_sequence_orientation(sequence_name,data_dirname)
                    align_sam_sequence(sequence_name, image_dirname, sam_orientation=sam_orientation, save_files=True, verbose=args.verbose, debug=args.debug, loglevel=1)
                    detect_organ_primordia(sequence_name, image_dirname, sam_orientation=sam_orientation, save_files=True, verbose=args.verbose, debug=args.debug, loglevel=1)

                if np.any([p in args.nuclei_plot for p in ['sequence_aligned','experiment_aligned','all_aligned']]):
                    signal_data = load_sequence_signal_data(sequence_name, image_dirname, normalized=True, aligned=True, verbose=args.verbose, debug=args.debug, loglevel=1)
                    sequence_aligned_signal_data[exp][sequence_name] = signal_data

                if 'sequence_aligned' in args.nuclei_plot:
                    logging.info("--> Plotting aligned nuclei signals "+sequence_name)
                    figure = signal_nuclei_plot(sequence_aligned_signal_data[exp][sequence_name], aligned=True, normalized=args.normalized, verbose=args.verbose, debug=args.debug, loglevel=1)
                    figure.savefig(image_dirname+"/"+sequence_name+"/"+sequence_name+"_L1_aligned_nuclei_signals.png")

                if np.any([p in args.nuclei_plot for p in ['sequence_primordia','experiment_primordia','all_primordia']]):
                    primordia_signal_data = sequence_signal_data_primordium_slices(sequence_name, image_dirname, width=5., verbose=args.verbose, debug=args.debug, loglevel=1)
                    sequence_primordia_signal_data[exp][sequence_name] = primordia_signal_data

                if 'sequence_primordia' in args.nuclei_plot:
                    logging.info("--> Plotting primordia nuclei signals "+sequence_name)
                    figure = signal_nuclei_all_primordia_plot(sequence_primordia_signal_data[exp][sequence_name], normalized=args.normalized, verbose=args.verbose, debug=args.debug, loglevel=1)
                    figure.savefig(image_dirname + "/" + sequence_name + "/" + sequence_name + "_primordia_nuclei_signals.png")

                if np.any([p in args.map_plot for p in ['sequence_aligned','experiment_aligned','all_aligned']]):
                    signal_aligned_data = load_sequence_signal_data(sequence_name, image_dirname, normalized=True, aligned=True, verbose=args.verbose, debug=args.debug, loglevel=1)  
                    signal_aligned_maps = compute_signal_maps(signal_aligned_data, aligned=True, normalized=args.normalized, verbose=args.verbose, debug=args.debug, loglevel=1)
                    sequence_aligned_signal_maps[exp][sequence_name] = signal_aligned_maps

                if 'sequence_aligned' in args.map_plot:
                    logging.info("--> Plotting signal maps "+sequence_name)
                    figure = signal_map_plot(sequence_aligned_signal_maps[exp][sequence_name], aligned=True, verbose=args.verbose, debug=args.debug, loglevel=1)
                    figure.savefig(image_dirname+"/"+sequence_name+"/"+sequence_name+"_L1_aligned_signal_maps.png")

                if np.any([p in args.map_plot for p in ['sequence_primordia', 'experiment_primordia', 'all_primordia']]):
                    primordia_signal_data = sequence_signal_data_primordium_slices(sequence_name, image_dirname, width=5., verbose=args.verbose, debug=args.debug, loglevel=1)
                    primordia_signal_maps = compute_primordia_signal_maps(primordia_signal_data, normalized=args.normalized, verbose=args.verbose, debug=args.debug, loglevel=1)
                    sequence_primordia_signal_maps[exp][sequence_name] = primordia_signal_maps

                if 'sequence_primordia' in args.map_plot:
                    logging.info("--> Plotting primordia signal maps "+sequence_name)
                    figure = signal_map_all_primordia_plot(sequence_primordia_signal_maps[exp][sequence_name], verbose=args.verbose, debug=args.debug, loglevel=1)
                    figure.savefig(image_dirname + "/" + sequence_name + "/" + sequence_name + "_primordia_signal_maps.png")

                if 'sequence_aligned' in args.image_plot:
                    logging.info("--> Plotting signal images "+sequence_name)
                    # signal_images = load_sequence_signal_images(sequence_name, image_dirname, verbose=args.verbose, debug=args.debug, loglevel=1)
                    # signal_data = load_sequence_signal_data(sequence_name, image_dirname, normalized=True, aligned=True, verbose=args.verbose, debug=args.debug, loglevel=1)  
                    signal_image_slices = load_sequence_signal_image_slices(sequence_name, image_dirname, projection_type=args.projection_type, aligned=True, verbose=args.verbose, debug=args.debug, loglevel=1)
                    if len(signal_image_slices)==0:
                        signal_image_slices = sequence_signal_image_slices(sequence_name, image_dirname, projection_type=args.projection_type, resolution=0.25, aligned=True, verbose=args.verbose, debug=args.debug, loglevel=1)
                    figure = signal_image_plot(signal_image_slices, projection_type=args.projection_type, resolution=0.25, aligned=True, verbose=args.verbose, debug=args.debug, loglevel=1)
                    figure.savefig(image_dirname+"/"+sequence_name+"/"+sequence_name+"_"+args.projection_type+"_aligned_signals.png")

                if 'sequence_primordia' in args.image_plot:
                    logging.info("--> Plotting signal primordia images "+sequence_name)
                    signal_image_slices = sequence_image_primordium_slices(sequence_name, image_dirname, r_max=80, verbose=args.verbose, debug=args.debug, loglevel=1)
                    figure = signal_image_all_primordia_plot(signal_image_slices, r_max=80, verbose=args.verbose, debug=args.debug, loglevel=1)
                    figure.savefig(image_dirname + "/" + sequence_name + "/" + sequence_name + "_primordia_signals.png")
                    # for primordium in signal_image_slices.values()[0].keys():
                    #     figure = signal_image_primordium_plot(signal_image_slices, r_max=80, primordium=primordium, verbose=args.verbose, debug=args.debug, loglevel=1)
                    #     if figure is not None:
                    #         figure.savefig(image_dirname + "/" + sequence_name + "/" + sequence_name + "_P" + str(primordium) + "_signals.png")

            if 'experiment_aligned' in args.nuclei_plot:
                experiment_data = sequence_aligned_signal_data[exp].values()
                experiment_times = np.sort(np.unique(np.concatenate([[int(f[-2:]) for f in d.keys()] for d in  experiment_data])))
                experiment_aligned_signal_data = dict([("t"+str(time).zfill(2),pd.concat([d[f] for d in experiment_data for f in d.keys() if int(f[-2:])==time])) for time in experiment_times])
                figure = signal_nuclei_plot(experiment_aligned_signal_data, aligned=True, normalized=args.normalized, alpha=1./len(sequence_aligned_signal_data[exp]), verbose=args.verbose, debug=args.debug, loglevel=1)
                figure.savefig(image_dirname + "/" + exp + "_L1_aligned_nuclei_signals.png")

            if 'experiment_primordia' in args.nuclei_plot:
                experiment_data = sequence_primordia_signal_data[exp].values()
                experiment_primordia = np.sort(np.unique([int(p) for d in experiment_data for p in d.keys() ]))
                experiment_times = np.sort(np.unique([int(f[-2:]) for d in experiment_data for p in d.values() for f in p.keys() ]))
                experiment_primordium_data = dict([(p,dict([(f,v) for d in experiment_data for f,v in d[p].items()])) for p in experiment_primordia])
                experiment_primordia_files = dict([(p,dict([(time,[f for f in experiment_primordium_data[p].keys() if int(f[-2:]) == time]) for time in experiment_times])) for p in experiment_primordia])
                experiment_primordia_data = dict([(p,dict([("t" + str(time).zfill(2), pd.concat([experiment_primordium_data[p][f] for f in experiment_primordia_files[p][time]])) for time in experiment_times if len(experiment_primordia_files[p][time])>0])) for p in experiment_primordia])
                figure = signal_nuclei_all_primordia_plot(experiment_primordia_data, normalized=args.normalized, alpha=1./len(sequence_primordia_signal_data[exp]), verbose=args.verbose, debug=args.debug, loglevel=1)
                figure.savefig(image_dirname + "/" + exp + "_primordia_nuclei_signals.png")

            if 'experiment_aligned' in args.map_plot:
                logging.info("--> Plotting average signal maps "+exp)
                experiment_signal_maps = dict([(f,m) for s in sequence_aligned_signal_maps[exp].keys() for f,m in sequence_aligned_signal_maps[exp][s].items()])
                time_average_maps = compute_average_signal_maps(experiment_signal_maps, verbose=args.verbose, debug=args.debug, loglevel=1)
                figure = signal_map_plot(time_average_maps, aligned=True, verbose=args.verbose, debug=args.debug, loglevel=1)
                figure.savefig(image_dirname + "/" + exp + "_average_L1_aligned_signal_maps.png")

            if 'experiment_primordia' in args.map_plot:
                logging.info("--> Plotting average primordia signal maps "+exp)
                experiment_primordia_signal_maps = dict([(f,m) for s in sequence_primordia_signal_maps[exp].keys() for f,m in sequence_primordia_signal_maps[exp][s].items()])
                time_primordia_average_maps = compute_average_primordia_signal_maps(experiment_primordia_signal_maps, verbose=args.verbose, debug=args.debug, loglevel=1)
                figure = signal_map_all_primordia_plot(time_primordia_average_maps, verbose=args.verbose, debug=args.debug, loglevel=1)
                figure.savefig(image_dirname + "/" + exp + "_average_primordia_signal_maps.png")

        if 'all_aligned' in args.nuclei_plot:
            all_experiments = [exp for exp in sequence_aligned_signal_data.keys() if len(sequence_aligned_signal_data[exp])>0]
            experiment_string = "".join([all_experiments[0]] + ["_" + exp for exp in all_experiments[1:]]) if len(all_experiments) > 1 else all_experiments[0]
            all_data = np.concatenate([sequence_aligned_signal_data[exp].values() for exp in all_experiments])
            all_times = np.sort(np.unique(np.concatenate([[int(f[-2:]) for f in d.keys()] for d in all_data])))
            all_aligned_signal_data = dict([("t"+str(time).zfill(2),pd.concat([d[f] for d in all_data for f in d.keys() if int(f[-2:])==time])) for time in all_times])
            figure = signal_nuclei_plot(all_aligned_signal_data, aligned=True, normalized=args.normalized, alpha=1./len(all_data), verbose=args.verbose, debug=args.debug, loglevel=1)
            figure.savefig(image_dirname + "/" + experiment_string + "_L1_aligned_nuclei_signals.png")
        
        if 'all_primordia' in args.nuclei_plot:
            all_experiments = [exp for exp in sequence_primordia_signal_data.keys() if len(sequence_primordia_signal_data[exp])>0]
            experiment_string = "".join([all_experiments[0]] + ["_" + exp for exp in all_experiments[1:]]) if len(all_experiments)>1 else all_experiments[0]
            all_data = np.concatenate([sequence_primordia_signal_data[exp].values() for exp in all_experiments])
            all_primordia = np.sort(np.unique([int(p) for d in all_data for p in d.keys()]))
            all_times = np.sort(np.unique([int(f[-2:]) for d in all_data for p in d.values() for f in p.keys()]))
            all_primordium_data = dict([(p, dict([(f, v) for d in all_data for f, v in d[p].items()])) for p in all_primordia])
            all_primordia_files = dict([(p, dict([(time, [f for f in all_primordium_data[p].keys() if int(f[-2:]) == time]) for time in all_times])) for p in all_primordia])
            all_primordia_data = dict([(p, dict([("t" + str(time).zfill(2), pd.concat([all_primordium_data[p][f] for f in all_primordia_files[p][time]])) for time in all_times if len(all_primordia_files[p][time]) > 0])) for p in all_primordia])
            figure = signal_nuclei_all_primordia_plot(all_primordia_data, normalized=args.normalized, alpha=1./len(all_data), verbose=args.verbose, debug=args.debug, loglevel=1)
            figure.savefig(image_dirname + "/" + experiment_string + "_primordia_nuclei_signals.png")

        if 'all_aligned' in args.map_plot:
            all_experiments = [exp for exp in sequence_aligned_signal_maps.keys() if len(sequence_aligned_signal_maps[exp])>0]
            experiment_string = "".join([all_experiments[0]] + ["_" + exp for exp in all_experiments[1:]]) if len(all_experiments) > 1 else all_experiments[0]
            all_signal_maps = dict([(f, m) for exp in all_experiments for s in sequence_aligned_signal_maps[exp].keys() for f, m in sequence_aligned_signal_maps[exp][s].items()])
            time_average_maps = compute_average_signal_maps(all_signal_maps, verbose=args.verbose, debug=args.debug, loglevel=1)
            figure = signal_map_plot(time_average_maps, aligned=True, verbose=args.verbose, debug=args.debug, loglevel=1)
            figure.savefig(image_dirname + "/" + experiment_string + "_average_L1_aligned_signal_maps.png")

        if 'all_primordia' in args.map_plot:
            all_experiments = [exp for exp in sequence_primordia_signal_maps.keys() if len(sequence_primordia_signal_maps[exp]) > 0]
            experiment_string = "".join([all_experiments[0]] + ["_" + exp for exp in all_experiments[1:]]) if len(all_experiments) > 1 else all_experiments[0]
            all_primordia_signal_maps = dict([(f,m) for exp in all_experiments for s in sequence_primordia_signal_maps[exp].keys() for f,m in sequence_primordia_signal_maps[exp][s].items()])
            time_primordia_average_maps = compute_average_primordia_signal_maps(all_primordia_signal_maps, verbose=args.verbose, debug=args.debug, loglevel=1)
            figure = signal_map_all_primordia_plot(time_primordia_average_maps, verbose=args.verbose, debug=args.debug, loglevel=1)
            figure.savefig(image_dirname + "/" + experiment_string + "_average_primordia_signal_maps.png")

        if args.data_compilation:
            logging.info("--> Compiling signal data from all experiments "+str(experiments))
            compile_signal_data(experiments,save_files=True, image_dirname=image_dirname, data_dirname=data_dirname, aligned=True, verbose=args.verbose, debug=args.debug, loglevel=1)
            compile_primordia_data(experiments,save_files=True, image_dirname=image_dirname, data_dirname=data_dirname, verbose=args.verbose, debug=args.debug, loglevel=1)


if __name__ == "__main__":
    main()