import numpy as np
import pandas as pd

import sam_spaghetti
from sam_spaghetti.sam_sequence_info import get_experiment_name, get_experiment_microscopy, get_nomenclature_name, get_sequence_orientation
from sam_spaghetti.detection_quantification import detect_from_czi
from sam_spaghetti.sam_sequence_loading import load_sequence_signal_images, load_sequence_signal_data
from sam_spaghetti.signal_image_plot import signal_image_plot, signal_nuclei_plot, signal_map_plot
from sam_spaghetti.sequence_image_registration import register_sequence_images
from sam_spaghetti.signal_data_compilation import compile_signal_data
from sam_spaghetti.sequence_growth_estimation import compute_surfacic_growth
from sam_spaghetti.sam_sequence_primordia_alignment import align_sam_sequence, detect_organ_primordia

import logging
import argparse
import os

guillaume_dirname = "/Users/gcerutti/Desktop/WorkVP/SamMaps"
calculus_dirname = "/projects/SamMaps/"
sam_spaghetti_dirname = sam_spaghetti.__path__[0]+"/../../share/data"

# dirname = guillaume_dirname
# dirname = calculus_dirname
dirname = sam_spaghetti_dirname

max_sam_id = 100
max_time = 100

def main():
    """

    Returns:

    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiments', help='List of experiment identifiers', nargs='+', required=True)
    parser.add_argument('-dir', '--data-directory', help='Path to SAM sequence data files directory (nomenclature, orientation...)', default=dirname)
    parser.add_argument('-M', '--microscopy-directory', help='Path to CZI image directory [default : data_directory/microscopy]', default=None)
    parser.add_argument('-N', '--nuclei-directory', help='Path to detected nuclei directory [default : data_directory/nuclei_images]', default=None)
    parser.add_argument('-D', '--detection', default=False, action='store_true', help='Run nuclei detection on all experiments')
    parser.add_argument('-s', '--save-channels', default=False, action='store_true', help='Save INR image files for each microscopy image channel')
    parser.add_argument('-R', '--registration', default=False, action='store_true', help='Run sequence image registration on all experiments')
    parser.add_argument('-i', '--image-plot', default=[], nargs='+', help='List of image projections types to plot [\'sequence_raw\', \'sequence_registered\']',choices=['sequence_raw', 'sequence_registered'])
    parser.add_argument('-p', '--projection-type', default='max_intensity', help='Projection type for the image plots [\'max_intensity\', \'L1_slice\']',choices=['max_intensity', 'L1_slice'])
    parser.add_argument('-G', '--growth-estimation', default=False, action='store_true', help='Estimate surfacic growth on all experiments')
    parser.add_argument('-P', '--primordia-alignment', default=False, action='store_true', help='Align sequences of all experiments based on the detection of CZ and P0')
    # parser.add_argument('-o', '--output', help='Path to output files directory', default="./output")
    # parser.add_argument('-n', '--nbcells', help='Number of cells', default=100, type=int)
    # parser.add_argument('--step', help='Maximal number of steps for CVT', default=1e9, type=int)
    # parser.add_argument('-ns', '--no_save', default=False, action='store_true', help='skip saving output image')
    # parser.add_argument('--voxelsize', help='Voxel size', default=[.025, .025, .025], nargs=3, type=float)
    # parser.add_argument('-m', '--method', help='Method for CVT [\'lloyd\', \'mcqueen\']', default='lloyd')
    # parser.add_argument('-r', '--resolution', help='Resolution (in voxels)', default=-1, type=int)
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

            if microscopy_dirname is not None:
                experiment_dirname = microscopy_dirname+"/"+get_experiment_microscopy(exp,data_dirname)
                if os.path.exists(experiment_dirname+"/RAW"):
                    experiment_dirname += "/RAW"

                if not os.path.exists(experiment_dirname):
                    logging.warning("Microscopy directory not found for "+exp+", no detection will be performed.")
                else:                    
                    czi_filenames = [experiment_dirname+"/"+f for f in os.listdir(experiment_dirname) if '.czi' in f]
                    nomenclature_names = [get_nomenclature_name(czi_filename,data_dirname) for czi_filename in czi_filenames]
                    is_not_detected = dict(zip(nomenclature_names,[not os.path.exists(image_dirname+"/"+filename[:-4]+"/"+filename+"/"+filename+"_signal_data.csv") for filename in nomenclature_names]))

                    # if args.detection or np.any(is_not_detected.values()):
                    if not os.path.exists(image_dirname):
                        os.makedirs(image_dirname)

                    for czi_filename in czi_filenames:
                        nomenclature_name = get_nomenclature_name(czi_filename,data_dirname)
                        sequence_name = nomenclature_name[:-4]

                        if not os.path.exists(image_dirname+"/"+sequence_name):
                            os.makedirs(image_dirname+"/"+sequence_name)

                        if not os.path.exists(image_dirname+"/"+sequence_name+"/"+nomenclature_name):
                            os.makedirs(image_dirname+"/"+sequence_name+"/"+nomenclature_name)
                        
                        if args.detection or is_not_detected[nomenclature_name]:
                            logging.info("--> Running detection on "+nomenclature_name)
                            detect_from_czi(czi_filename, save_files=True, save_images=args.save_channels, image_dirname=image_dirname, nomenclature_name=nomenclature_name, verbose=args.verbose, debug=args.debug, loglevel=1)
                        else:
                            logging.info("--> Found detection output for "+nomenclature_name)

    if not os.path.exists(image_dirname):
        logging.error("Result output directory not found, nothing left to do!")
    else:
        sequence_signal_data = {}
        for exp in experiments:
            experiment_name = get_experiment_name(exp,data_dirname)

            sequence_signal_data[exp] = {}
            for sam_id in xrange(max_sam_id):
                sequence_name = experiment_name+"_sam"+str(sam_id).zfill(2)
                signal_data = load_sequence_signal_data(sequence_name, image_dirname, normalized=False, aligned=False, verbose=args.verbose, debug=args.debug, loglevel=1)
                if len(signal_data)>0:
                    sequence_signal_data[exp][sequence_name] = signal_data

        for exp in experiments:
            for sequence_name in sequence_signal_data[exp]:
                if True:
                    signal_data = sequence_signal_data[exp][sequence_name]
                    logging.info("--> Plotting detected nuclei signals "+sequence_name)
                    figure = signal_nuclei_plot(signal_data, verbose=args.verbose, debug=args.debug, loglevel=1) 
                    figure.savefig(image_dirname+"/"+sequence_name+"/"+sequence_name+"_L1_nuclei_signals.png")

                if 'sequence_raw' in args.image_plot:
                    logging.info("--> Plotting signal images "+sequence_name)
                    signal_images = load_sequence_signal_images(sequence_name, image_dirname, verbose=args.verbose, debug=args.debug, loglevel=1)
                    signal_data = sequence_signal_data[exp][sequence_name]
                    figure = signal_image_plot(signal_images, signal_data, projection_type=args.projection_type, resolution=0.25, aligned=False, verbose=args.verbose, debug=args.debug, loglevel=1)
                    figure.savefig(image_dirname+"/"+sequence_name+"/"+sequence_name+"_"+args.projection_type+"_signals.png")

        for exp in experiments:
            for sequence_name in sequence_signal_data[exp]:
                if args.registration:
                    logging.info("--> Sequence image registration "+sequence_name)
                    register_sequence_images(sequence_name, save_files=True, image_dirname=image_dirname, verbose=args.verbose, debug=args.debug, loglevel=1)

        logging.info("--> Compiling signal data from all experiments "+str(experiments))
        compile_signal_data(experiments,save_files=True, image_dirname=image_dirname, data_dirname=data_dirname, verbose=args.verbose, debug=args.debug, loglevel=1)
                            
        for exp in experiments:
            for sequence_name in sequence_signal_data[exp]:
                if args.growth_estimation:
                    logging.info("--> Computing sequence surfacic growth "+sequence_name)
                    compute_surfacic_growth(sequence_name, image_dirname, save_files=True, verbose=args.verbose, debug=args.debug, loglevel=1)

                    signal_normalized_data = load_sequence_signal_data(sequence_name, image_dirname, normalized=True, aligned=False, verbose=args.verbose, debug=args.debug, loglevel=1)  
                    logging.info("--> Plotting nuclei growth "+sequence_name)
                    figure = signal_nuclei_plot(signal_normalized_data, signal_names=['next_relative_surfacic_growth','previous_relative_surfacic_growth'], registered=True, verbose=args.verbose, debug=args.debug, loglevel=1) 
                    figure.savefig(image_dirname+"/"+sequence_name+"/"+sequence_name+"_L1_registered_nuclei_growth.png")  

                if True:
                    signal_normalized_data = load_sequence_signal_data(sequence_name, image_dirname, normalized=True, aligned=False, verbose=args.verbose, debug=args.debug, loglevel=1)  
                    logging.info("--> Plotting maps "+sequence_name)
                    figure = signal_map_plot(signal_normalized_data, verbose=args.verbose, debug=args.debug, loglevel=1) 
                    figure.savefig(image_dirname+"/"+sequence_name+"/"+sequence_name+"_L1_signal_maps.png")  


        for exp in experiments:
            for sequence_name in sequence_signal_data[exp]:
                if args.primordia_alignment:
                    logging.info("--> Sequence primordia alignment "+sequence_name)
                    sam_orientation = get_sequence_orientation(sequence_name,data_dirname)
                    align_sam_sequence(sequence_name, image_dirname, sam_orientation=sam_orientation, save_files=True, verbose=args.verbose, debug=args.debug, loglevel=1)
                    detect_organ_primordia(sequence_name, image_dirname, sam_orientation=sam_orientation, save_files=True, verbose=args.verbose, debug=args.debug, loglevel=1)

                if True:
                    signal_aligned_data = load_sequence_signal_data(sequence_name, image_dirname, normalized=True, aligned=True, verbose=args.verbose, debug=args.debug, loglevel=1)  
                    logging.info("--> Plotting maps "+sequence_name)
                    figure = signal_map_plot(signal_aligned_data, aligned=True, verbose=args.verbose, debug=args.debug, loglevel=1) 
                    figure.savefig(image_dirname+"/"+sequence_name+"/"+sequence_name+"_L1_aligned_signal_maps.png")  

        if True:
            logging.info("--> Compiling signal data from all experiments "+str(experiments))
            compile_signal_data(experiments,save_files=True, image_dirname=image_dirname, data_dirname=data_dirname, aligned=True, verbose=args.verbose, debug=args.debug, loglevel=1)





if __name__ == "__main__":
    main()