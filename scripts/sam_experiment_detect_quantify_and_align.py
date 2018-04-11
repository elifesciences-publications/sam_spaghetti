import numpy as np
import pandas as pd

import sam_spaghetti
from sam_spaghetti.sam_sequence_info import get_experiment_name, get_experiment_microscopy, get_nomenclature_name
from sam_spaghetti.detection_quantification import detect_from_czi
from sam_spaghetti.signal_image_plot import load_sequence_signal_images, load_sequence_signal_data, signal_image_plot, signal_nuclei_plot

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
    parser.add_argument('-I', '--information-directory', help='Path to SAM sequence information files directory (nomenclature, orientation...)', default=dirname)
    parser.add_argument('-M', '--microscopy-directory', help='Path to CZI image directory', default=None)
    parser.add_argument('-N', '--nuclei-directory', help='Path to detected nuclei directory', default=None)
    parser.add_argument('-D', '--detection', default=False, action='store_true', help='Run nuclei detection on all experiments')
    # parser.add_argument('-o', '--output', help='Path to output files directory', default="./output")
    # parser.add_argument('-n', '--nbcells', help='Number of cells', default=100, type=int)
    # parser.add_argument('--step', help='Maximal number of steps for CVT', default=1e9, type=int)
    parser.add_argument('-v', '--verbose', default=False, action='store_true', help='Verbose')
    parser.add_argument('-d', '--debug', default=False, action='store_true', help='Debug')
    # parser.add_argument('-ns', '--no_save', default=False, action='store_true', help='skip saving output image')
    # parser.add_argument('--voxelsize', help='Voxel size', default=[.025, .025, .025], nargs=3, type=float)
    # parser.add_argument('-m', '--method', help='Method for CVT [\'lloyd\', \'mcqueen\']', default='lloyd')
    # parser.add_argument('-r', '--resolution', help='Resolution (in voxels)', default=-1, type=int)


    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO if args.verbose else logging.DEBUG if args.debug else logging.ERROR)

    # mesh_to_cvt_image(input=args.input, output=args.output, method=args.method, verbose=args.verbose, debug=args.debug,
    #     save=not(args.no_save), voxelsize=args.voxelsize, nbcells=args.nbcells, max_step=args.step, res=args.resolution)
    info_dirname =args.information_directory

    microscopy_dirname = args.microscopy_directory if args.microscopy_directory is not None else info_dirname+"/microscopy"
    if not os.path.exists(microscopy_dirname):
        logging.warning("Microscopy directory not found! No detection will be performed.")

    image_dirname = args.nuclei_directory if args.nuclei_directory is not None else info_dirname+"/nuclei_images"

    for exp in args.experiments:

        experiment_name = get_experiment_name(exp,info_dirname)
        if experiment_name == "":
            logging.error("Experiment identifier \""+exp+"\" not recognized (consider adding it to the experiment info file in the information directory)")
        else:
            if microscopy_dirname is not None:
                experiment_dirname = microscopy_dirname+"/"+get_experiment_microscopy(exp,info_dirname)
                if os.path.exists(experiment_dirname+"/RAW"):
                    experiment_dirname += "/RAW"

                if not os.path.exists(experiment_dirname):
                    logging.warning("Microscopy directory not found for "+exp+", no detection will be performed.")
                else:                    
                    czi_filenames = [experiment_dirname+"/"+f for f in os.listdir(experiment_dirname) if '.czi' in f]
                    nomenclature_names = [get_nomenclature_name(czi_filename,info_dirname) for czi_filename in czi_filenames]
                    is_not_detected = dict(zip(nomenclature_names,[not os.path.exists(image_dirname+"/"+filename+"/"+filename+"_signal_data.csv") for filename in nomenclature_names]))

                    # if args.detection or np.any(is_not_detected.values()):
                    if not os.path.exists(image_dirname):
                        os.makedirs(image_dirname)

                    for czi_filename in czi_filenames:
                        nomenclature_name = get_nomenclature_name(czi_filename,info_dirname)

                        if not os.path.exists(image_dirname+"/"+nomenclature_name):
                            os.makedirs(image_dirname+"/"+nomenclature_name)
                        
                        if args.detection or is_not_detected[nomenclature_name]:
                            logging.info("--> Running detection on "+nomenclature_name)
                            detect_from_czi(czi_filename, save_files=True, image_dirname=image_dirname, nomenclature_name=nomenclature_name, verbose=args.verbose, debug=args.debug)
                        else:
                            logging.info("--> Found detection output for "+nomenclature_name)

    for exp in args.experiments:
        experiment_name = get_experiment_name(exp,info_dirname)
        if experiment_name == "":
            logging.error("Experiment identifier \""+exp+"\" not recognized (consider adding it to the experiment info file in the information directory)")
        else:
            if not os.path.exists(image_dirname):
                logging.error("Result output directory not found, nothing left to do!")
            else:
                sequence_signal_data = {}
                for sam_id in xrange(max_sam_id):
                    sequence_name = experiment_name+"_sam"+str(sam_id).zfill(2)
                    signal_data = load_sequence_signal_data(sequence_name, image_dirname, aligned=False, verbose=args.verbose, debug=args.debug)
                    if len(signal_data)>0:
                        sequence_signal_data[sequence_name] = signal_data

                for sequence_name in sequence_signal_data:
                    logging.info("--> Plotting signal images "+sequence_name)
                    signal_data = sequence_signal_data[sequence_name]

                    figure = signal_nuclei_plot(signal_data, aligned=False, verbose=args.verbose, debug=args.debug) 
                    figure.savefig(image_dirname+"/"+sequence_name+"_L1_nuclei_signals.png")                   

                    signal_images = load_sequence_signal_images(sequence_name, image_dirname, verbose=args.verbose, debug=args.debug)

                    figure = signal_image_plot(signal_images, signal_data, projection_type="L1_slice", resolution=0.25, aligned=False, verbose=args.verbose, debug=args.debug)
                    figure.savefig(image_dirname+"/"+sequence_name+"_signals.png")




if __name__ == "__main__":
    main()