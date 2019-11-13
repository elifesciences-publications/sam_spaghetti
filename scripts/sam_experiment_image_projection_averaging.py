import numpy as np
import pandas as pd

import sam_spaghetti
from sam_spaghetti.sam_sequence_info import get_experiment_name
from sam_spaghetti.sam_sequence_loading import load_sequence_signal_images, load_sequence_signal_image_slices, load_sequence_signal_data
from sam_spaghetti.signal_image_plot import signal_image_plot

import logging
import argparse
import os

guillaume_dirname = "/Users/gcerutti/Desktop/WorkVP/SamMaps"
calculus_dirname = "/projects/SamMaps/"
sam_spaghetti_dirname = sam_spaghetti.__path__[0]+"/../../share/data"

dirname = guillaume_dirname
# dirname = calculus_dirname
# dirname = sam_spaghetti_dirname

max_sam_id = 100
# max_time = 100
max_time = 12

experiment_excluded_sams = {}
experiment_excluded_sams['E25'] = [2]
experiment_excluded_sams['E27'] = [2,7,10]
# experiment_excluded_sams['E27'] = []
experiment_excluded_sams['E31'] = [4]
experiment_excluded_sams['E33'] = []

def main():
    """

    Returns:

    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiments', help='List of experiment identifiers', nargs='+', required=True)
    parser.add_argument('-dir', '--data-directory', help='Path to SAM sequence data files directory (nomenclature, orientation...)', default=dirname)
    parser.add_argument('-N', '--nuclei-directory', help='Path to detected nuclei directory [default : data_directory/nuclei_images]', default=None)
    parser.add_argument('-p', '--projection-type', default='max_intensity', help='Projection type for the image plots [\'max_intensity\', \'L1_slice\']',choices=['max_intensity', 'L1_slice'])
    parser.add_argument('-a', '--aligned', default=False, action='store_true', help='Whether to use raw or aligned image projections')
    parser.add_argument('-t', '--time-averaging', default=False, action='store_true', help='Compute one average image per acquisition time')
    
    parser.add_argument('-v', '--verbose', default=False, action='store_true', help='Verbose')
    parser.add_argument('-d', '--debug', default=False, action='store_true', help='Debug')


    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO if args.verbose else logging.DEBUG if args.debug else logging.ERROR)

    # mesh_to_cvt_image(input=args.input, output=args.output, method=args.method, verbose=args.verbose, debug=args.debug,
    #     save=not(args.no_save), voxelsize=args.voxelsize, nbcells=args.nbcells, max_step=args.step, res=args.resolution)
    data_dirname = args.data_directory

    experiments = args.experiments
    image_dirname = args.nuclei_directory if args.nuclei_directory is not None else data_dirname+"/nuclei_images"

    if not os.path.exists(image_dirname):
        logging.error("Result output directory not found, nothing left to do!")
    else:
        sequence_signal_data = {}
        for exp in experiments:
            experiment_name = get_experiment_name(exp,data_dirname)

            sequence_signal_data[exp] = {}
            for sam_id in range(max_sam_id):
                if not sam_id in experiment_excluded_sams[exp]:
                    sequence_name = experiment_name+"_sam"+str(sam_id).zfill(2)
                    signal_data = load_sequence_signal_data(sequence_name, image_dirname, normalized=False, aligned=False, verbose=args.verbose, debug=args.debug, loglevel=1)
                    if len(signal_data)>0:
                        sequence_signal_data[exp][sequence_name] = signal_data

        signal_image_slices = {}
        for exp in experiments:
            for sequence_name in sequence_signal_data[exp]:
                sequence_signal_image_slices = load_sequence_signal_image_slices(sequence_name, image_dirname, projection_type=args.projection_type, aligned=args.aligned, verbose=args.verbose, debug=args.debug, loglevel=1)
                for signal_name in sequence_signal_image_slices:
                    if not signal_name in signal_image_slices:
                        signal_image_slices[signal_name] = {}
                    signal_image_slices[signal_name].update(sequence_signal_image_slices[signal_name])

        experiment_string = "".join([experiments[0]]+["_"+exp for exp in experiments[1:]])
        average_signal_image_slices = {}
        for signal_name in signal_image_slices:
            if args.time_averaging:
                average_signal_image_slices[signal_name] = {}
                for time in range(max_time):
                    time_filenames = [f for f in signal_image_slices[signal_name].keys() if "_t"+str(time).zfill(2) in f]
                    if len(time_filenames)>0:
                        average_signal_image_slices[signal_name][experiment_string+"_t"+str(time).zfill(2)] = np.mean([signal_image_slices[signal_name][f] for f in time_filenames],axis=0)
            else:
                average_signal_image_slices[signal_name] = {experiment_string+"_t00":np.mean(signal_image_slices[signal_name].values(),axis=0)}
        figure = signal_image_plot(average_signal_image_slices, projection_type=args.projection_type, resolution=0.25, aligned=args.aligned, verbose=args.verbose, debug=args.debug, loglevel=1)
        figure.savefig(data_dirname+"/"+experiment_string+"_"+args.projection_type+("_time_" if args.time_averaging else "_")+"average"+("_aligned_" if args.aligned else "_")+"signals.png")


if __name__ == "__main__":
    main()




