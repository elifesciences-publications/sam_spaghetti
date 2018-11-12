========================
sam_spaghetti
========================

.. {# pkglts, doc

.. #}

SAM Sequence Primordia Alignment, GrowtH Estimation, Tracking & Temporal Indexation

Two scripts are provided in `scripts/` to perform complex computational analysis on CZI images of Shoot Apical Meristems (SAMs)

## Detection, quantification and alignment from CZI files (with optional visualization)

usage: sam_experiment_detect_quantify_and_align.py [-h] -e EXPERIMENTS
                                                   [EXPERIMENTS ...]
                                                   [-dir DATA_DIRECTORY]
                                                   [-M MICROSCOPY_DIRECTORY]
                                                   [-N NUCLEI_DIRECTORY] [-D]
                                                   [-s] [-R]
                                                   [-i {sequence_raw,sequence_aligned} [{sequence_raw,sequence_aligned} ...]]
                                                   [-p {max_intensity,L1_slice}]
                                                   [-n {sequence_raw,sequence_aligned} [{sequence_raw,sequence_aligned} ...]]
                                                   [-m {sequence_raw,sequence_aligned} [{sequence_raw,sequence_aligned} ...]]
                                                   [-G] [-P] [-C] [-v] [-d]

optional arguments:
  -h, --help            show this help message and exit
  -e EXPERIMENTS [EXPERIMENTS ...], --experiments EXPERIMENTS [EXPERIMENTS ...]
                        List of experiment identifiers
  -dir DATA_DIRECTORY, --data-directory DATA_DIRECTORY
                        Path to SAM sequence data files directory
                        (nomenclature, orientation...)
  -M MICROSCOPY_DIRECTORY, --microscopy-directory MICROSCOPY_DIRECTORY
                        Path to CZI image directory [default :
                        data_directory/microscopy]
  -N NUCLEI_DIRECTORY, --nuclei-directory NUCLEI_DIRECTORY
                        Path to detected nuclei directory [default :
                        data_directory/nuclei_images]
  -D, --detection       Run nuclei detection on all experiments
  -s, --save-channels   Save INR image files for each microscopy image channel
  -R, --registration    Run sequence image registration on all experiments
  -i {sequence_raw,sequence_aligned} [{sequence_raw,sequence_aligned} ...], --image-plot {sequence_raw,sequence_aligned} [{sequence_raw,sequence_aligned} ...]
                        List of image projections types to plot
                        ['sequence_raw', 'sequence_aligned']
  -p {max_intensity,L1_slice}, --projection-type {max_intensity,L1_slice}
                        Projection type for the image plots ['max_intensity',
                        'L1_slice']
  -n {sequence_raw,sequence_aligned} [{sequence_raw,sequence_aligned} ...], --nuclei-plot {sequence_raw,sequence_aligned} [{sequence_raw,sequence_aligned} ...]
                        List of signal map types to plot ['sequence_raw',
                        'sequence_aligned']
  -m {sequence_raw,sequence_aligned} [{sequence_raw,sequence_aligned} ...], --map-plot {sequence_raw,sequence_aligned} [{sequence_raw,sequence_aligned} ...]
                        List of signal map types to plot ['sequence_raw',
                        'sequence_aligned']
  -G, --growth-estimation
                        Estimate surfacic growth on all experiments
  -P, --primordia-alignment
                        Align sequences of all experiments based on the
                        detection of CZ and P0
  -C, --data-compilation
                        Compile all the data from the experiments into .csv
                        files in the data directory
  -v, --verbose         Verbose
  -d, --debug           Debug
  
  ## Image averaging 
  
  usage: sam_experiment_image_projection_averaging.py [-h] -e EXPERIMENTS
                                                    [EXPERIMENTS ...]
                                                    [-dir DATA_DIRECTORY]
                                                    [-N NUCLEI_DIRECTORY]
                                                    [-p {max_intensity,L1_slice}]
                                                    [-a] [-t] [-v] [-d]

optional arguments:
  -h, --help            show this help message and exit
  -e EXPERIMENTS [EXPERIMENTS ...], --experiments EXPERIMENTS [EXPERIMENTS ...]
                        List of experiment identifiers
  -dir DATA_DIRECTORY, --data-directory DATA_DIRECTORY
                        Path to SAM sequence data files directory
                        (nomenclature, orientation...)
  -N NUCLEI_DIRECTORY, --nuclei-directory NUCLEI_DIRECTORY
                        Path to detected nuclei directory [default :
                        data_directory/nuclei_images]
  -p {max_intensity,L1_slice}, --projection-type {max_intensity,L1_slice}
                        Projection type for the image plots ['max_intensity',
                        'L1_slice']
  -a, --aligned         Whether to use raw or aligned image projections
  -t, --time-averaging  Compute one average image per acquisition time
  -v, --verbose         Verbose
  -d, --debug           Debug
