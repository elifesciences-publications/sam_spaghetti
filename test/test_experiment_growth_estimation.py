import unittest
import logging
import os

import numpy as np
import pandas as pd

from timagetk.io import imsave, save_trsf

import sam_spaghetti
from sam_spaghetti.sam_sequence_loading import load_sequence_signal_images, load_sequence_signal_data, load_sequence_rigid_transformations, load_sequence_vectorfield_transformations
from sam_spaghetti.sam_sequence_info import get_experiment_name, get_experiment_microscopy, get_nomenclature_name, get_experiment_channels, get_experiment_reference, get_experiment_microscope_orientation
from sam_spaghetti.signal_data_compilation import compile_signal_data
from sam_spaghetti.sequence_growth_estimation import compute_growth

dirname = os.path.abspath(sam_spaghetti.__path__[0] + "/../../share/data")
test_dirname = os.path.abspath(sam_spaghetti.__path__[0] + "/../../test")

max_sam_id = 100
max_time = 100


class TestExperimentRegistration(unittest.TestCase):

    def setUp(self):
        self.data_dirname = dirname

        self.sequence_names = {}

        self.experiments = ['SAM-TEST']
        self.test_image_dirname = test_dirname + "/nuclei_images"
        self.image_dirname = self.data_dirname + "/nuclei_images"

        if not os.path.exists(self.test_image_dirname):
            os.makedirs(self.test_image_dirname)

        for exp in self.experiments:
            self.sequence_names[exp] = []
            experiment_name = get_experiment_name(exp, self.data_dirname)

            for sam_id in range(max_sam_id):
                sequence_name = experiment_name + "_sam" + str(sam_id).zfill(2)
                signal_data = load_sequence_signal_data(sequence_name,
                                                        self.image_dirname,
                                                        normalized=False,
                                                        aligned=False,
                                                        verbose=True, loglevel=1)

                normalized_signal_data = load_sequence_signal_data(sequence_name,
                                                                   self.image_dirname,
                                                                   normalized=False,
                                                                   aligned=False,
                                                                   verbose=True, loglevel=1)
                if len(signal_data) > 0:
                    self.sequence_names[exp] += [sequence_name]

                    if not os.path.exists(self.test_image_dirname + "/" + sequence_name):
                        os.makedirs(self.test_image_dirname + "/" + sequence_name)

                    for filename in signal_data.keys():
                        file_dirname = self.test_image_dirname + "/" + sequence_name + "/" + filename
                        if not os.path.exists(file_dirname):
                            os.makedirs(file_dirname)
                        signal_data[filename].to_csv(file_dirname+"/"+filename+"_signal_data.csv")
                        normalized_signal_data[filename].to_csv(file_dirname+"/"+filename+"_normalized_signal_data.csv")

                    sequence_rigid_transforms = load_sequence_rigid_transformations(sequence_name,
                                                                                    self.image_dirname,
                                                                                    verbose=True, loglevel=1)

                    sequence_vectorfield_transforms = load_sequence_vectorfield_transformations(sequence_name,
                                                                                                self.image_dirname,
                                                                                                verbose=True, loglevel=1)

                    for filename, registered_filename in sequence_rigid_transforms.keys():
                        file_dirname = self.test_image_dirname + "/" + sequence_name + "/" + filename
                        transform_file = file_dirname + "/" + filename + "_to_" + registered_filename[-3:] + "_rigid_transform.csv"
                        np.savetxt(transform_file, sequence_rigid_transforms[(filename, registered_filename)], delimiter=";")

                    for filename, registered_filename in sequence_vectorfield_transforms.keys():
                        file_dirname = self.test_image_dirname + "/" + sequence_name + "/" + filename
                        vector_field_file = file_dirname + "/" + filename+"_to_" + registered_filename[-3:] + "_vector_field.inr.gz"
                        save_trsf(sequence_vectorfield_transforms[(filename, registered_filename)], vector_field_file)

    def tearDown(self):
        for exp in self.experiments:
            for sequence_name in self.sequence_names[exp]:
                if os.path.exists(self.test_image_dirname + "/" + sequence_name):
                    for filename in os.listdir(self.test_image_dirname + "/" + sequence_name):
                        file_dirname = self.test_image_dirname + "/" + sequence_name + "/" + filename
                        if os.path.isdir(file_dirname):
                            for filename in os.listdir(file_dirname):
                                os.remove(file_dirname + "/" + filename)
                            os.removedirs(file_dirname)
                        else:
                            os.remove(self.test_image_dirname + "/" + sequence_name + "/" + filename)

                    if os.path.exists(self.test_image_dirname + "/" + sequence_name):
                        os.removedirs(self.test_image_dirname + "/" + sequence_name)

        if os.path.exists(self.test_image_dirname):
            os.removedirs(self.test_image_dirname)

    def test_growth_estimation(self):
        for exp in self.experiments:
            experiment_name = get_experiment_name(exp, self.data_dirname)

            compile_signal_data(self.experiments,
                                save_files=True,
                                image_dirname=self.test_image_dirname,
                                data_dirname=self.data_dirname,
                                verbose=True, loglevel=1)

            for exp in self.experiments:
                microscope_orientation = get_experiment_microscope_orientation(exp, self.data_dirname)
                for sequence_name in self.sequence_names[exp]:
                    logging.info("--> Computing sequence surfacic growth " + sequence_name)
                    compute_growth(sequence_name,
                                   image_dirname=self.test_image_dirname,
                                   save_files=True,
                                   growth_type='surfacic',
                                   microscope_orientation=microscope_orientation,
                                   verbose=True, loglevel=1)
