import unittest
import logging
import os

import numpy as np
import pandas as pd

import sam_spaghetti
from sam_spaghetti.sam_microscopy_loading import load_image_from_microscopy
from sam_spaghetti.sam_sequence_loading import load_sequence_signal_images
from sam_spaghetti.sam_sequence_info import get_experiment_name, get_experiment_microscopy, get_nomenclature_name, get_experiment_channels, get_experiment_reference, get_experiment_microscope_orientation
from sam_spaghetti.detection_quantification import detect_and_quantify


dirname = sam_spaghetti.__path__[0] + "/../../share/data"
test_dirname = sam_spaghetti.__path__[0] + "/../../test"

max_sam_id = 100
max_time = 100


class TestExperimentDetection(unittest.TestCase):


    def setUp(self):
        self.data_dirname = dirname
        self.microscopy_dirname = self.data_dirname + "/microscopy"

        self.experiments = ['SAM-TEST']
        self.test_image_dirname = test_dirname + "/nuclei_images"
        self.image_dirname = self.data_dirname + "/nuclei_images"

        for exp in self.experiments:
            experiment_dirname = self.microscopy_dirname + "/" + get_experiment_microscopy(exp, self.data_dirname)
            if os.path.exists(experiment_dirname + "/RAW"):
                experiment_dirname += "/RAW"

            self.microscopy_filenames = [experiment_dirname + "/" + f for f in os.listdir(experiment_dirname) if np.any([ext in f for ext in ['.czi', '.lsm']])]

            if not os.path.exists(self.test_image_dirname):
                os.makedirs(self.test_image_dirname)

            for microscopy_filename in self.microscopy_filenames:
                nomenclature_name = get_nomenclature_name(microscopy_filename, self.data_dirname)

                if nomenclature_name is not None:
                    sequence_name = nomenclature_name[:-4]

                    if not os.path.exists(self.test_image_dirname + "/" + sequence_name):
                        os.makedirs(self.test_image_dirname + "/" + sequence_name)

                    file_dirname = self.test_image_dirname + "/" + sequence_name + "/" + nomenclature_name
                    if not os.path.exists(file_dirname):
                        os.makedirs(file_dirname)

    def tearDown(self):
        for exp in self.experiments:
            for microscopy_filename in self.microscopy_filenames:
                nomenclature_name = get_nomenclature_name(microscopy_filename, self.data_dirname)
                if nomenclature_name is not None:
                    sequence_name = nomenclature_name[:-4]

                    file_dirname = self.test_image_dirname + "/" + sequence_name + "/" + nomenclature_name
                    if os.path.exists(file_dirname):
                        for filename in os.listdir(file_dirname):
                            os.remove(file_dirname+"/"+filename)
                        os.removedirs(file_dirname)

            for microscopy_filename in self.microscopy_filenames:
                nomenclature_name = get_nomenclature_name(microscopy_filename, self.data_dirname)
                if nomenclature_name is not None:
                    sequence_name = nomenclature_name[:-4]
                    if os.path.exists(self.test_image_dirname + "/" + sequence_name):
                        for filename in os.listdir(self.test_image_dirname + "/" + sequence_name):
                            os.remove(self.test_image_dirname + "/" + sequence_name + "/" + filename)
                        os.removedirs(self.test_image_dirname + "/" + sequence_name)

        if os.path.exists(self.test_image_dirname):
            os.removedirs(self.test_image_dirname)

    def test_read_microscopy(self):
        for exp in self.experiments:
            channel_names = get_experiment_channels(exp, self.data_dirname)

            for microscopy_filename in self.microscopy_filenames:
                nomenclature_name = get_nomenclature_name(microscopy_filename, self.data_dirname)

                if nomenclature_name is not None:
                    sequence_name = nomenclature_name[:-4]

                    img_dict = load_image_from_microscopy(microscopy_filename,
                                                          save_images=True,
                                                          image_dirname=self.test_image_dirname,
                                                          nomenclature_name=nomenclature_name,
                                                          channel_names=channel_names,
                                                          verbose=True,
                                                          loglevel=1)

                    file_dirname = self.test_image_dirname + "/" + sequence_name + "/" + nomenclature_name
                    assert np.all([os.path.exists(file_dirname + "/" + nomenclature_name + "_" + c + ".inr.gz") for c in channel_names])

    def test_detection(self):
        for exp in self.experiments:
            channel_names = get_experiment_channels(exp, self.data_dirname)
            reference_name = get_experiment_reference(exp, self.data_dirname)

            for microscopy_filename in self.microscopy_filenames:
                nomenclature_name = get_nomenclature_name(microscopy_filename, self.data_dirname)

                if nomenclature_name is not None:
                    sequence_name = nomenclature_name[:-4]

                    sequence_dict = load_sequence_signal_images(sequence_name,
                                                                image_dirname=self.image_dirname,
                                                                signal_names=channel_names,
                                                                verbose=True, loglevel=1)

                    img_dict = dict(zip(channel_names,[sequence_dict[c][nomenclature_name] for c in channel_names]))

                    detect_and_quantify(img_dict,
                                        reference_name=reference_name,
                                        signal_names=channel_names,
                                        image_dirname=self.test_image_dirname,
                                        nomenclature_name=nomenclature_name,
                                        verbose=True, loglevel=1)

                    file_dirname = self.test_image_dirname + "/" + sequence_name + "/" + nomenclature_name
                    assert(os.path.exists(file_dirname+"/"+nomenclature_name+"_signal_data.csv"))

                    signal_df = pd.read_csv(file_dirname+"/"+nomenclature_name+"_signal_data.csv")
                    assert len(signal_df)>0
                    assert np.all([c in signal_df.columns for c in channel_names])
