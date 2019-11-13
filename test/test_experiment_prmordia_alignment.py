import unittest
import logging
import os

import numpy as np
import pandas as pd

from timagetk.io import imsave

import sam_spaghetti
from sam_spaghetti.sam_sequence_loading import load_sequence_signal_images, load_sequence_signal_data
from sam_spaghetti.sam_sequence_info import get_experiment_name, get_experiment_microscopy, get_nomenclature_name, get_experiment_channels, get_experiment_reference, get_experiment_microscope_orientation, get_sequence_orientation
from sam_spaghetti.sam_sequence_primordia_alignment import align_sam_sequence, detect_organ_primordia

dirname = os.path.abspath(sam_spaghetti.__path__[0] + "/../../share/data")
test_dirname = os.path.abspath(sam_spaghetti.__path__[0] + "/../../test")

max_sam_id = 100
max_time = 100


class TestExperimentPrimordiaAlignment(unittest.TestCase):

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
                if len(signal_data) > 0:
                    self.sequence_names[exp] += [sequence_name]

                    if not os.path.exists(self.test_image_dirname + "/" + sequence_name):
                        os.makedirs(self.test_image_dirname + "/" + sequence_name)

                    for filename in signal_data.keys():
                        file_dirname = self.test_image_dirname + "/" + sequence_name + "/" + filename
                        if not os.path.exists(file_dirname):
                            os.makedirs(file_dirname)
                        signal_data[filename].to_csv(file_dirname+"/"+filename+"_signal_data.csv")

                    channel_names = get_experiment_channels(exp, self.data_dirname)
                    sequence_dict = load_sequence_signal_images(sequence_name,
                                                                image_dirname=self.image_dirname,
                                                                signal_names=channel_names,
                                                                verbose=True, loglevel=1)
                    for channel_name in sequence_dict.keys():
                        for filename in sequence_dict[channel_name].keys():
                            file_dirname = self.test_image_dirname + "/" + sequence_name + "/" + filename
                            imsave(file_dirname+"/"+filename+"_"+channel_name+".inr.gz",sequence_dict[channel_name][filename])

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

    def test_primordia_alignment(self):

        sequence_aligned_signal_data = {}
        sequence_primordia_signal_data = {}
        sequence_aligned_signal_maps = {}
        sequence_primordia_signal_maps = {}
        for exp in self.experiments:
            sequence_aligned_signal_data[exp] = {}
            sequence_primordia_signal_data[exp] = {}
            sequence_aligned_signal_maps[exp] = {}
            sequence_primordia_signal_maps[exp] = {}
            for sequence_name in self.sequence_names[exp]:
                logging.info("--> Sequence primordia alignment " + sequence_name)
                sam_orientation = get_sequence_orientation(sequence_name, self.data_dirname)

                align_sam_sequence(sequence_name,
                                   image_dirname = self.image_dirname,
                                   sam_orientation=sam_orientation,
                                   save_files=True,
                                   verbose=True, loglevel=1)

                detect_organ_primordia(sequence_name,
                                       image_dirname = self.image_dirname,
                                       sam_orientation=sam_orientation,
                                       save_files=True,
                                       verbose=True, loglevel=1)

