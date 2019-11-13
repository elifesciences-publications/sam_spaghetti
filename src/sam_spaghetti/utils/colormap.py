# -*- coding: utf-8 -*-
# -*- python -*-
#
#       TissueLab
#
#       Copyright 2015 INRIA - CIRAD - INRA
#
#       File author(s): Guillaume Cerutti <guillaume.cerutti@inria.fr>
#
#       Distributed under the Cecill-C License.
#       See accompanying file LICENSE.txt or copy at
#           http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html
#
#       TissueLab Website : http://virtualplants.github.io/
#
###############################################################################

import numpy as np
import os


class Colormap (object):

    def __init__(self, name=None):
        self._color_points = {}
        self._color_map = None
        self.name = name

    def __getitem__(self, key):
        if key == 'name':
            return self.name
        elif key == 'color_points':
            return self._color_points
        else:
            raise NameError("'Colormap' object has no attribute "+key)

    def get_color(self, value):
        from scipy.interpolate import splrep, splev
        if len(self._color_points) == 0:
            return (0.0, 0.0, 0.0)
        elif value <= np.min(list(self._color_points.keys())):
            return self._color_points[np.min(list(self._color_points.keys()))]
        elif value >= np.max(list(self._color_points.keys())):
            return self._color_points[np.max(list(self._color_points.keys()))]
        else:
            return tuple([float(splev(value, self._color_map[channel], der=0)) for channel in [0, 1, 2]])

    def get_values(self):
        return list(np.sort(list(self._color_points.keys())))

    def __call__(self, value):
        return self.get_color(value)

    def add_rgb_point(self, value, color):
        assert (value >= 0.0) & (value <= 1.0)
        if np.max(color) > 1.0:
            color = tuple(np.array(color) / 255.)
        self._color_points[value] = tuple(color)
        self._compute()

    def _compute(self):
        from scipy.interpolate import splrep, splev
        if len(self._color_points) > 1:
            self._color_map = [splrep(np.sort(list(self._color_points.keys())), np.array(list(self._color_points.values()))[
                                      np.argsort(list(self._color_points.keys())), channel], s=0, k=1) for channel in [0, 1, 2]]

    def __str__(self):
        # return self._color_points.__str__()
        sorted_values = np.sort(list(self._color_points.keys()))

        cmap_string = "{"
        for i, value in enumerate(sorted_values):
            cmap_string = cmap_string + \
                str(value) + " : " + str(self._color_points[value])
            if i < sorted_values.size - 1:
                cmap_string = cmap_string + ",\n"
        cmap_string = cmap_string + "}\n"
        return cmap_string


def colormap_from_file(filename, name=None, delimiter=' '):
    import csv

    colormap = Colormap(name=name)

    colormap_file = open(filename, "rU")
    colormap_reader = csv.reader(colormap_file, delimiter=delimiter)
    colormap_data = []
    for color_line in colormap_reader:
        if color_line[0].rfind('Index') == -1:
            colormap_data.append(np.array(color_line).astype(float))
    colormap_data = np.array(colormap_data)
    colormap_file.close()

    for c, color in enumerate(colormap_data):
        if colormap_data.shape[1] == 4:
            if colormap_data[:, 0].max() > 1:
                colormap.add_rgb_point(color[0] / 255., color[1:])
            else:
                colormap.add_rgb_point(color[0], color[1:])
        else:
            colormap.add_rgb_point(
                float(c) / (colormap_data.shape[0] - 1), color)
    return colormap
