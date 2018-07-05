import numpy as np
import scipy.ndimage as nd
import pandas as pd

import matplotlib.pyplot as plt

from vplants.tissue_nukem_3d.epidermal_maps import nuclei_density_function

import warnings

class SignalMap(object):

    def __init__(self, signal_data, extent, origin=np.array([0,0]), resolution=1, position_name='center', radius=7.5, density_k=0.55, polar=False):
        self.signal_data = signal_data
        self.extent = extent
        self.origin = origin
        self.resolution = resolution
        self.position_name = position_name
        self.radius = radius
        self.density_k = density_k
        self.polar = polar

        self.xx = None
        self.yy = None
        self.rr = None
        self.tt = None

        self.potential = None
        self.signal_maps = {}

        self.compute_map_grid()
        self.compute_potential(position_name=self.position_name,radius=self.radius,density_k=self.density_k)


    @property
    def shape(self):
        return self.xx.shape


    def signal_names(self):
        return [s for s in self.signal_maps.keys() if s in self.signal_data.columns]


    def compute_map_grid(self):
        if not self.polar:
            x_range = np.linspace(self.origin[0]-self.extent,self.origin[0]+self.extent,2*self.extent/self.resolution+1)
            y_range = np.linspace(self.origin[1]-self.extent,self.origin[1]+self.extent,2*self.extent/self.resolution+1)

            self.xx, self.yy = np.meshgrid(x_range,y_range)
            self.rr = np.linalg.norm([self.xx,self.yy],axis=0)
            self.tt = np.sign(self.yy)*np.arccos(self.xx/self.rr)
            self.tt[(self.rr==0)] = 0.
            self.tt[(self.yy==0)&(self.xx<0)] = np.pi
        else:
            r_range = np.linspace(0,self.extent,self.extent/self.resolution+1)
            t_range = np.radians(np.linspace(-180.,180.,180./self.resolution+1))

            self.rr, self.tt = np.meshgrid(r_range,t_range)
            self.xx = self.rr*np.cos(self.tt)
            self.yy = self.rr*np.sin(self.tt)


    def compute_potential(self, position_name=None, radius=None, density_k=None):
        if len(self.signal_data):
            if position_name is not None:
                self.position_name = position_name
            if radius is not None:
                self.radius = radius
            if density_k is not None:
                self.density_k = density_k

            X = self.signal_data[self.position_name+'_x'].values
            Y = self.signal_data[self.position_name+'_y'].values
            projected_positions = dict(zip(range(len(X)),np.transpose([X,Y,np.zeros_like(X)])))

            potential = np.array([nuclei_density_function(dict([(p,projected_positions[p])]),cell_radius=self.radius,k=self.density_k)(self.xx,self.yy,np.zeros_like(self.xx)) for p in xrange(len(X))])
            self.potential = np.transpose(potential,(1,2,0))
            # density = np.sum(potential,axis=-1)

            self.confidence_map = np.sum(self.potential,axis=-1)
            if np.all(self.origin==0):
                self.confidence_map += np.maximum(1-np.linalg.norm([self.xx,self.yy],axis=0)/(self.extent/2.),0)
            self.confidence_map = nd.gaussian_filter(self.confidence_map,sigma=1.0/self.resolution)
    

    def signal_map(self, signal_name):
        if not signal_name in self.signal_maps:
            if not signal_name in self.signal_data.columns:
                raise KeyError("The map for \""+signal_name+"\" is not defined!")
            else:
                self.compute_signal_map(signal_name)
        return self.signal_maps[signal_name]


    def compute_signal_map(self, signal_name):
        if not signal_name in self.signal_data.columns:
            raise KeyError("The signal \""+signal_name+"\" is not defined in the data!")
        else:
            signal_values = self.signal_data[signal_name].values
            signal_map = np.sum(self.potential*signal_values[np.newaxis,:],axis=-1)/np.sum(self.potential,axis=-1)
            self.signal_maps[signal_name] = signal_map


    def update_signal_map(self, signal_name, signal_map):
        if not signal_map.shape == self.shape:
            raise IndexError("The map passed as argument for \""+signal_name+"\" has a different shape!")
        else:
            if signal_name in self.signal_maps:
                warnings.warn("The map for \""+signal_name+"\" already exists, it will be owerwritten")
            self.signal_maps[signal_name] = signal_map

    def map_grid(self, polar=False):
        if not polar:
            return self.xx, self.yy
        else:
            return self.rr, self.tt

def plot_signal_map(signal_map, signal_name, figure=None, colormap="Greys_r", signal_range=None, signal_lut_range=None, distance_rings=True):

    if figure is None:
        figure = plt.figure(0)
        figure.clf()
        figure.patch.set_facecolor('w')

    if signal_range is None:
        if signal_name in signal_map.signal_data.columns:
            signal_range = (np.nanmin(signal_map.signal_data[signal_name].values),np.nanmax(signal_map.signal_data[signal_name].values))
        else:
            signal_range = (np.nanmin(signal_map.signal_map(signal_name)),np.nanmax(signal_map.signal_map(signal_name)))
    if signal_lut_range is None:
        signal_lut_range = (np.nanmin(signal_map.signal_map(signal_name)),np.nanmax(signal_map.signal_map(signal_name)))

    figure.gca().contourf(signal_map.xx,signal_map.yy,signal_map.signal_map(signal_name),np.linspace(signal_range[0],signal_range[1],51),cmap=colormap,alpha=1,antialiased=True,vmin=signal_lut_range[0],vmax=signal_lut_range[1])
    figure.gca().contour(signal_map.xx,signal_map.yy,signal_map.signal_map(signal_name),np.linspace(signal_range[0],signal_range[1],51),cmap='gray',alpha=0.2,linewidths=1,antialiased=True,vmin=-1,vmax=0)

    for a in xrange(16):
        figure.gca().contourf(signal_map.xx,signal_map.yy,signal_map.confidence_map,[-100,0.1+a/24.],cmap='gray_r',alpha=1-a/15.,vmin=1,vmax=2)
    
    # c = patch.Circle(xy=[0,0],radius=clv3_radius,ec="#c94389",fc='None',lw=5,alpha=0.5)
    # figure.gca().add_artist(c)

    if distance_rings:
        CS = figure.gca().contour(signal_map.xx, signal_map.yy, signal_map.rr, np.linspace(0,signal_map.extent,signal_map.extent/10.+1),cmap='Greys',vmin=-1,vmax=0,linewidth=1,alpha=0.2)
        figure.gca().clabel(CS, inline=1, fontsize=8,alpha=0.2)
    
    figure.gca().axis('equal')
    figure.gca().set_xlim(-signal_map.extent*1.05,signal_map.extent*1.05)
    figure.gca().set_ylim(-signal_map.extent*1.05,signal_map.extent*1.05)
    figure.gca().axis('off')

    return figure


