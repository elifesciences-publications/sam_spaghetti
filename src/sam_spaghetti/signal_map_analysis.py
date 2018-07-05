import numpy as np
import scipy.ndimage as nd
import pandas as pd

def zero_crossings(polar_map, sign=1, angle=0):
    zero_crossings = np.zeros_like(polar_map)
    if polar_map.ndim == 2:
        zero_crossings += np.maximum(np.cos(angle*np.pi/180.)*np.concatenate([np.zeros_like(polar_map[:1]),(sign*polar_map[:-1]>0) & (sign*polar_map[1:]<0)],axis=0),0)
        #zero_crossings += np.concatenate([(polar_map[:-1]>0) & (polar_map[1:]<0),np.zeros_like(polar_map[:1])],axis=0)
        zero_crossings += np.maximum(np.sin(angle*np.pi/180.)*np.concatenate([np.zeros_like(polar_map[:,:1]),(sign*polar_map[:,:-1]>0) & (sign*polar_map[:,1:]<0)],axis=1),0)
        #zero_crossings += np.concatenate([(polar_map[:,:-1]>0) & (polar_map[:,:1]<0),np.zeros_like(polar_map[:,:1])],axis=1)
    return zero_crossings


def compute_signal_map_gradient(signal_map, signal_names=None, gaussian_sigma=2.0, norm=True):
    
    xx = signal_map.xx 
    yy = signal_map.yy

    dX = xx[1,1]-xx[0,0]
    dY = yy[1,1]-yy[0,0]

    if signal_names is None:
        signal_names = signal_map.signal_names()

    if signal_map.polar:
        R = signal_map.rr
        dR = R[1,1]-R[0,0]

        T = signal_map.tt
        dT = T[1,1]-T[0,0]

    for signal_name in signal_names:

        if signal_map.polar:
            signal_map_gradient_r = nd.gaussian_filter1d(np.tile(signal_map.signal_map(signal_name),(1,3)),sigma=gaussian_sigma,order=1,axis=0)
            signal_map_gradient_theta = nd.gaussian_filter1d(np.tile(signal_map.signal_map(signal_name),(1,3)),sigma=gaussian_sigma,order=1,axis=1)
                        
            signal_map_gradient_x = np.cos(np.tile(T,(1,3)))*signal_map_gradient_r/dR - np.sin(np.tile(T,(1,3)))/(1e-5+np.tile(R,(1,3)))*signal_map_gradient_theta/dT
            signal_map_gradient_y = np.sin(np.tile(T,(1,3)))*signal_map_gradient_r/dR + np.cos(np.tile(T,(1,3)))/(1e-5+np.tile(R,(1,3)))*signal_map_gradient_theta/dT

            signal_map_gradient_x = signal_map_gradient_x[:,xx.shape[1]:2*xx.shape[1]]
            signal_map_gradient_y = signal_map_gradient_y[:,xx.shape[1]:2*xx.shape[1]]
        else:
            signal_map_gradient_x = nd.gaussian_filter1d(signal_map.signal_map(signal_name),sigma=gaussian_sigma,order=1,axis=0)
            signal_map_gradient_y = nd.gaussian_filter1d(signal_map.signal_map(signal_name),sigma=gaussian_sigma,order=1,axis=1)

        signal_map.update_signal_map(signal_name+"_gradient",np.linalg.norm([signal_map_gradient_x,signal_map_gradient_y],axis=0))
        signal_map.update_signal_map(signal_name+"_gradient_x",signal_map_gradient_x)
        signal_map.update_signal_map(signal_name+"_gradient_y",signal_map_gradient_y)


def compute_signal_map_landscape(signal_map, signal_names=None):

    if signal_names is None:
        signal_names = signal_map.signal_names()

    for signal_name in signal_names:
        if not signal_name+"_gradient" in signal_map.signal_maps:
            compute_signal_map_gradient(signal_map, signal_names, norm=False)

        G_x = np.tile(signal_map.signal_maps[signal_name+"_gradient_x"],(1,3))
        G_y = np.tile(signal_map.signal_maps[signal_name+"_gradient_y"],(1,3))

        gradient_zero_crossings = np.zeros_like(G_x)
        gradient_valleys = np.zeros_like(G_x)
        gradient_ridges = np.zeros_like(G_x)

        for angle in np.arange(360):
            if signal_map.polar:
                G_angle = G_x*np.cos(np.tile(signal_map.tt,(1,3))+np.radians(angle)) + G_y*np.sin(np.tile(signal_map.tt,(1,3))+np.radians(angle))
            else:
                G_angle = G_x*np.cos(np.radians(angle)) + G_y*np.sin(np.radians(angle))
            #gradient_zero_crossings += zero_crossings(G_angle,sign=1,angle=angle)
            #gradient_zero_crossings -= zero_crossings(G_angle,sign=-1,angle=angle)
            gradient_valleys += zero_crossings(G_angle,sign=-1,angle=angle)
            gradient_ridges += zero_crossings(G_angle,sign=1,angle=angle)
            
        gradient_valleys = gradient_valleys[:,signal_map.shape[1]:2*signal_map.shape[1]]/360.
        gradient_ridges = gradient_ridges[:,signal_map.shape[1]:2*signal_map.shape[1]]/360.

        signal_map.update_signal_map(signal_name+"_valleys",gradient_valleys)
        signal_map.update_signal_map(signal_name+"_ridges",gradient_ridges)
        

def signal_map_landscape_analysis(signal_map, signal_name, threshold=0.04, min_area=4.0):

    mask = signal_map.confidence_map>0.5

    xx = signal_map.xx 
    yy = signal_map.yy

    R = signal_map.rr
    T = signal_map.tt

    if signal_map.polar:
        dR = R[1,1]-R[0,0]
        dT = T[1,1]-T[0,0]

        radial_areas = (R+dR)*dR*dT
    else:
        dX = xx[1,1]-xx[0,0]
        dY = yy[1,1]-yy[0,0]

        radial_areas = dX*dY*np.ones_like(xx)

    gradient_valleys = signal_map.signal_map(signal_name+"_valleys")
    gradient_ridges = signal_map.signal_map(signal_name+"_ridges")
        
    saddles = (np.sqrt(gradient_valleys*gradient_ridges))>threshold
    saddles = nd.binary_dilation(saddles,iterations=2)
    if mask is not None:
        saddles[True-mask]=0
    labelled_saddles,_ = nd.label(saddles)
    saddle_labels = np.arange(labelled_saddles.max())+1
    
    valleys = (gradient_valleys>=threshold).astype(int)
    valleys[saddles] = 0

    labelled_valleys,_ = nd.label(valleys)
    valley_labels = np.arange(labelled_valleys.max())+1
    valley_areas = dict(zip(valley_labels,nd.sum(radial_areas,labelled_valleys,index=valley_labels)))

    for v in valley_labels:
        if valley_areas[v]<min_area:
            labelled_valleys[labelled_valleys==v] = 0    
    if mask is not None:
        labelled_valleys[True-mask]=0
        
    ridges = (gradient_ridges>=threshold).astype(int)
    ridges[saddles] = 0
    labelled_ridges,_ = nd.label(ridges)
    ridge_labels = np.arange(labelled_ridges.max())+1
    ridge_areas = dict(zip(ridge_labels,nd.sum(radial_areas,labelled_ridges,index=ridge_labels)))
    for v in ridge_labels:
        if ridge_areas[v]<min_area:
            labelled_ridges[labelled_ridges==v] = 0
    if mask is not None:
        labelled_ridges[True-mask]=0


    minima_data = pd.DataFrame()
    minima_data['label'] = np.sort(np.unique(labelled_valleys))[1:]
    minima_data['point'] = [np.transpose([xx[labelled_valleys==v],yy[labelled_valleys==v]])[np.argmin(signal_map.signal_map(signal_name)[labelled_valleys==v])] for v in minima_data['label']]
    minima_data[signal_name] = [np.min(signal_map.signal_map(signal_name)[labelled_valleys==v]) for v in minima_data['label']]            
    minima_data['area'] = [valley_areas[v] for v in minima_data['label']]
    minima_data['extremality'] = [(gradient_valleys[labelled_valleys==v][np.argmin(signal_map.signal_map(signal_name)[labelled_valleys==v])]+np.max(gradient_valleys[labelled_valleys==v])) for v in minima_data['label']]
    
    minima_data['radial_distance'] = [R[labelled_valleys==v][np.argmin(signal_map.signal_map(signal_name)[labelled_valleys==v])] for v in minima_data['label']]
    minima_data['aligned_theta'] = [np.degrees(T)[labelled_valleys==v][np.argmin(signal_map.signal_map(signal_name)[labelled_valleys==v])] for v in minima_data['label']]
    minima_data['aligned_x'] = minima_data['radial_distance']*np.cos(np.radians(minima_data['aligned_theta']))
    minima_data['aligned_y'] = minima_data['radial_distance']*np.sin(np.radians(minima_data['aligned_theta']))

    minima_data['extremum_type'] = 'minimum'
    minima_data = minima_data.set_index('label',drop=False)


    maxima_data = pd.DataFrame()
    maxima_data['label'] = np.sort(np.unique(labelled_ridges))[1:]
    maxima_data['point'] = [np.transpose([xx[labelled_ridges==v],yy[labelled_ridges==v]])[np.argmax(signal_map.signal_map(signal_name)[labelled_ridges==v])] for v in maxima_data['label']]
    maxima_data[signal_name] = [np.max(signal_map.signal_map(signal_name)[labelled_ridges==v]) for v in maxima_data['label']]            
    maxima_data['area'] = [ridge_areas[v] for v in maxima_data['label']]
    maxima_data['extremality'] = [(gradient_ridges[labelled_ridges==v][np.argmax(signal_map.signal_map(signal_name)[labelled_ridges==v])]+np.max(gradient_ridges[labelled_ridges==v])) for v in maxima_data['label']]
    
    maxima_data['radial_distance'] = [R[labelled_ridges==v][np.argmax(signal_map.signal_map(signal_name)[labelled_ridges==v])] for v in maxima_data['label']]
    maxima_data['aligned_theta'] = [np.degrees(T)[labelled_ridges==v][np.argmax(signal_map.signal_map(signal_name)[labelled_ridges==v])] for v in maxima_data['label']]
    maxima_data['aligned_x'] = maxima_data['radial_distance']*np.cos(np.radians(maxima_data['aligned_theta']))
    maxima_data['aligned_y'] = maxima_data['radial_distance']*np.sin(np.radians(maxima_data['aligned_theta']))
    
    maxima_data['extremum_type'] = 'maximum'
    maxima_data = maxima_data.set_index('label',drop=False)

    
    saddle_data = pd.DataFrame()
    saddle_data['label'] = np.sort(np.unique(labelled_saddles))[1:]
    saddle_data['point'] = [np.transpose([xx[labelled_saddles==s],yy[labelled_saddles==s]])[np.argmax((gradient_valleys*gradient_ridges)[labelled_saddles==s])] for s in  saddle_data['label']]
    saddle_data[signal_name] = [signal_map.signal_map(signal_name)[labelled_saddles==s][np.argmax((gradient_valleys*gradient_ridges)[labelled_saddles==s])] for s in  saddle_data['label']]            
    # saddle_data['extremality'] = [np.max((gradient_valleys*gradient_ridges)[labelled_saddles==v])*16. for v in saddle_data['label']]
    saddle_data['extremality'] = [np.max(4.*np.sqrt(gradient_valleys*gradient_ridges)[labelled_saddles==v]) for v in saddle_data['label']]
    
    saddle_data['radial_distance'] = [R[labelled_saddles==v][np.argmax((gradient_valleys*gradient_ridges)[labelled_saddles==v])] for v in saddle_data['label']]
    saddle_data['aligned_theta'] = [np.degrees(T)[labelled_saddles==v][np.argmax((gradient_valleys*gradient_ridges)[labelled_saddles==v])] for v in saddle_data['label']]
    saddle_data['aligned_x'] = saddle_data['radial_distance']*np.cos(np.radians(saddle_data['aligned_theta']))
    saddle_data['aligned_y'] = saddle_data['radial_distance']*np.sin(np.radians(saddle_data['aligned_theta']))
        
    saddle_data['extremum_type']='saddle'
    saddle_data = saddle_data.set_index('label',drop=False)

    extrema_data = pd.concat([minima_data,maxima_data,saddle_data])

    return extrema_data
    


