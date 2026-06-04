"""

LIST OF FUNCTIONS

LoadAndCrop
cropframe
Reference
Locate
TrackLocation
LocationThresh_View
ROI_plot
ROI_Location
Batch_LoadFiles
Batch_Process
PlayVideo
PlayVideo_ext
showtrace
Heatmap
DistanceTool
ScaleDistance

Added by me (May 5, 2026)
VelocityTrace
AccelerationTrace

"""





########################################################################################

import os
import sys
import cv2
import fnmatch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import PIL.Image
import time
import warnings
import functools as fct
from scipy import ndimage
from tqdm import tqdm
import holoviews as hv
from holoviews import opts
from holoviews import streams
from holoviews.streams import Stream, param
from io import BytesIO
from IPython.display import clear_output, Image, display
from scipy.signal import savgol_filter
hv.notebook_extension('bokeh')
warnings.filterwarnings("ignore")




########################################################################################

def _rotate_frame_and_coords(frame, X=None, Y=None, k=3):
    """
    -------------------------------------------------------------------------------------

    Rotate a 2-D frame by 90° increments and optionally transform scatter
    coordinates so they remain correctly positioned on the rotated image.

    -------------------------------------------------------------------------------------
    Args:
        frame :: [numpy.ndarray]
            2-D grayscale array (e.g. reference frame or heatmap).

        X :: [array-like or None]
            Pixel column coordinates in the *original* frame space.

        Y :: [array-like or None]
            Pixel row coordinates in the *original* frame space.

        k :: [int]
            Number of 90° counter-clockwise rotations (numpy convention).
            Pass k=3 for a single clockwise 90° rotation (default), which
            turns a tall | track into a wide ---- layout.

    -------------------------------------------------------------------------------------
    Returns:
        rotated :: [numpy.ndarray]
            Rotated frame.  Shape is (W, H) for k=1 or k=3 starting from (H, W).

        new_X, new_Y :: [numpy.ndarray]  (only when X and Y are supplied)
            Transformed coordinates matching the rotated frame.

    -------------------------------------------------------------------------------------
    Notes:
        Coordinate mapping for k=3 (clockwise 90°):
            new_X = (H - 1 - old_Y)
            new_Y = old_X
        where H is the *original* frame height.

    """
    rotated = np.rot90(frame, k=k)
    if X is not None and Y is not None:
        H = frame.shape[0]
        X_arr = np.asarray(X, dtype=float)
        Y_arr = np.asarray(Y, dtype=float)
        if k % 4 == 3:   # clockwise 90°
            new_X = (H - 1) - Y_arr
            new_Y = X_arr
        elif k % 4 == 1: # counter-clockwise 90°
            W = frame.shape[1]
            new_X = Y_arr
            new_Y = (W - 1) - X_arr
        elif k % 4 == 2: # 180°
            W = frame.shape[1]
            new_X = (W - 1) - X_arr
            new_Y = (H - 1) - Y_arr
        else:             # 0° / no-op
            new_X = X_arr
            new_Y = Y_arr
        return rotated, new_X, new_Y
    return rotated




########################################################################################    

def LoadAndCrop(video_dict, cropmethod=None, fstfile=False, accept_p_frames=False, rotate=False):
    """ 
    -------------------------------------------------------------------------------------
    
    Loads video and creates interactive cropping tool (video_dict['crop'] from first frame. In the 
    case of batch processing, the first frame of the first video is used. Additionally, 
    when batch processing, the same cropping parameters will be appplied to every video.  
    Care should therefore be taken that the region of interest is in the same position across 
    videos.
    
    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection for selection of cropping parameters
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
                
        cropmethod:: [str]
            Method of cropping video.  cropmethod takes the following values:
                None : No cropping 
                'Box' : Create box selection tool for cropping video
                
        fstfile:: [bool]
            Dictates whether to use first file in video_dict['FileNames'] to generate
            reference.  True/False
        
        accept_p_frames::[bool]
            Dictates whether to allow videos with temporal compresssion.  Currenntly, if
            more than 1/100 frames returns false, error is flagged.
    
    -------------------------------------------------------------------------------------
    Returns:
        image:: [holoviews.Image]
            Holoviews hv.Image displaying first frame
            
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection for selection of cropping parameters
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
                
    
    -------------------------------------------------------------------------------------
    Notes:
        - in the case of batch processing, video_dict['file'] is set to first 
          video in file 
        - prior cropping method HLine has been removed
    
    """
    
    #if batch processing, set file to first file to be processed
    video_dict['file'] = video_dict['FileNames'][0] if fstfile else video_dict['file']      
    
    #Upoad file and check that it exists
    video_dict['fpath'] = os.path.join(os.path.normpath(video_dict['dpath']), video_dict['file'])
    if os.path.isfile(video_dict['fpath']):
        print('file: {file}'.format(file=video_dict['fpath']))
        cap = cv2.VideoCapture(video_dict['fpath'])
    else:
        raise FileNotFoundError('{file} not found. Check that directory and file names are correct'.format(
            file=video_dict['fpath']))

    #Print video information. Note that max frame is updated later if fewer frames detected
    cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    print('total frames: {frames}'.format(frames=cap_max))
    print('nominal fps: {fps}'.format(fps=cap.get(cv2.CAP_PROP_FPS)))
    print('dimensions (h x w): {h},{w}'.format(
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))))
    
    #check for video p-frames
    if accept_p_frames is False:
        check_p_frames(cap)
    
    #Set first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_dict['start']) 
    ret, frame = cap.read() 
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if (video_dict['dsmpl'] < 1):
        frame = cv2.resize(
                    frame,
                    (
                        int(frame.shape[1]*video_dict['dsmpl']),
                        int(frame.shape[0]*video_dict['dsmpl'])
                    ),
                    cv2.INTER_NEAREST)
    video_dict['f0'] = frame
    cap.release()

    # --- rotation for display only ---
    # NOTE: rotation is intentionally disabled when cropmethod='Box'.
    # The BoxEdit stream feeds raw pixel coordinates into cropframe(), which
    # operates on the original (unrotated) frame.  Rotating the preview while
    # the BoxEdit tool is active would make the drawn box incorrect.
    if rotate and cropmethod is not None:
        print("WARNING: rotate=True is ignored when cropmethod='Box'. "
              "Crop selection must be performed on the original, unrotated frame "
              "so that the crop coordinates are valid for video processing. "
              "Use cropmethod=None with rotate=True to preview the rotated frame.")
        rotate = False

    frame_disp = _rotate_frame_and_coords(frame, k=3) if rotate else frame

    #Make first image reference frame on which cropping can be performed
    image = hv.Image((np.arange(frame_disp.shape[1]), np.arange(frame_disp.shape[0]), frame_disp))
    image.opts(
        width=int(frame_disp.shape[1]*video_dict['stretch']['width']),
        height=int(frame_disp.shape[0]*video_dict['stretch']['height']),
        invert_yaxis=True,
        cmap='gray',
        colorbar=True,
        toolbar='below',
        title="First Frame.  Crop if Desired"
    )
    
    #Create polygon element on which to draw and connect via stream to poly drawing tool
    if cropmethod==None:
        image.opts(title="First Frame")
        video_dict['crop'] = None
        return image, video_dict
    
    if cropmethod=='Box':         
        box = hv.Polygons([])
        box.opts(alpha=.5)
        video_dict['crop'] = streams.BoxEdit(source=box,num_objects=1)     
        return (image*box), video_dict
    
    
    
    

########################################################################################

def cropframe(frame, crop=None):
    """ 
    -------------------------------------------------------------------------------------
    
    Crops passed frame with `crop` specification
    
    -------------------------------------------------------------------------------------
    Args:
        frame:: [numpy.ndarray]
            2d numpy array 
        crop:: [hv.streams.stream]
            Holoviews stream object enabling dynamic selection in response to 
            cropping tool. `crop.data` contains x and y coordinates of crop
            boundary vertices. Set to None if no cropping supplied.
    
    -------------------------------------------------------------------------------------
    Returns:
        frame:: [numpy.ndarray]
            2d numpy array
    
    -------------------------------------------------------------------------------------
    Notes:

    """
    
    try:
        Xs=[crop.data['x0'][0],crop.data['x1'][0]]
        Ys=[crop.data['y0'][0],crop.data['y1'][0]]
        fxmin,fxmax=int(min(Xs)), int(max(Xs))
        fymin,fymax=int(min(Ys)), int(max(Ys))
        return frame[fymin:fymax,fxmin:fxmax]
    except:
        return frame
 
    
    
    

########################################################################################

def Reference(video_dict, num_frames=100,
              altfile=False, fstfile=False, frames=None, rotate=False):
    """ 
    -------------------------------------------------------------------------------------
    
    Generates reference frame by taking median of random subset of frames.  This has the 
    effect of removing animal from frame provided animal is not inactive for >=50% of
    the video segment.  
    
    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
        
        num_frames:: [uint]
            Number of frames to base reference frame on.
            
        altfile:: [bool]
            Specify whether alternative file than video to be processed will be
            used to generate reference frame. If `altfile=True`, it is expected
            that `video_dict` contains `altfile` key.
        
        fstfile:: [bool]
            Dictates whether to use first file in video_dict['FileNames'] to generate
            reference.  True/False
        
        frames:: [np array]
            User defined selection of frames to use for generating reference
    
    -------------------------------------------------------------------------------------
    Returns:
        reference:: [numpy.array]
            Reference image. Median of random subset of frames.
        image:: [holoviews.image]
            Holoviews Image of reference image.
    
    -------------------------------------------------------------------------------------
    Notes:
        - If `altfile` is specified, it will be used to generate reference.
    
    """
    
    #set file to use for reference
    video_dict['file'] = video_dict['FileNames'][0] if fstfile else video_dict['file']      
    vname = video_dict.get("altfile","") if altfile else video_dict['file']    
    fpath = os.path.join(os.path.normpath(video_dict['dpath']), vname)
    if os.path.isfile(fpath):
        cap = cv2.VideoCapture(fpath)
    else:
        raise FileNotFoundError('File not found. Check that directory and file names are correct.')
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    
    #Get video dimensions with any cropping applied
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if (video_dict['dsmpl'] < 1):
        frame = cv2.resize(
                    frame,
                    (
                        int(frame.shape[1]*video_dict['dsmpl']),
                        int(frame.shape[0]*video_dict['dsmpl'])
                    ),
                    cv2.INTER_NEAREST)
    frame = cropframe(
        frame, 
        video_dict.get('crop')
    )
    h,w = frame.shape[0], frame.shape[1]
    cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    cap_max = int(video_dict['end']) if video_dict['end'] is not None else cap_max
    
    #Collect subset of frames
    if frames is None:
        #frames = np.random.randint(video_dict['start'],cap_max,num_frames)
        frames = np.linspace(start=video_dict['start'], stop=cap_max, num=num_frames)
    else:
        num_frames = len(frames) #make sure num_frames equals length of passed list
        
    collection = np.zeros((num_frames,h,w))  
    for (idx,framenum) in enumerate(frames):    
        grabbed = False
        while grabbed == False: 
            cap.set(cv2.CAP_PROP_POS_FRAMES, framenum)
            ret, frame = cap.read()
            if ret == True:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if (video_dict['dsmpl'] < 1):
                    gray = cv2.resize(
                        gray,
                        (
                            int(gray.shape[1]*video_dict['dsmpl']),
                            int(gray.shape[0]*video_dict['dsmpl'])
                        ),
                        cv2.INTER_NEAREST)
                gray = cropframe(
                    gray, 
                    video_dict.get('crop')
                )
                collection[idx,:,:]=gray
                grabbed = True
            elif ret == False:
                framenum = np.random.randint(video_dict['start'],cap_max,1)[0]
                pass
    cap.release() 

    reference = np.median(collection,axis=0)

    # --- rotation for display only ---
    # video_dict['reference'] is stored unrotated so that Locate() / TrackLocation()
    # can correctly compare it against raw (unrotated) video frames.
    reference_disp = _rotate_frame_and_coords(reference, k=3) if rotate else reference

    image = hv.Image((np.arange(reference_disp.shape[1]),
                      np.arange(reference_disp.shape[0]), 
                      reference_disp)).opts(width=int(reference_disp.shape[1]*video_dict['stretch']['width']),
                                       height=int(reference_disp.shape[0]*video_dict['stretch']['height']),
                                       invert_yaxis=True,
                                       cmap='gray',
                                       colorbar=True,
                                       toolbar='below',
                                       title="Reference Frame") 
    return reference, image    





########################################################################################

def Locate(cap,tracking_params,video_dict,prior=None):
    """ 
    -------------------------------------------------------------------------------------
    
    Return location of animal in frame, in x/y coordinates. 
    
    -------------------------------------------------------------------------------------
    Args:
        cap:: [cv2.VideoCapture]
            OpenCV VideoCapture class instance for video.
        
        tracking_params:: [dict]
            Dictionary with the following keys:
                'loc_thresh' : Percentile of difference values below which are set to 0. 
                               After calculating pixel-wise difference between passed 
                               frame and reference frame, these values are tthresholded 
                               to make subsequent defining of center of mass more 
                               reliable. [float between 0-100]
                'use_window' : Will window surrounding prior location be 
                               imposed?  Allows changes in area surrounding animal's 
                               location on previous frame to be more heavily influential
                               in determining animal's current location.
                               After finding pixel-wise difference between passed frame 
                               and reference frame, difference values outside square window 
                               of prior location will be multiplied by (1 - window_weight), 
                               reducing their overall influence. [bool]
                'window_size' : If `use_window=True`, the length of one side of square 
                                window, in pixels. [uint] 
                'window_weight' : 0-1 scale for window, if used, where 1 is maximal 
                                  weight of window surrounding prior locaiton. 
                                  [float between 0-1]
                'method' : 'abs', 'light', or 'dark'.  If 'abs', absolute difference
                           between reference and current frame is taken, and thus the 
                           background of the frame doesn't matter. 'light' specifies that
                           the animal is lighter than the background. 'dark' specifies that 
                           the animal is darker than the background. 
                'rmv_wire' : True/False, indicating whether to use wire removal function.  [bool] 
                'wire_krn' : size of kernel used for morphological opening to remove wire. [int]
                
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
        
        prior:: [list]
            If window is being used, list of length 2 is passed, where first index is 
            prior y position, and second index is prior x position.
    
    -------------------------------------------------------------------------------------
    Returns:
        ret:: [bool]
            Specifies whether frame is returned in response to cv2.VideoCapture.read.
        
        dif:: [numpy.array]
            Pixel-wise difference from prior frame, after thresholding and
            applying window weight.
        
        com:: [tuple]
            Indices of center of mass as tuple in the form: (y,x).
        
        frame:: [numpy.array]
            Original video frame after cropping.
    
    -------------------------------------------------------------------------------------
    Notes:
    
    """
    
    #attempt to load frame
    ret, frame = cap.read() 
    
    #set window dimensions
    if prior != None and tracking_params['use_window']==True:
        window_size = tracking_params['window_size']//2
        ymin,ymax = prior[0]-window_size, prior[0]+window_size
        xmin,xmax = prior[1]-window_size, prior[1]+window_size

    if ret == True:
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if (video_dict['dsmpl'] < 1):
            frame = cv2.resize(
                frame,
                (
                    int(frame.shape[1]*video_dict['dsmpl']),
                    int(frame.shape[0]*video_dict['dsmpl'])
                ),
                cv2.INTER_NEAREST)
        frame = cropframe(
            frame,
            video_dict.get('crop')
        )
        
        #find difference from reference
        if tracking_params['method'] == 'abs':
            dif = np.absolute(frame-video_dict['reference'])
        elif tracking_params['method'] == 'light':
            dif = frame-video_dict['reference']
        elif tracking_params['method'] == 'dark':
            dif = video_dict['reference']-frame
        dif = dif.astype('int16')
        if 'mask' in video_dict.keys():
            if video_dict['mask']['mask'] is not None:
                    dif[video_dict['mask']['mask']] = 0
              
        #apply window
        weight = 1 - tracking_params['window_weight']
        if prior != None and tracking_params['use_window']==True:
            dif = dif + (dif.min() * -1) #scale so lowest value is 0
            dif_weights = np.ones(dif.shape)*weight
            dif_weights[slice(ymin if ymin>0 else 0, ymax),
                        slice(xmin if xmin>0 else 0, xmax)]=1
            dif = dif*dif_weights
            
        #threshold differences and find center of mass for remaining values
        dif[dif<np.percentile(dif,tracking_params['loc_thresh'])]=0
        
        #remove influence of wire
        if tracking_params['rmv_wire'] == True:
            ksize = tracking_params['wire_krn']
            kernel = np.ones((ksize,ksize),np.uint8)
            dif_wirermv = cv2.morphologyEx(dif, cv2.MORPH_OPEN, kernel)
            krn_violation =  dif_wirermv.sum()==0
            dif = dif if krn_violation else dif_wirermv
            if krn_violation:
                print("WARNING: wire_krn too large. Reverting to rmv_wire=False for frame {x}".format(
                    x= int(cap.get(cv2.CAP_PROP_POS_FRAMES)-1-video_dict['start'])))
            
        com=ndimage.measurements.center_of_mass(dif)
        return ret, dif, com, frame
    
    else:
        return ret, None, None, frame

    
    
    
    
########################################################################################        

def TrackLocation(video_dict,tracking_params):
    """ 
    -------------------------------------------------------------------------------------
    
    For each frame in video define location of animal, in x/y coordinates, and distance
    travelled from previous frame.
    
    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array] 
                              
        tracking_params:: [dict]
            Dictionary with the following keys:
                'loc_thresh' : Percentile of difference values below which are set to 0. 
                               After calculating pixel-wise difference between passed 
                               frame and reference frame, these values are tthresholded 
                               to make subsequent defining of center of mass more 
                               reliable. [float between 0-100]
                'use_window' : Will window surrounding prior location be 
                               imposed?  Allows changes in area surrounding animal's 
                               location on previous frame to be more heavily influential
                               in determining animal's current location.
                               After finding pixel-wise difference between passed frame 
                               and reference frame, difference values outside square window 
                               of prior location will be multiplied by (1 - window_weight), 
                               reducing their overall influence. [bool]
                'window_size' : If `use_window=True`, the length of one side of square 
                                window, in pixels. [uint] 
                'window_weight' : 0-1 scale for window, if used, where 1 is maximal 
                                  weight of window surrounding prior locaiton. 
                                  [float between 0-1]
                'method' : 'abs', 'light', or 'dark'.  If 'abs', absolute difference
                           between reference and current frame is taken, and thus the 
                           background of the frame doesn't matter. 'light' specifies that
                           the animal is lighter than the background. 'dark' specifies that 
                           the animal is darker than the background. 
                'rmv_wire' : True/False, indicating whether to use wire removal function.  [bool] 
                'wire_krn' : size of kernel used for morphological opening to remove wire. [int]    
    
    -------------------------------------------------------------------------------------
    Returns:
        df:: [pandas.dataframe]
            Pandas dataframe with frame by frame x and y locations,
            distance travelled, as well as video information and parameter values.
    
    -------------------------------------------------------------------------------------
    Notes:
    
    """
          
    #load video
    cap = cv2.VideoCapture(video_dict['fpath'])#set file
    cap.set(cv2.CAP_PROP_POS_FRAMES,video_dict['start']) #set starting frame
    cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    cap_max = int(video_dict['end']) if video_dict['end'] is not None else cap_max  
    
    #Initialize vector to store motion values in
    X = np.zeros(cap_max - video_dict['start'])
    Y = np.zeros(cap_max - video_dict['start'])
    D = np.zeros(cap_max - video_dict['start'])

    #Loop through frames to detect frame by frame differences
    time.sleep(.2) #allow printing
    for f in tqdm(range(len(D))):
        
        if f>0: 
            yprior = np.around(Y[f-1]).astype(int)
            xprior = np.around(X[f-1]).astype(int)
            ret,dif,com,frame = Locate(cap,tracking_params,video_dict,prior=[yprior,xprior])
        else:
            ret,dif,com,frame = Locate(cap,tracking_params,video_dict)
                                                
        if ret == True:          
            Y[f] = com[0]
            X[f] = com[1]
            if f>0:
                D[f] = np.sqrt((Y[f]-Y[f-1])**2 + (X[f]-X[f-1])**2)
        else:
            #if no frame is detected
            f = f-1
            X = X[:f] #Amend length of X vector
            Y = Y[:f] #Amend length of Y vector
            D = D[:f] #Amend length of D vector
            break   
            
    #release video
    cap.release()
    time.sleep(.2) #allow printing
    print('total frames processed: {f}\n'.format(f=len(D)))
    
    #create pandas dataframe
    df = pd.DataFrame(
    {'File' : video_dict['file'],
     'Location_Thresh': np.ones(len(D))*tracking_params['loc_thresh'],
     'Use_Window': str(tracking_params['use_window']),
     'Window_Weight': np.ones(len(D))*tracking_params['window_weight'],
     'Window_Size': np.ones(len(D))*tracking_params['window_size'],
     'Start_Frame': np.ones(len(D))*video_dict['start'],
     'Frame': np.arange(len(D)),
     'X': X,
     'Y': Y,
     'Distance_px': D
    })
    
    #add region of interest info
    df = ROI_Location(video_dict, df) 
    if video_dict['region_names'] is not None:
        print('Defining transitions...')
        df['ROI_location'] = ROI_linearize(df[video_dict['region_names']])
        df['ROI_transition'] = ROI_transitions(df['ROI_location'])
    
    #update scale, if known
    df = ScaleDistance(video_dict, df=df, column='Distance_px')
       
    return df





########################################################################################

def LocationThresh_View(video_dict, tracking_params, examples=4, rotate=False):
    """ 
    -------------------------------------------------------------------------------------
    
    Display example tracking with selected parameters for a random subset of frames. 
    NOTE that because individual frames are analyzed independently, weighting 
    based upon prior location is not implemented.
    
    -------------------------------------------------------------------------------------
    Args:
  
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
            
        tracking_params:: [dict]
            Dictionary with the following keys:
                'loc_thresh' : Percentile of difference values below which are set to 0. 
                               After calculating pixel-wise difference between passed 
                               frame and reference frame, these values are tthresholded 
                               to make subsequent defining of center of mass more 
                               reliable. [float between 0-100]
                'use_window' : Will window surrounding prior location be 
                               imposed?  Allows changes in area surrounding animal's 
                               location on previous frame to be more heavily influential
                               in determining animal's current location.
                               After finding pixel-wise difference between passed frame 
                               and reference frame, difference values outside square window 
                               of prior location will be multiplied by (1 - window_weight), 
                               reducing their overall influence. [bool]
                'window_size' : If `use_window=True`, the length of one side of square 
                                window, in pixels. [uint] 
                'window_weight' : 0-1 scale for window, if used, where 1 is maximal 
                                  weight of window surrounding prior locaiton. 
                                  [float between 0-1]
                'method' : 'abs', 'light', or 'dark'.  If 'abs', absolute difference
                           between reference and current frame is taken, and thus the 
                           background of the frame doesn't matter. 'light' specifies that
                           the animal is lighter than the background. 'dark' specifies that 
                           the animal is darker than the background. 
                'rmv_wire' : True/False, indicating whether to use wire removal function.  [bool] 
                'wire_krn' : size of kernel used for morphological opening to remove wire. [int] 
                           
        examples:: [uint]
            The number of frames for location tracking to be tested on.
        
    
    -------------------------------------------------------------------------------------
    Returns:
        df:: [holoviews.Layout]
            Returns Holoviews Layout with original images on left and heat plots with 
            animal's estimated position marked on right.
    
    -------------------------------------------------------------------------------------
    Notes:
        - if `stretch` values are modified, this will only influence display and not
          calculation
    
    """
    
    #load video
    cap = cv2.VideoCapture(video_dict['fpath'])
    cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    cap_max = int(video_dict['end']) if video_dict['end'] is not None else cap_max
    
    #examine random frames
    images = []
    for example in range (examples):
        
        #analyze frame
        ret = False
        while ret is False:     
            frm=np.random.randint(video_dict['start'],cap_max) #select random frame
            cap.set(cv2.CAP_PROP_POS_FRAMES,frm) #sets frame to be next to be grabbed
            ret,dif,com,frame = Locate(cap, tracking_params, video_dict) 

        #plot original frame
        frame_disp = frame
        dif_disp   = dif * (255 // dif.max())
        com_x, com_y = com[1], com[0]   # com = [row, col] → display [X=col, Y=row]
        if rotate:
            frame_disp = _rotate_frame_and_coords(frame, k=3)
            dif_disp   = _rotate_frame_and_coords(dif * (255 // dif.max()), k=3)
            H = frame.shape[0]
            com_x, com_y = (H - 1 - com[0]), com[1]   # CW: new_X=H-1-old_Y, new_Y=old_X

        image_orig = hv.Image((np.arange(frame_disp.shape[1]), np.arange(frame_disp.shape[0]), frame_disp))
        image_orig.opts(
            width=int(frame_disp.shape[1]*video_dict['stretch']['width']),
            height=int(frame_disp.shape[0]*video_dict['stretch']['height']),
            invert_yaxis=True, cmap='gray', toolbar='below',
            title="Frame: " + str(frm))
        orig_overlay = image_orig * hv.Points(([com_x], [com_y])).opts(
            color='red', size=20, marker='+', line_width=3)

        #plot heatmap
        image_heat = hv.Image((
            np.arange(dif_disp.shape[1]),
            np.arange(dif_disp.shape[0]),
            dif_disp))
        image_heat.opts(
            width=int(dif_disp.shape[1]*video_dict['stretch']['width']),
            height=int(dif_disp.shape[0]*video_dict['stretch']['height']),
            invert_yaxis=True, cmap='jet', toolbar='below',
            title="Frame: " + str(frm - video_dict['start']))
        heat_overlay = image_heat * hv.Points(([com_x], [com_y])).opts(
            color='red', size=20, marker='+', line_width=3)
        
        images.extend([orig_overlay,heat_overlay])
    
    cap.release()
    layout = hv.Layout(images)
    return layout





########################################################################################    
    
def ROI_plot(video_dict, rotate=False):
    """ 
    -------------------------------------------------------------------------------------
    
    Creates interactive tool for defining regions of interest, based upon array
    `region_names`. If `region_names=None`, reference frame is returned but no regions
    can be drawn.
    
    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
                                      
    
    -------------------------------------------------------------------------------------
    Returns:
        image * poly * dmap:: [holoviews.Overlay]
            Reference frame that can be drawn upon to define regions of interest.
        
        poly_stream:: [hv.streams.stream]
            Holoviews stream object enabling dynamic selection in response to 
            selection tool. `poly_stream.data` contains x and y coordinates of roi 
            vertices.
    
    -------------------------------------------------------------------------------------
    Notes:
        - if `stretch` values are modified, this will only influence dispplay and not
          calculation
    
    """
    
    #get number of objects to be drawn
    nobjects = len(video_dict['region_names']) if video_dict['region_names'] else 0

    # --- rotation for display ---
    # When rotate=True the reference is displayed rotated so the user draws ROI polygons
    # in rotated pixel space.  ROI_Location() checks this flag and back-transforms the
    # polygon vertices to original frame space before creating the binary masks, keeping
    # tracking coordinates (X, Y) and masks consistent.
    video_dict['rotate'] = rotate
    ref_disp = _rotate_frame_and_coords(video_dict['reference'], k=3) if rotate else video_dict['reference']

    #Make reference image the base image on which to draw
    image = hv.Image((
        np.arange(ref_disp.shape[1]),
        np.arange(ref_disp.shape[0]),
        ref_disp))
    image.opts(
        width=int(ref_disp.shape[1]*video_dict['stretch']['width']),
        height=int(ref_disp.shape[0]*video_dict['stretch']['height']),
        invert_yaxis=True,cmap='gray', colorbar=True,toolbar='below',
        title="No Regions to Draw" if nobjects == 0 else "Draw Regions: "+', '.join(video_dict['region_names']))

    #Create polygon element on which to draw and connect via stream to PolyDraw drawing tool
    poly = hv.Polygons([])
    poly_stream = streams.PolyDraw(source=poly, drag=True, num_objects=nobjects, show_vertices=True)
    poly.opts(fill_alpha=0.3, active_tools=['poly_draw'])

    def centers(data):
        try:
            x_ls, y_ls = data['xs'], data['ys']
        except TypeError:
            x_ls, y_ls = [], []
        xs = [np.mean(x) for x in x_ls]
        ys = [np.mean(y) for y in y_ls]
        rois = video_dict['region_names'][:len(xs)]
        return hv.Labels((xs, ys, rois))
    
    if nobjects > 0:
        dmap = hv.DynamicMap(centers, streams=[poly_stream])
        return (image * poly * dmap), poly_stream
    else:
        return (image),None
    

    
    
    
########################################################################################    

def ROI_Location(video_dict, location):
    """ 
    -------------------------------------------------------------------------------------
    
    For each frame, determine which regions of interest the animal is in.  For each
    region of interest, boolean array is added to `location` dataframe passed, with 
    column name being the region name.
    
    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
        
        location:: [pandas.dataframe]
            Pandas dataframe with frame by frame x and y locations,
            distance travelled, as well as video information and parameter values. 
            Must contain column names 'X' and 'Y'.

    -------------------------------------------------------------------------------------
    Returns:
        location:: [pandas.dataframe]
            For each region of interest, boolean array is added to `location` dataframe 
            passed, with column name being the region name. Additionally, under column
            `ROI_coordinates`, coordinates of vertices of each region of interest are
            printed. This takes the form of a dictionary of x and y coordinates, e.g.:
                'xs' : [[region 1 x coords], [region 2 x coords]],
                'ys' : [[region 1 y coords], [region 2 y coords]]
                                      
    -------------------------------------------------------------------------------------
    Notes:
    
    """
    
    if video_dict['region_names'] == None:
        return location

    # When ROI_plot was called with rotate=True the polygon vertices recorded by
    # poly_stream are in *rotated* pixel space (CW 90°, k=3).  We must invert the
    # transform back to original frame space before building the binary masks,
    # because cv2.fillPoly and the tracked X/Y coordinates are both in original space.
    #
    # Inverse of CW 90° (k=3):  original_x = rot_y,  original_y = (W_rot - 1) - rot_x
    # where W_rot = rotated frame width = original frame height (H).
    needs_unrotate = video_dict.get('rotate', False)
    H_orig = video_dict['reference'].shape[0]   # original frame height

    #Create ROI Masks
    ROI_masks = {}
    for poly in range(len(video_dict['roi_stream'].data['xs'])):
        x = np.array(video_dict['roi_stream'].data['xs'][poly]) #x coordinates (rotated if rotate=True)
        y = np.array(video_dict['roi_stream'].data['ys'][poly]) #y coordinates (rotated if rotate=True)
        if needs_unrotate:
            # Back-transform rotated coords → original frame coords
            x_orig = y.copy()               # original_x = rotated_y
            y_orig = (H_orig - 1) - x       # original_y = (H - 1) - rotated_x
            x, y = x_orig, y_orig
        xy = np.column_stack((x, y)).astype('uint64') #xy coordinate pairs
        mask = np.zeros(video_dict['reference'].shape) # create empty mask
        cv2.fillPoly(mask, pts =[xy], color=255) #fill polygon  
        ROI_masks[video_dict['region_names'][poly]] = mask==255 #save to ROI masks as boolean 

    #Create arrays to store whether animal is within given ROI
    ROI_location = {}
    for mask in ROI_masks:
        ROI_location[mask]=np.full(len(location['Frame']),False,dtype=bool)

    #For each frame assess truth of animal being in each ROI
    for f in location['Frame']:
        y,x = location['Y'][f], location['X'][f]
        for mask in ROI_masks:
            ROI_location[mask][f] = ROI_masks[mask][int(y),int(x)]
    
    #Add data to location data frame
    for x in ROI_location:
        location[x]=ROI_location[x]
    
    #Add ROI coordinates
    location['ROI_coordinates']=str(video_dict['roi_stream'].data)
    
    return location





########################################################################################        

def ROI_linearize(rois, null_name = 'non_roi'):
    
    """ 
    -------------------------------------------------------------------------------------
    
    Creates array defining ROI as string for each frame
    
    -------------------------------------------------------------------------------------
    Args:
        rois:: [pd.DataFrame]
            Pandas dataframe where each column corresponds to an ROI, with boolean values
            defining if animal is in said roi.
        null_name:: [string]
            Name used when animals is not in any defined roi.
    
    -------------------------------------------------------------------------------------
    Returns:
        rois['ROI_location']:: [pd.Series]
            pd.Series defining ROI as string for each frame
    
    -------------------------------------------------------------------------------------
    Notes:
    
    """
    region_names = rois.columns.values
    rois['ROI_location'] = null_name
    for region in region_names:
        rois['ROI_location'][rois[region]] = rois['ROI_location'][rois[region]].apply(
            lambda x: '_'.join([x, region]) if x!=null_name else region
        )
    return rois['ROI_location']






########################################################################################        

def ROI_transitions(regions, include_first=False):
    """ 
    -------------------------------------------------------------------------------------
    
    Creates boolean array defining where transitions between each ROI occur.
    
    -------------------------------------------------------------------------------------
    Args:
        regions:: [Pandas Series]
            Pandas Series defining ROI as string for each frame
        include_first:: [string]
            Whether to count first frame as transition
    
    -------------------------------------------------------------------------------------
    Returns:
        transitions:: [Boolean array]
            pd.Series defining where transitions between ROIs occur.
    
    -------------------------------------------------------------------------------------
    Notes:
    
    """
    regions_offset = np.append(regions[0], regions[0:-1])
    transitions = regions!=regions_offset
    if include_first:
        transitions[0] = True
    return transitions





########################################################################################        
    
def Summarize_Location(location, video_dict, bin_dict=None):
    """ 
    -------------------------------------------------------------------------------------
    
    Generates summary of distance travelled and proportional time spent in each region
    of interest according to user defined time bins.  If bins are not provided 
    (`bin_dict=None`), average of entire video segment will be provided.
    
    -------------------------------------------------------------------------------------
    Args:
        location:: [pandas.dataframe]
            Pandas dataframe with frame by frame x and y locations,
            distance travelled, as well as video information and parameter values. 
            Additionally, for each region of interest, boolean array indicating whether 
            animal is in the given region for each frame.
      
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
                              
        bin_dict:: [dict]
            Dictionary specifying bins.  Dictionary keys should be names of the bins.  
            Dictionary value for each bin should be a tuple, with the start and end of 
            the bin, in seconds, relative to the start of the analysis period 
            (i.e. if start frame is 100, it will be relative to that). If no bins are to 
            be specified, set bin_dict = None.
            example: bin_dict = {1:(0,100), 2:(100,200)}                             

    
    -------------------------------------------------------------------------------------
    Returns:
        bins:: [pandas.dataframe]
            Pandas dataframe with distance travelled and proportional time spent in each 
            region of interest according to user defined time bins, as well as video 
            information and parameter values. If no region names are supplied 
            (`region_names=None`), only distance travelled will be included.
                                      
    -------------------------------------------------------------------------------------
    Notes:
    
    """
    
    #define bins
    avg_dict = {'all': (location['Frame'].min(), location['Frame'].max())}   
    bin_dict = bin_dict if bin_dict is not None else avg_dict
    
    #get summary info
    bins = (pd.Series(bin_dict).rename('range(f)')
            .reset_index().rename(columns=dict(index='bin')))    
    bins['Distance_px'] = bins['range(f)'].apply(
        lambda r: location[location['Frame'].between(*r)]['Distance_px'].sum())
    if video_dict['region_names'] is not None:
        bins_reg = bins['range(f)'].apply(
            lambda r: location[location['Frame'].between(*r)][video_dict['region_names']].mean())
        bins = bins.join(bins_reg)
        drp_cols = ['Distance_px', 'Frame', 'X', 'Y'] + video_dict['region_names']
    else:
        drp_cols = ['Distance_px', 'Frame', 'X', 'Y']
    bins = pd.merge(
        location.drop(drp_cols, axis='columns'),
        bins,
        left_index=True,
        right_index=True)
    
    #scale distance
    bins = ScaleDistance(video_dict,df=bins,column='Distance_px') 
    
    return bins





######################################################################################## 

def Batch_LoadFiles(video_dict):
    """ 
    -------------------------------------------------------------------------------------
    
    Populates list of files in directory (`dpath`) that are of the specified file type
    (`ftype`).  List is held in `video_dict['FileNames']`.
    
    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]

    
    -------------------------------------------------------------------------------------
    Returns:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
    
    -------------------------------------------------------------------------------------
    Notes:
    
    """

    #Get list of video files of designated type
    if os.path.isdir(video_dict['dpath']):
        video_dict['FileNames'] = sorted(os.listdir(video_dict['dpath']))
        video_dict['FileNames'] = fnmatch.filter(video_dict['FileNames'], ('*.' + video_dict['ftype'])) 
        return video_dict
    else:
        raise FileNotFoundError('{path} not found. Check that directory is correct'.format(
            path=video_dict['dpath']))

        
        
        
        
######################################################################################## 

def Batch_Process(video_dict,tracking_params,bin_dict,accept_p_frames=False):   
    """ 
    -------------------------------------------------------------------------------------
    
    Run LocationTracking on folder of videos of specified filetype. 
    
    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
        
        tracking_params:: [dict]
            Dictionary with the following keys:
                'loc_thresh' : Percentile of difference values below which are set to 0. 
                               After calculating pixel-wise difference between passed 
                               frame and reference frame, these values are tthresholded 
                               to make subsequent defining of center of mass more 
                               reliable. [float between 0-100]
                'use_window' : Will window surrounding prior location be 
                               imposed?  Allows changes in area surrounding animal's 
                               location on previous frame to be more heavily influential
                               in determining animal's current location.
                               After finding pixel-wise difference between passed frame 
                               and reference frame, difference values outside square window 
                               of prior location will be multiplied by (1 - window_weight), 
                               reducing their overall influence. [bool]
                'window_size' : If `use_window=True`, the length of one side of square 
                                window, in pixels. [uint] 
                'window_weight' : 0-1 scale for window, if used, where 1 is maximal 
                                  weight of window surrounding prior locaiton. 
                                  [float between 0-1]
                'method' : 'abs', 'light', or 'dark'.  If 'abs', absolute difference
                           between reference and current frame is taken, and thus the 
                           background of the frame doesn't matter. 'light' specifies that
                           the animal is lighter than the background. 'dark' specifies that 
                           the animal is darker than the background. 
                'rmv_wire' : True/False, indicating whether to use wire removal function.  [bool] 
                'wire_krn' : size of kernel used for morphological opening to remove wire. [int]
                
         accept_p_frames::[bool]
            Dictates whether to allow videos with temporal compresssion.  Currenntly, if
            more than 1/100 frames returns false, error is flagged.
    
    -------------------------------------------------------------------------------------
    Returns:
        summary_all:: [pandas.dataframe]
            Pandas dataframe with distance travelled and proportional time spent in each 
            region of interest according to user defined time bins, as well as video 
            information and parameter values. If no region names are supplied 
            (`region_names=None`), only distance travelled will be included.
            
        layout:: [hv.Layout]
            Holoviews layout wherein for each session the reference frame is returned
            with the regions of interest highlightted and the animals location across
            the session overlaid atop the reference image.
    
    -------------------------------------------------------------------------------------
    Notes:
    
    """
    
    images = []
    for file in video_dict['FileNames']:
        
        print ('Processing File: {f}'.format(f=file))  
        video_dict['file'] = file 
        video_dict['fpath'] = os.path.join(os.path.normpath(video_dict['dpath']), file)
        
        #Print video information. Note that max frame is updated later if fewer frames detected
        cap = cv2.VideoCapture(video_dict['fpath'])
        cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
        print('total frames: {frames}'.format(frames=cap_max))
        print('nominal fps: {fps}'.format(fps=cap.get(cv2.CAP_PROP_FPS)))
        print('dimensions (h x w): {h},{w}'.format(
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))))
        
        #check for video p-frames
        if accept_p_frames is False:
            check_p_frames(cap)
        
        video_dict['reference'], image = Reference(video_dict,num_frames=50) 
        location = TrackLocation(video_dict,tracking_params)
        location.to_csv(os.path.splitext(video_dict['fpath'])[0] + '_LocationOutput.csv', index=False)
        file_summary = Summarize_Location(location, video_dict, bin_dict=bin_dict)
               
        try: 
            summary_all = pd.concat([summary_all,file_summary],sort=False)
        except NameError: 
            summary_all = file_summary
        
        trace = showtrace(video_dict,location)
        heatmap = Heatmap(video_dict, location, sigma=None)
        images = images + [(trace.opts(title=file)), (heatmap.opts(title=file))]

    #Write summary data to csv file
    sum_pathout = os.path.join(os.path.normpath(video_dict['dpath']), 'BatchSummary.csv')
    summary_all.to_csv(sum_pathout, index=False)
    
    layout = hv.Layout(images)
    return summary_all, layout





########################################################################################        

def PlayVideo(video_dict,display_dict,location):  
    """ 
    -------------------------------------------------------------------------------------
    
    Play portion of video back, displaying animal's estimated location. Video is played
    in notebook

    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
                
        display_dict:: [dict]
            Dictionary with the following keys:
                'start' : start point of video segment in frames [int]
                'end' : end point of video segment in frames [int]
                'resize' : Default is None, in which original size is retained.
                           Alternatively, set to tuple as follows: (width,height).
                           Because this is in pixel units, must be integer values.
                'fps' : frames per second of video file/files to be processed [int]
                'save_video' : option to save video if desired [bool]
                               Currently, will be saved at 20 fps even if video 
                               is something else
                               
        location:: [pandas.dataframe]
            Pandas dataframe with frame by frame x and y locations,
            distance travelled, as well as video information and parameter values. 
            Additionally, for each region of interest, boolean array indicating whether 
            animal is in the given region for each frame. 
            
    
    -------------------------------------------------------------------------------------
    Returns:
        Nothing returned
    
    -------------------------------------------------------------------------------------
    Notes:

    """


    #Load Video and Set Saving Parameters
    cap = cv2.VideoCapture(video_dict['fpath'])#set file\
    if display_dict['save_video']==True:
        ret, frame = cap.read() #read frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if (video_dict['dsmpl'] < 1):
            frame = cv2.resize(
                frame,
                (
                    int(frame.shape[1]*video_dict['dsmpl']),
                    int(frame.shape[0]*video_dict['dsmpl'])
                ),
                cv2.INTER_NEAREST)
        frame = cropframe(frame, video_dict['crop'])
        height, width = int(frame.shape[0]), int(frame.shape[1])
        fourcc = 0#cv2.VideoWriter_fourcc(*'jpeg') #only writes up to 20 fps, though video read can be 30.
        writer = cv2.VideoWriter(os.path.join(os.path.normpath(video_dict['dpath']), 'video_output.avi'), 
                                 fourcc, 20.0, 
                                 (width, height),
                                 isColor=False)

    #Initialize video play options   
    cap.set(cv2.CAP_PROP_POS_FRAMES,video_dict['start']+display_dict['start']) 

    #Play Video
    for f in range(display_dict['start'],display_dict['stop']):
        ret, frame = cap.read() #read frame
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if (video_dict['dsmpl'] < 1):
                frame = cv2.resize(
                    frame,
                    (
                        int(frame.shape[1]*video_dict['dsmpl']),
                        int(frame.shape[0]*video_dict['dsmpl'])
                    ),
                    cv2.INTER_NEAREST)
            frame = cropframe(frame, video_dict['crop'])
            markposition = (int(location['X'][f]),int(location['Y'][f]))
            cv2.drawMarker(img=frame,position=markposition,color=255)
            display_image(frame,display_dict['fps'],display_dict['resize'])
            #Save video (if desired). 
            if display_dict['save_video']==True:
                writer.write(frame) 
        if ret == False:
            print('warning. failed to get video frame')

    #Close video window and video writer if open
    print('Done playing segment')
    if display_dict['save_video']==True:
        writer.release()

def display_image(frame,fps,resize):
    img = PIL.Image.fromarray(frame, "L")
    img = img.resize(size=resize) if resize else img
    buffer = BytesIO()
    img.save(buffer,format="JPEG")    
    display(Image(data=buffer.getvalue()))
    time.sleep(1/fps)
    clear_output(wait=True)

    
    

    
########################################################################################

def PlayVideo_ext(video_dict,display_dict,location,crop=None):  
    """ 
    -------------------------------------------------------------------------------------
    
    Play portion of video back, displaying animal's estimated location

    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
                
        display_dict:: [dict]
            Dictionary with the following keys:
                'start' : start point of video segment in frames [int]
                'end' : end point of video segment in frames [int]
                'fps' : frames per second of video file/files to be processed [int]
                'save_video' : option to save video if desired [bool]
                               Currently, will be saved at 20 fps even if video 
                               is something else
                               
        location:: [pandas.dataframe]
            Pandas dataframe with frame by frame x and y locations,
            distance travelled, as well as video information and parameter values. 
            Additionally, for each region of interest, boolean array indicating whether 
            animal is in the given region for each frame. 
  
    
    -------------------------------------------------------------------------------------
    Returns:
        Nothing returned
    
    -------------------------------------------------------------------------------------
    Notes:

    """

    #Load Video and Set Saving Parameters
    cap = cv2.VideoCapture(video_dict['fpath'])#set file\
    if display_dict['save_video']==True:
        ret, frame = cap.read() #read frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if (video_dict['dsmpl'] < 1):
            frame = cv2.resize(
                frame,
                (
                    int(frame.shape[1]*video_dict['dsmpl']),
                    int(frame.shape[0]*video_dict['dsmpl'])
                ),
                cv2.INTER_NEAREST)
        frame = cropframe(frame, crop)
        height, width = int(frame.shape[0]), int(frame.shape[1])
        fourcc = 0#cv2.VideoWriter_fourcc(*'jpeg') #only writes up to 20 fps, though video read can be 30.
        writer = cv2.VideoWriter(os.path.join(os.path.normpath(video_dict['dpath']), 'video_output.avi'), 
                                 fourcc, 20.0, 
                                 (width, height),
                                 isColor=False)

    #Initialize video play options   
    cap.set(cv2.CAP_PROP_POS_FRAMES,video_dict['start']+display_dict['start']) 
    rate = int(1000/display_dict['fps']) 

    #Play Video
    for f in range(display_dict['start'],display_dict['stop']):
        ret, frame = cap.read() #read frame
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if (video_dict['dsmpl'] < 1):
                frame = cv2.resize(
                    frame,
                    (
                        int(frame.shape[1]*video_dict['dsmpl']),
                        int(frame.shape[0]*video_dict['dsmpl'])
                    ),
                    cv2.INTER_NEAREST)
            frame = cropframe(frame, crop)
            markposition = (int(location['X'][f]),int(location['Y'][f]))
            cv2.drawMarker(img=frame,position=markposition,color=255)
            cv2.imshow("preview",frame)
            cv2.waitKey(rate)
            #Save video (if desired). 
            if display_dict['save_video']==True:
                writer.write(frame) 
        if ret == False:
            print('warning. failed to get video frame')

    #Close video window and video writer if open        
    cv2.destroyAllWindows()
    _=cv2.waitKey(1) 
    if display_dict['save_video']==True:
        writer.release()

    
    
    
    
########################################################################################

def showtrace(video_dict, location, color="red", alpha=.8, size=3, rotate=False):
    """ 
    -------------------------------------------------------------------------------------
    
    Create image where animal location across session is displayed atop reference frame

    -------------------------------------------------------------------------------------
    Args:
        
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
        
        location:: [pandas.dataframe]
            Pandas dataframe with frame by frame x and y locations,
            distance travelled, as well as video information and parameter values. 

            
        color:: [str]
            Color of trace.  See Holoviews documentation for color options
                               
        alpha:: [float]
            Alpha of trace.  See Holoviews documentation for details
        
        size:: [float]
            Size of trace.  See Holoviews documentation for details.     
    
    -------------------------------------------------------------------------------------
    Returns:
        holoviews.Overlay
            Location of animal superimposed upon reference. If poly_stream is passed
            than regions of interest will also be outlined.
    
    -------------------------------------------------------------------------------------
    Notes:

    """
    
    video_dict['roi_stream'] = video_dict['roi_stream'] if 'roi_stream' in video_dict else None

    # --- rotation for display (does not affect stored coordinates) ---
    ref = video_dict['reference']
    X_disp = location['X'].values
    Y_disp = location['Y'].values
    if rotate:
        ref, X_disp, Y_disp = _rotate_frame_and_coords(ref, X_disp, Y_disp, k=3)

    if video_dict['roi_stream'] != None:
        lst = []
        for poly_idx in range(len(video_dict['roi_stream'].data['xs'])):
            x = np.array(video_dict['roi_stream'].data['xs'][poly_idx]) #x coordinates
            y = np.array(video_dict['roi_stream'].data['ys'][poly_idx]) #y coordinates
            if rotate:
                _, x_r, y_r = _rotate_frame_and_coords(
                    video_dict['reference'], x, y, k=3)
                lst.append([(x_r[vert], y_r[vert]) for vert in range(len(x_r))])
            else:
                lst.append([(x[vert], y[vert]) for vert in range(len(x))])
        poly = hv.Polygons(lst).opts(fill_alpha=0.1, line_dash='dashed')

    white_bg = np.ones_like(ref) * 255
    image = hv.Image((np.arange(ref.shape[1]),
                      np.arange(ref.shape[0]),
                      white_bg)
                    ).opts(width=int(ref.shape[1]*video_dict['stretch']['width']),
                           height=int(ref.shape[0]*video_dict['stretch']['height']),
                           invert_yaxis=True, cmap='gray', toolbar='below',
                           title="Motion Trace", bgcolor='white', show_frame=True)

    points = hv.Scatter(np.array([X_disp, Y_disp]).T).opts(
        color='red', alpha=alpha, size=size)

    return (image*poly*points) if video_dict['roi_stream']!=None else (image*points)





########################################################################################    

def Heatmap (video_dict, location, sigma=None, rotate=False):
    """ 
    -------------------------------------------------------------------------------------
    
    Create heatmap of relative time in each location. Max value is set to maxiumum
    in any one location.

    -------------------------------------------------------------------------------------
    Args:
        
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
        
        location:: [pandas.dataframe]
            Pandas dataframe with frame by frame x and y locations,
            distance travelled, as well as video information and parameter values. 
                
        sigma:: [numeric]
            Optional number specifying sigma of guassian filter
  
    
    -------------------------------------------------------------------------------------
    Returns:
        map_i:: [holoviews.Image]
            Heatmap image
    
    -------------------------------------------------------------------------------------
    Notes:
        stretch only affects display

    """    
    heatmap = np.zeros(video_dict['reference'].shape)
    for frame in range(len(location)):
        Y,X = int(location.Y[frame]), int(location.X[frame])
        heatmap[Y,X]+=1
    
    sigma = np.mean(heatmap.shape)*.05 if sigma == None else sigma
    heatmap = cv2.GaussianBlur(heatmap,(0,0),sigma)
    heatmap = (heatmap / heatmap.max())*255

    # --- rotation for display only ---
    if rotate:
        heatmap = _rotate_frame_and_coords(heatmap, k=3)
    
    map_i = hv.Image((np.arange(heatmap.shape[1]), np.arange(heatmap.shape[0]), heatmap))
    map_i.opts(width=int(heatmap.shape[1]*video_dict['stretch']['width']),
           height=int(heatmap.shape[0]*video_dict['stretch']['height']),
           invert_yaxis=True, cmap='jet', alpha=1,
           colorbar=False, toolbar='below', title="Heatmap")
    
    return map_i





########################################################################################    

def DistanceTool(video_dict, rotate=False):
    """ 
    -------------------------------------------------------------------------------------
    
    Creates interactive tool for measuring length between two points, in pixel units, in 
    order to ease process of converting pixel distance measurements to some other scale.
    Use point drawing tool to calculate distance beteen any two popints.
    
    -------------------------------------------------------------------------------------
    Args:
        
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
        
    
    -------------------------------------------------------------------------------------
    Returns:
        image * points * dmap:: [holoviews.Overlay]
            Reference frame that can be drawn upon to define 2 points, the distance 
            between which will be measured and displayed.
        
        distance:: [dict]
            Dictionary with the following keys:
                'd' : Euclidean distance between two reference points, in pixel units, 
                      rounded to thousandth. Returns None if no less than 2 points have 
                      been selected.
    
    -------------------------------------------------------------------------------------
    Notes:
        - if `stretch` values are modified, this will only influence dispplay and not
          calculation
    
    """

    # --- rotation for display only ---
    # Euclidean distance is rotation-invariant, so the measured px_distance is
    # identical regardless of whether rotate=True or False.
    ref_disp = _rotate_frame_and_coords(video_dict['reference'], k=3) if rotate else video_dict['reference']

    #Make reference image the base image on which to draw
    image = hv.Image((
        np.arange(ref_disp.shape[1]),
        np.arange(ref_disp.shape[0]),
        ref_disp))
    image.opts(width=int(ref_disp.shape[1]*video_dict['stretch']['width']),
               height=int(ref_disp.shape[0]*video_dict['stretch']['height']),
              invert_yaxis=True,cmap='gray',
              colorbar=True,
               toolbar='below',
              title="Select Points")

    #Create Point instance on which to draw and connect via stream to pointDraw drawing tool 
    points = hv.Points([]).opts(active_tools=['point_draw'], color='red',size=10)
    pointDraw_stream = streams.PointDraw(source=points,num_objects=2) 
    
    def markers(data, distance):
        try:
            x_ls, y_ls = data['x'], data['y']
        except TypeError:
            x_ls, y_ls = [], []
        
        x_ctr, y_ctr = np.mean(x_ls), np.mean(y_ls)
        if len(x_ls) > 1:
            x_dist = (x_ls[0] - x_ls[1])
            y_dist = (y_ls[0] - y_ls[1])
            distance['px_distance'] = np.around( (x_dist**2 + y_dist**2)**(1/2), 3)
            text = "{dist} px".format(dist=distance['px_distance'])
        return hv.Labels((x_ctr, y_ctr, text if len(x_ls) > 1 else "")).opts(
            text_color='blue',text_font_size='14pt')
    
    distance = dict(px_distance=None)
    markers_ptl = fct.partial(markers, distance=distance)
    dmap = hv.DynamicMap(markers_ptl, streams=[pointDraw_stream])
    return (image * points * dmap), distance


########################################################################################

def setScale(distance, scale, scale_dict):

    """ 
    -------------------------------------------------------------------------------------
    
    Updates dictionary with scale information, given the true distance between points 
    (e.g. 100), and the scale unit (e.g. 'cm')
    
    -------------------------------------------------------------------------------------
    Args:
    
        distance :: [numeric]
            The real-world distance between the selected points
        
        scale :: [string]
            The scale used for defining the real world distance.  Can be any string
            (e.g. 'cm', 'in', 'inch', 'stone')

        scale_dict :: [dict]
            Dictionary with the following keys:
                'px_distance' : distance between reference points, in pixels [numeric]
                'true_distance' : distance between reference points, in desired scale 
                                   (e.g. cm) [numeric]
                'true_scale' : string containing name of scale (e.g. 'cm') [str]
                'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]

    -------------------------------------------------------------------------------------
    Returns:
        scale_dict :: [dict]
                Dictionary with the following keys:
                    'px_distance' : distance between reference points, in pixels [numeric]
                    'true_distance' : distance between reference points, in desired scale 
                                       (e.g. cm) [numeric]
                    'true_scale' : string containing name of scale (e.g. 'cm') [str]
                    'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
    -------------------------------------------------------------------------------------
    Notes:

    """

    scale_dict['true_distance'] = distance
    scale_dict['true_scale'] = scale
    return scale_dict
    


########################################################################################    

def ScaleDistance(video_dict, df=None, column=None):
    """ 
    -------------------------------------------------------------------------------------
    
    Adds column to dataframe by multiplying existing column by scaling factor to change
    scale. Used in order to convert distance from pixel scale to desired real world 
    distance scale.
    
    -------------------------------------------------------------------------------------
    Args:

        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]
        
        df:: [pandas.dataframe]
            Pandas dataframe with column to be scaled.
        
        column:: [str]
            Name of column in df to be scaled
        
    -------------------------------------------------------------------------------------
    Returns:
        df:: [pandas.dataframe]
            Pandas dataframe with column of scaled distance values.
    
    -------------------------------------------------------------------------------------
    Notes:
        - if `stretch` values are modified, this will only influence dispplay and not
          calculation
    
    """
    
    if 'scale' not in video_dict.keys():
        return df

    if video_dict['scale']['px_distance']!= None:
        video_dict['scale']['factor'] = video_dict['scale']['true_distance']/video_dict['scale']['px_distance']
        new_column = "_".join(['Distance', video_dict['scale']['true_scale']])
        df[new_column] = df[column]*video_dict['scale']['factor']
        order = [col for col in df if col not in [column,new_column]]
        order = order + [column,new_column]
        df = df[order]
    else:
        print('Distance between reference points undefined. Cannot scale column: {c}.\
        Returning original dataframe'.format(c=column))
    return df



########################################################################################    
    
def Mask_select(video_dict, fstfile=False):
    """ 
    -------------------------------------------------------------------------------------
    
    Creates interactive tool for defining regions of interest, based upon array
    `region_names`. If `region_names=None`, reference frame is returned but no regions
    can be drawn.
    
    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing 
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proptional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to 
                               selection tool. `poly_stream.data` contains x and y coordinates of roi 
                               vertices. [hv.streams.stream]
                'crop' : Enables dynamic box selection of cropping parameters.  
                         Holoviews stream object enabling dynamic selection in response to 
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                'mask' : [dict]
                    Dictionary with the following keys:
                        'mask' : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)   
                        'mask_stream' : Holoviews stream object enabling dynamic selection 
                                in response to selection tool. `mask_stream.data` contains 
                                x and y coordinates of region vertices. [holoviews polystream]
                'scale:: [dict]
                        Dictionary with the following keys:
                            'px_distance' : distance between reference points, in pixels [numeric]
                            'true_distance' : distance between reference points, in desired scale 
                                               (e.g. cm) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                'ftype' : (only if batch processing) 
                          video file type extension (e.g. 'wmv') [str]
                'FileNames' : (only if batch processing)
                              List of filenames of videos in folder to be batch 
                              processed.  [list]
                'f0' : (only if batch processing)
                        first frame of video [numpy array]

        fstfile:: [bool]
            Dictates whether to use first file in video_dict['FileNames'] to generate
            reference.  True/False
    
    -------------------------------------------------------------------------------------
    Returns:
        image * poly * dmap:: [holoviews.Overlay]
            First frame of video that can be drawn upon to define regions of interest.
            
        mask:: [dict]
            Dictionary with the following keys:
                'mask' : boolean numpy array identifying regions to exlude
                         from analysis.  If no such regions, equal to
                         None. [bool numpy array)   
                'mask_stream' : Holoviews stream object enabling dynamic selection 
                        in response to selection tool. `mask_stream.data` contains 
                        x and y coordinates of region vertices. [holoviews polystream]
    
    -------------------------------------------------------------------------------------
    Notes:
        - if `stretch` values are modified, this will only influence dispplay and not
          calculation
    
    """
    
    #Load first file if batch processing
    if fstfile:
        video_dict['file'] = video_dict['FileNames'][0] 
        video_dict['fpath'] = os.path.join(os.path.normpath(video_dict['dpath']), video_dict['file'])
        if os.path.isfile(video_dict['fpath']):
            print('file: {file}'.format(file=video_dict['fpath']))
            cap = cv2.VideoCapture(video_dict['fpath'])
        else:
            raise FileNotFoundError('{file} not found. Check that directory and file names are correct'.format(
                file=video_dict['fpath']))
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_dict['start']) 
        ret, frame = cap.read() 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if (video_dict['dsmpl'] < 1):
            frame = cv2.resize(
                frame,
                (
                    int(frame.shape[1]*video_dict['dsmpl']),
                    int(frame.shape[0]*video_dict['dsmpl'])
                ),
                cv2.INTER_NEAREST)
        video_dict['f0'] = frame
    
    #Make first image the base image on which to draw
    f0 = cropframe(
        video_dict['f0'],
        video_dict.get('crop')
    )
    image = hv.Image((np.arange(f0.shape[1]), np.arange(f0.shape[0]), f0))
    image.opts(width=int(f0.shape[1]*video_dict['stretch']['width']),
               height=int(f0.shape[0]*video_dict['stretch']['height']),
              invert_yaxis=True,cmap='gray',
              colorbar=True,
               toolbar='below',
              title="Draw Regions to be Exluded")

    #Create polygon element on which to draw and connect via stream to PolyDraw drawing tool
    mask = dict(mask=None)
    poly = hv.Polygons([])
    mask['stream'] = streams.PolyDraw(source=poly, drag=True, show_vertices=True)
    #poly_stream = streams.PolyDraw(source=poly, drag=True, show_vertices=True)
    poly.opts(fill_alpha=0.3, active_tools=['poly_draw'])
    points = hv.Points([]).opts(active_tools=['point_draw'], color='red',size=10)
    pointDraw_stream = streams.PointDraw(source=points,num_objects=2) 
    
    def make_mask(data, mask):
        try:
            x_ls, y_ls = data['xs'], data['ys'] 
        except TypeError:
            x_ls, y_ls = [], []
        
        if len(x_ls)>0:
            mask['mask'] = np.zeros(f0.shape) 
            for submask in range(len(x_ls)):
                x = np.array(mask['stream'].data['xs'][submask]) #x coordinates
                y = np.array(mask['stream'].data['ys'][submask]) #y coordinates
                xy = np.column_stack((x,y)).astype('uint64') #xy coordinate pairs
                cv2.fillPoly(mask['mask'], pts =[xy], color=1) #fill polygon  
            mask['mask'] = mask['mask'].astype('bool')
        return hv.Labels((0,0,""))
    
    
    make_mask_ptl = fct.partial(make_mask, mask=mask)        
    dmap = hv.DynamicMap(make_mask_ptl, streams=[mask['stream']])
    return image*poly*dmap, mask



def check_p_frames(cap, p_prop_allowed=.01, frames_checked=300):
    """ 
    -------------------------------------------------------------------------------------
    
    Checks whether video contains substantial portion of p/blank frames
    
    -------------------------------------------------------------------------------------
    Args:
        cap:: [cv2.videocapture]
            OpenCV video capture object.
        p_prop_allowed:: [numeric]
            Proportion of putative p-frames permitted.  Alternatively, proportion of 
            frames permitted to return False when grabbed.
        frames_checked:: [numeric]
            Number of frames to scan for p/blank frames.  If video is shorter
            than number of frames specified, will use number of frames in video.
    
    -------------------------------------------------------------------------------------
    Returns:
    
    -------------------------------------------------------------------------------------
    Notes:
    
    """
    
    frames_checked = min(frames_checked, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    p_allowed = int(frames_checked*p_prop_allowed)
    
    p_frms = 0
    for i in range(frames_checked):
        ret, frame = cap.read()
        p_frms = p_frms+1 if ret==False else p_frms
    if p_frms>p_allowed:
        raise RuntimeError(
            'Video compression method not supported. ' + \
            'Approximately {p}% frames are p frames or blank. '.format(
                p=(p_frms/frames_checked)*100) + \
            'Consider video conversion.')


######################################################################################## 

def _compute_kinematics(location, video_dict, fps=None, sg_window=11, sg_polyorder=3):
    """
    Robust computation of velocity and acceleration from X, Y position data
    using Savitzky-Golay filtering.

    Parameters
    ----------
    location    : dataframe with columns X, Y (and Distance, though we recompute from positions)
    video_dict  : used for scale factor if available
    fps         : frames per second of the video. If None, output is in units/frame.
                  If provided, output is in units/second and units/second².
    sg_window   : window length for Savitzky-Golay filter (must be odd, > sg_polyorder).
                  Rule of thumb: ~5-15 frames for 30fps mouse tracking.
    sg_polyorder: polynomial order for Savitzky-Golay filter. 2 or 3 is standard.

    Returns
    -------
    speed       : 1D numpy array, instantaneous speed (always >= 0)
    acceleration: 1D numpy array, signed scalar acceleration along the path
                  (positive = speeding up, negative = slowing down)
    speed_label : str, label for colorbar
    accel_label : str, label for colorbar
    """

    x = location['X'].values.astype(float)
    y = location['Y'].values.astype(float)

    # --- Scale factor ---
    if 'scale' in video_dict and video_dict['scale'] is not None:
        factor     = video_dict['scale'].get('factor', 1.0)
        unit       = video_dict['scale'].get('true_scale', 'px')
    else:
        factor = 1.0
        unit   = 'px'

    x = x * factor
    y = y * factor

    # --- Enforce sg_window is valid ---
    # window must be odd and at least polyorder + 1
    # also must not exceed the number of frames
    n_frames   = len(x)
    sg_window  = min(sg_window, n_frames if n_frames % 2 != 0 else n_frames - 1)
    sg_window  = sg_window if sg_window % 2 != 0 else sg_window - 1
    sg_window  = max(sg_window, sg_polyorder + 1 + (sg_polyorder + 1) % 2)  # ensure > polyorder and odd

    # --- Temporal unit ---
    if fps is not None and fps > 0:
        delta     = 1.0 / fps           # seconds per frame
        time_unit = 's'
    else:
        delta     = 1.0                 # 1 frame per frame (no normalization)
        time_unit = 'frame'

    # --- Smoothed first derivative of position = velocity components (px or scaled units / time) ---
    vx = savgol_filter(x, window_length=sg_window, polyorder=sg_polyorder, deriv=1, delta=delta)
    vy = savgol_filter(y, window_length=sg_window, polyorder=sg_polyorder, deriv=1, delta=delta)

    # --- Instantaneous speed (scalar, always >= 0) ---
    speed = np.sqrt(vx**2 + vy**2)

    # --- Smoothed second derivative of position = acceleration components ---
    ax = savgol_filter(x, window_length=sg_window, polyorder=sg_polyorder, deriv=2, delta=delta)
    ay = savgol_filter(y, window_length=sg_window, polyorder=sg_polyorder, deriv=2, delta=delta)

    # --- Signed scalar acceleration along the direction of motion ---
    # Project acceleration vector onto velocity unit vector.
    # Positive = speeding up along current heading, Negative = braking.
    # This is more informative than raw acceleration magnitude.
    speed_nonzero = np.where(speed > 0, speed, np.nan)   # avoid divide-by-zero at stationary frames
    accel = (ax * vx + ay * vy) / speed_nonzero          # dot product with unit tangent
    accel = np.nan_to_num(accel, nan=0.0)                 # stationary frames = 0 acceleration

    speed_label = f'Speed ({unit}/{time_unit})'
    accel_label = f'Acceleration ({unit}/{time_unit}²)'

    return speed, accel, speed_label, accel_label



########################################################################################

def VelocityTrace(video_dict, location, fps=None, sg_window=11, sg_polyorder=3, cmap='plasma', alpha=0.8, size=3, rotate=False):
    """
    -------------------------------------------------------------------------------------

    Create image where animal location across session is displayed atop reference frame,
    with each point colored by the instantaneous velocity (distance/frame) of the animal
    at that position.

    -------------------------------------------------------------------------------------
    Args:

        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proportional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to
                               selection tool. [hv.streams.stream]
                'scale' : [dict] (optional)
                        Dictionary with the following keys:
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]

        location:: [pandas.dataframe]
            Pandas dataframe with frame by frame x and y locations and distance travelled.
            Must contain columns: 'X', 'Y', 'Distance'.

        cmap:: [str]
            Matplotlib/Holoviews colormap name for velocity encoding.
            Default is 'plasma' (dark=slow, bright=fast).

        alpha:: [float]
            Alpha transparency of trace points. Between 0-1.

        size:: [float]
            Size of trace points.

    -------------------------------------------------------------------------------------
    Returns:
        holoviews.Overlay
            Animal location trace superimposed upon reference frame, with points colored
            by instantaneous velocity. Colorbar included. ROI polygons overlaid if present.

    -------------------------------------------------------------------------------------
    Notes:
        - Velocity is taken directly from the 'Distance' column (px/frame).
        - If video_dict['scale'] is defined, velocity is converted to true_scale units/frame.
        - np.gradient is NOT used here; Distance already represents frame-to-frame displacement.

    """
    speed, _, speed_label, _ = _compute_kinematics(
        location, video_dict, fps=fps,
        sg_window=sg_window, sg_polyorder=sg_polyorder
    )
    
    # Compute velocity from Distance column (px/frame by default)
    velocity = location['Distance_px'].values.astype(float)

    # Scale velocity if scale factor is available
    if 'scale' in video_dict and video_dict['scale'] is not None:
        factor = video_dict['scale'].get('factor', 1)
        vel_label = 'Velocity ({}/frame)'.format(video_dict['scale'].get('true_scale', 'px'))
        velocity = velocity * factor
    else:
        vel_label = 'Velocity (px/frame)'

    # --- rotation for display only ---
    reference = video_dict['reference']
    X_disp = location['X'].values.copy()
    Y_disp = location['Y'].values.copy()
    if rotate:
        reference, X_disp, Y_disp = _rotate_frame_and_coords(
            reference, X_disp, Y_disp, k=3)

    data = np.column_stack([X_disp, Y_disp, speed])

    # White background (no reference image)
    white_bg = np.ones_like(reference) * 255
    image = hv.Image((
        np.arange(reference.shape[1]),
        np.arange(reference.shape[0]),
        white_bg
    )).opts(
        width=int(reference.shape[1] * video_dict['stretch']['width']),
        height=int(reference.shape[0] * video_dict['stretch']['height']),
        invert_yaxis=True,
        cmap='gray',
        toolbar='below',
        title='Velocity Trace',
        bgcolor='white',
        show_frame=True
    )

    # Scatter plot colored by velocity
    points = hv.Scatter(
        data, kdims=['X', 'Y'], vdims=[vel_label]
    ).opts(
        color=vel_label,
        cmap=cmap,
        colorbar=True,
        alpha=alpha,
        size=size,
        toolbar='below'
    )

    # ROI polygons (if present)
    video_dict['roi_stream'] = video_dict.get('roi_stream', None)
    if video_dict['roi_stream'] is not None:
        lst = []
        for poly in range(len(video_dict['roi_stream'].data['xs'])):
            x = np.array(video_dict['roi_stream'].data['xs'][poly])
            y = np.array(video_dict['roi_stream'].data['ys'][poly])
            if rotate:
                _, xr, yr = _rotate_frame_and_coords(
                    video_dict['reference'], x, y, k=3)
                lst.append([(xr[v], yr[v]) for v in range(len(xr))])
            else:
                lst.append([(x[vert], y[vert]) for vert in range(len(x))])
        poly_overlay = hv.Polygons(lst).opts(fill_alpha=0.1, line_dash='dashed')
        return image * poly_overlay * points
    else:
        return image * points

########################################################################################

def AccelerationTrace(video_dict, location,fps=None, sg_window=11, sg_polyorder=3, cmap='coolwarm', alpha=0.8, size=3, rotate=False):
    """
    -------------------------------------------------------------------------------------

    Create image where animal location across session is displayed atop reference frame,
    with each point colored by the instantaneous acceleration (rate of change of velocity)
    of the animal at that position.

    Positive values (warm colors) indicate the animal is speeding up.
    Negative values (cool colors) indicate the animal is slowing down.
    Values near zero (neutral) indicate roughly constant speed.

    -------------------------------------------------------------------------------------
    Args:

        video_dict:: [dict]
            Dictionary with the following keys:
                'dpath' : directory containing files [str]
                'file' : filename with extension, e.g. 'myvideo.wmv' [str]
                'start' : frame at which to start. 0-based [int]
                'end' : frame at which to end.  set to None if processing
                        whole video [int]
                'region_names' : list of names of regions.  if no regions, set to None
                'dsmpl' : proportional degree to which video should be downsampled
                        by (0-1).
                'stretch' : Dictionary used to alter display of frames, with the following keys:
                        'width' : proportion by which to stretch frame width [float]
                        'height' : proportion by which to stretch frame height [float]
                'reference': Reference image that the current frame is compared to. [numpy.array]
                'roi_stream' : Holoviews stream object enabling dynamic selection in response to
                               selection tool. [hv.streams.stream]
                'scale' : [dict] (optional)
                        Dictionary with the following keys:
                            'factor' : ratio of desired scale to pixel (e.g. cm/pixel) [numeric]
                            'true_scale' : string containing name of scale (e.g. 'cm') [str]

        location:: [pandas.dataframe]
            Pandas dataframe with frame by frame x and y locations and distance travelled.
            Must contain columns: 'X', 'Y', 'Distance'.

        cmap:: [str]
            Matplotlib/Holoviews colormap name for acceleration encoding.
            Default is 'coolwarm': blue=decelerating, red=accelerating, white=constant speed.

        alpha:: [float]
            Alpha transparency of trace points. Between 0-1.

        size:: [float]
            Size of trace points.

    -------------------------------------------------------------------------------------
    Returns:
        holoviews.Overlay
            Animal location trace superimposed upon reference frame, with points colored
            by instantaneous acceleration. Colorbar included. ROI polygons overlaid if present.

    -------------------------------------------------------------------------------------
    Notes:
        - Velocity is derived from location['Distance'] (px/frame or scaled).
        - Acceleration is computed via np.gradient(velocity), which uses central differences
          and returns an array of the same length as the input — keeping X/Y alignment exact.
        - np.gradient is preferred over np.diff to avoid index misalignment.

    """
    
    _, accel, _, accel_label = _compute_kinematics(location, video_dict, fps=fps, sg_window=sg_window, sg_polyorder=sg_polyorder)

    # Compute velocity from Distance column
    velocity = location['Distance_px'].values.astype(float)

    # Scale if scale factor is available
    if 'scale' in video_dict and video_dict['scale'] is not None:
        factor = video_dict['scale'].get('factor', 1)
        scale_label = video_dict['scale'].get('true_scale', 'px')
        velocity = velocity * factor
        accel_label = 'Acceleration ({}/frame²)'.format(scale_label)
    else:
        accel_label = 'Acceleration (px/frame²)'

    # --- rotation for display only ---
    reference = video_dict['reference']
    X_disp = location['X'].values.copy()
    Y_disp = location['Y'].values.copy()
    if rotate:
        reference, X_disp, Y_disp = _rotate_frame_and_coords(
            reference, X_disp, Y_disp, k=3)

    data = np.column_stack([X_disp, Y_disp, accel])

    # White background (no reference image)
    white_bg = np.ones_like(reference) * 255
    image = hv.Image((
        np.arange(reference.shape[1]),
        np.arange(reference.shape[0]),
        white_bg
    )).opts(
        width=int(reference.shape[1] * video_dict['stretch']['width']),
        height=int(reference.shape[0] * video_dict['stretch']['height']),
        invert_yaxis=True,
        cmap='gray',
        toolbar='below',
        title='Acceleration Trace',
        bgcolor='white',
        show_frame=True
    )

    # Scatter plot colored by acceleration
    points = hv.Scatter(
        data, kdims=['X', 'Y'], vdims=[accel_label]
    ).opts(
        color=accel_label,
        cmap=cmap,
        colorbar=True,
        alpha=alpha,
        size=size,
        toolbar='below',
        symmetric=True   # centers colorbar at 0 so +/- are visually balanced
    )

    # ROI polygons (if present)
    video_dict['roi_stream'] = video_dict.get('roi_stream', None)
    if video_dict['roi_stream'] is not None:
        lst = []
        for poly in range(len(video_dict['roi_stream'].data['xs'])):
            x = np.array(video_dict['roi_stream'].data['xs'][poly])
            y = np.array(video_dict['roi_stream'].data['ys'][poly])
            if rotate:
                _, xr, yr = _rotate_frame_and_coords(
                    video_dict['reference'], x, y, k=3)
                lst.append([(xr[v], yr[v]) for v in range(len(xr))])
            else:
                lst.append([(x[vert], y[vert]) for vert in range(len(x))])
        poly_overlay = hv.Polygons(lst).opts(fill_alpha=0.1, line_dash='dashed')
        return image * poly_overlay * points
    else:
        return image * points

##############################
########################################################################################

def _binned_kinematic_strip(video_dict, location, metric, metric_label,
                             bin_size=10, cmap='plasma', symmetric=False,
                             rotate=False, title=''):
    X = location['X'].values.astype(float)
    Y = location['Y'].values.astype(float)

    if rotate:
        H_orig = video_dict['reference'].shape[0]
        track_axis = (H_orig - 1) - Y
    else:
        track_axis = X

    t_min, t_max = track_axis.min(), track_axis.max()
    bin_edges   = np.arange(t_min, t_max + bin_size, bin_size)
    n_bins      = len(bin_edges) - 1
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    bin_vals = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (track_axis >= bin_edges[i]) & (track_axis < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_vals[i] = metric[mask].mean()

    valid = np.isfinite(bin_vals)
    if valid.any():
        idx      = np.arange(n_bins)
        bin_vals = np.interp(idx, idx[valid], bin_vals[valid])

    # --- per-file min-max normalisation to [0, 1] ---
    v_min, v_max = bin_vals.min(), bin_vals.max()
    if v_max > v_min:
        bin_vals = (bin_vals - v_min) / (v_max - v_min)
    else:
        bin_vals = np.zeros_like(bin_vals)   # flat signal → all 0

    STRIP_H  = 40
    strip_2d = np.tile(bin_vals[np.newaxis, :], (STRIP_H, 1))
    y_coords = np.arange(STRIP_H)

    ref      = video_dict['reference']
    ref_disp = _rotate_frame_and_coords(ref, k=3) if rotate else ref
    disp_w   = int(ref_disp.shape[1] * video_dict['stretch']['width'])

    # For acceleration the normalised midpoint 0.5 = zero acceleration,
    # so we fix clim at (0, 1) and let coolwarm straddle 0.5 naturally.
    extra_opts = dict(clim=(0, 1)) if symmetric else dict(clim=(0, 1))

    normalized_label = metric_label + ' (norm.)'

    strip = hv.Image(
        (bin_centers, y_coords, strip_2d),
        kdims=['Track position (px)', 'Y'],
        vdims=[normalized_label]
    ).opts(
        width=disp_w,
        height=110,
        cmap=cmap,
        colorbar=True,
        invert_yaxis=False,
        toolbar='below',
        title=title,
        yaxis=None,
        **extra_opts
    )
    return strip


########################################################################################

def VelocityBinned(video_dict, location, bin_size=10, fps=None,
                   sg_window=11, sg_polyorder=3, cmap='plasma', rotate=False):
    """
    Identical to before but the binned values are normalised per file:
    the highest mean-speed bin = 1, lowest = 0.
    """
    speed, _, speed_label, _ = _compute_kinematics(
        location, video_dict,
        fps=fps, sg_window=sg_window, sg_polyorder=sg_polyorder
    )

    return _binned_kinematic_strip(
        video_dict, location,
        metric=speed,
        metric_label=speed_label,
        bin_size=bin_size,
        cmap=cmap,
        symmetric=False,
        rotate=rotate,
        title='Binned velocity  (bin = {} px, normalised)'.format(bin_size)
    )


########################################################################################

def AccelerationBinned(video_dict, location, bin_size=10, fps=None,
                       sg_window=11, sg_polyorder=3, cmap='coolwarm', rotate=False):
    """
    Identical to before but the binned values are normalised per file:
    max mean-acceleration bin = 1, min = 0.
    The coolwarm midpoint (0.5) therefore corresponds to the neutral
    acceleration level for that particular session.
    """
    _, accel, _, accel_label = _compute_kinematics(
        location, video_dict,
        fps=fps, sg_window=sg_window, sg_polyorder=sg_polyorder
    )

    return _binned_kinematic_strip(
        video_dict, location,
        metric=accel,
        metric_label=accel_label,
        bin_size=bin_size,
        cmap=cmap,
        symmetric=True,
        rotate=rotate,
        title='Binned acceleration  (bin = {} px, normalised)'.format(bin_size)
    )

########################################################################################

def parse_trial_windows(events_path, start_state='000', end_state='121'):
    """
    -------------------------------------------------------------------------------------

    Parse a PuTTY-logged behavioral events file and return a list of
    (t_start_ms, t_end_ms) tuples marking successful trial windows.

    A successful trial is defined as the period from the MOST RECENT occurrence of
    `start_state` up to (and including) the next occurrence of `end_state`.
    Any `start_state` hit while a trial is already open resets the trial clock.

    -------------------------------------------------------------------------------------
    Args:
        events_path :: [str]
            Path to the PuTTY .txt events log.

        start_state :: [str]
            3-character state code that opens a trial window. Default '000'.

        end_state :: [str]
            3-character state code that closes the window (e.g. solenoid opening).
            Default '121'.

    -------------------------------------------------------------------------------------
    Returns:
        windows :: [list of tuples]
            List of (t_start_ms, t_end_ms) pairs.

    -------------------------------------------------------------------------------------
    Notes:
        Line format  : [3-char state][variable-length integer timestamp in ms]
        Session start: 'S…'          → skipped entirely
        Session end  : 'E<state>0000' → state extracted; the dummy '0000' suffix is
                        replaced with the timestamp of the immediately preceding line,
                        so a session that ends mid-trial (e.g. E1210000) is captured.
        Header lines (starting with '=' or 'Linear') are also skipped.

    """
    windows  = []
    pending  = None   # t_start_ms of the currently open trial, or None
    last_ts  = None   # last valid timestamp seen (used for the E-line)

    with open(events_path, 'r') as fh:
        for raw in fh:
            line = raw.strip()

            # --- skip non-data lines ---
            if (not line
                    or line.startswith('=')
                    or line.lower().startswith('linear')
                    or line.startswith('S')):
                continue

            # --- end-of-session marker: E<state><dummy_ts> ---
            if line.startswith('E'):
                state = line[1:4]     # 'E1210000' → state = '121'
                ts    = last_ts       # ignore '0000'; use last real timestamp
            else:
                state = line[:3]
                try:
                    ts = int(line[3:])
                except ValueError:
                    continue
                last_ts = ts

            if ts is None:
                continue

            # --- match trial start / end ---
            if state == start_state:
                pending = ts                       # open (or re-open) a window

            elif state == end_state and pending is not None:
                windows.append((pending, ts))
                pending = None                     # close; wait for next start_state

    if not windows:
        print('WARNING: No {s}→{e} windows found in {p}.'.format(
            s=start_state, e=end_state, p=events_path))
    else:
        print('{n} successful trial window(s) found.'.format(n=len(windows)))

    return windows


########################################################################################

def filter_location_by_trials(location, video_dict, events_path, fps,
                               start_state='101', end_state='020',
                               t_offset_ms=0):
    """
    -------------------------------------------------------------------------------------

    Filter a location dataframe so that only frames falling inside successful trial
    windows (start_state → end_state pairs from the events log) are retained.

    Use the returned dataframe in place of the full `location` when calling
    VelocityTrace, AccelerationTrace, VelocityBinned, AccelerationBinned, Heatmap,
    or showtrace to exclude random inter-trial exploration from all kinematics and
    spatial plots.

    -------------------------------------------------------------------------------------
    Args:
        location :: [pandas.DataFrame]
            Output of TrackLocation.  Must contain column 'Frame' (0-based, as
            produced by TrackLocation).

        video_dict :: [dict]
            Standard video_dict.

        events_path :: [str]
            Path to the PuTTY events log file.

        fps :: [float]
            Video frame rate in frames / second.  Used to convert ms timestamps
            to frame numbers.

        start_state :: [str]
            Events state marking trial onset. Default '000'.

        end_state :: [str]
            Events state marking reward / solenoid opening. Default '121'.

        t_offset_ms :: [float]
            Optional shift (ms) applied to ALL event timestamps before frame
            conversion.  Use when the events-log t = 0 does not coincide with
            location Frame 0.  Default 0 (assumes both clocks start together).

    -------------------------------------------------------------------------------------
    Returns:
        filtered :: [pandas.DataFrame]
            Subset of `location` restricted to trial frames.  A helper column
            '_trial_id' (0-based integer) identifies which window each row belongs
            to.  The 'Frame' column is re-indexed from 0 across the concatenated
            segments so that Savitzky-Golay filtering in _compute_kinematics
            treats the data as contiguous.

        windows_frames :: [list of tuples]
            (f_start, f_end) pairs in the ORIGINAL Frame numbering, one per trial.
            Useful for annotating plots or further per-trial slicing.

    -------------------------------------------------------------------------------------
    Notes:
        Frame conversion used:
            frame = round( (t_ms + t_offset_ms) / 1000 × fps )

        Because TrackLocation sets Frame = 0 for the first analysed frame
        (regardless of video_dict['start']), and the events log typically starts
        at t = 0 for the same moment, no additional offset is needed by default.

        Savitzky-Golay smoothing has edge artefacts ≤ sg_window // 2 frames at the
        seam between concatenated trial segments.  For the binned heatmap plots
        (VelocityBinned / AccelerationBinned) this is negligible because values are
        averaged over many frames per spatial bin.  For the scatter trace plots
        (VelocityTrace / AccelerationTrace) the artefacts are confined to a handful
        of frames at each trial boundary.

    -------------------------------------------------------------------------------------
    Example:
        fps = 30
        loc_trials, wins = filter_location_by_trials(
            location, video_dict, 'F1_Events.txt', fps=fps)

        VelocityTrace(video_dict, loc_trials, fps=fps, rotate=True)
        AccelerationTrace(video_dict, loc_trials, fps=fps, rotate=True)
        VelocityBinned(video_dict, loc_trials, fps=fps, rotate=True)
        AccelerationBinned(video_dict, loc_trials, fps=fps, rotate=True)

    """

    windows_ms = parse_trial_windows(events_path, start_state, end_state)
    if not windows_ms:
        return location, []

    # --- convert ms timestamps → location Frame numbers ---
    windows_frames = []
    for (t0, t1) in windows_ms:
        f0 = int(round((t0 + t_offset_ms) / 1000.0 * fps))
        f1 = int(round((t1 + t_offset_ms) / 1000.0 * fps))
        windows_frames.append((f0, f1))

    # --- print summary table ---
    print('\n{:>4}  {:>14}  {:>14}  {:>13}  {}'.format(
        '#', 'start (ms)', 'end (ms)', 'duration (s)', 'frames'))
    for i, ((t0, t1), (f0, f1)) in enumerate(zip(windows_ms, windows_frames)):
        dur = (t1 - t0) / 1000.0
        print('{:>4}  {:>14,.0f}  {:>14,.0f}  {:>13.1f}  {:>6} – {:>6}'.format(
            i + 1, t0, t1, dur, f0, f1))

    # --- extract per-trial segments and tag with trial id ---
    frames   = location['Frame'].values
    segments = []
    for trial_id, (f0, f1) in enumerate(windows_frames):
        seg = location[(frames >= f0) & (frames <= f1)].copy()
        if len(seg) == 0:
            print('WARNING: Trial {} (frames {}–{}) contains no location data. '
                  'Check fps / t_offset_ms.'.format(trial_id + 1, f0, f1))
            continue
        seg['_trial_id'] = trial_id
        segments.append(seg)

    if not segments:
        print('WARNING: No location frames match any trial window. '
              'Returning unfiltered dataframe.')
        return location, windows_frames

    # --- concatenate and re-index Frame so SG filter sees contiguous data ---
    filtered         = pd.concat(segments, ignore_index=True)
    filtered['Frame'] = np.arange(len(filtered))

    pct = 100.0 * len(filtered) / len(location)
    print('\n{} / {} frames retained ({:.1f} %)  across {} trial(s).'.format(
        len(filtered), len(location), pct, len(segments)))

    return filtered, windows_frames

import re
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


########################################################################################

def _extract_animal_label(filename):
    """
    Extract animal label from filename pattern 'D_1+M1.mp4' → 'M1'.
    Splits on '+' and strips the extension from the trailing part.
    Falls back to the full file stem if no '+' is found.
    """
    if '_' in filename:
        return os.path.splitext(filename.split('_')[-1])[0]
    return os.path.splitext(filename)[0]


########################################################################################

def _extract_day_label(filename):
    """
    Extract day label from filename.
    'D1_M1.mp4'  → 'D1'
    'D2_F3.mp4'  → 'D2'
    """
    stem = os.path.splitext(filename)[0]   # 'D1_M1'
    return stem.split('_')[0]              # 'D1'


########################################################################################

def _compute_binned_strip(video_dict, location, metric, bin_size, rotate):
    """
    Compute spatially binned metric values along the track axis,
    normalised per file to [0, 1].

    Returns
    -------
    bin_centers : 1-D numpy array   (pixel positions along track)
    bin_vals    : 1-D numpy array   (normalised mean metric per bin, in [0, 1])
    """
    X = location['X'].values.astype(float)
    Y = location['Y'].values.astype(float)

    track_axis = ((video_dict['reference'].shape[0] - 1) - Y) if rotate else X

    t_min, t_max = track_axis.min(), track_axis.max()
    bin_edges    = np.arange(t_min, t_max + bin_size, bin_size)
    n_bins       = max(len(bin_edges) - 1, 1)
    bin_centers  = (bin_edges[:n_bins] + bin_edges[1:n_bins + 1]) / 2.0

    bin_vals = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (track_axis >= bin_edges[i]) & (track_axis < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_vals[i] = metric[mask].mean()

    # gap-fill isolated NaN bins
    valid = np.isfinite(bin_vals)
    if valid.any():
        bin_vals = np.interp(np.arange(n_bins), np.arange(n_bins)[valid],
                             bin_vals[valid])
    else:
        bin_vals = np.zeros(n_bins)

    # per-file min-max normalisation → [0, 1]
    v_min, v_max = bin_vals.min(), bin_vals.max()
    if v_max > v_min:
        bin_vals = (bin_vals - v_min) / (v_max - v_min)
    else:
        bin_vals = np.zeros_like(bin_vals)

    return bin_centers, bin_vals

###########################################################################################################

def _compute_binned_occupancy(video_dict, location, bin_size, rotate, sigma=None):
    """
    1-D occupancy matching the smoothing of Heatmap() exactly:
      - pixel-level frame counts (same as 2-D Heatmap but collapsed to 1-D)
      - gaussian_filter1d with sigma = track_length * 0.05  (same formula)
      - normalised to [0, 1]
      - averaged into display bins of width bin_size
    """
    X = location['X'].values.astype(float)
    Y = location['Y'].values.astype(float)

    track_axis = ((video_dict['reference'].shape[0] - 1) - Y) if rotate else X

    t_int        = np.round(track_axis).astype(int)
    t_min, t_max = t_int.min(), t_int.max()
    n_px         = t_max - t_min + 1

    # --- pixel-level occupancy counts (mirrors Heatmap's heatmap[Y,X] += 1) ---
    occ = np.zeros(n_px)
    for t in t_int:
        occ[t - t_min] += 1

    # --- Gaussian smoothing — identical sigma formula to Heatmap() ---
    if sigma is None:
        sigma = n_px * 0.05
    occ = ndimage.gaussian_filter1d(occ, sigma=sigma)

    # --- normalise to [0, 1]  (Heatmap normalises to 255, we use 1) ---
    if occ.max() > 0:
        occ = occ / occ.max()

    # --- average smoothed pixel values into display bins ---
    px_pos     = np.arange(t_min, t_max + 1, dtype=float)
    bin_edges  = np.arange(t_min, t_max + bin_size, bin_size, dtype=float)
    n_bins     = max(len(bin_edges) - 1, 1)
    bin_centers = (bin_edges[:n_bins] + bin_edges[1:n_bins + 1]) / 2.0

    bin_vals = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (px_pos >= bin_edges[i]) & (px_pos < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_vals[i] = occ[mask].mean()

    # --- final rescale after binning ---
    v_min, v_max = bin_vals.min(), bin_vals.max()
    if v_max > v_min:
        bin_vals = (bin_vals - v_min) / (v_max - v_min)

    return bin_centers, bin_vals





########################################################################################

def _plot_stacked_heatmap(rows, cmap, title,
                           xlabel='Track position (px)',
                           cbar_label='Normalised (0 – 1)',
                           row_height=0.55, fig_width=10):
    n     = len(rows)
    fig, axes = plt.subplots(
        n, 1,
        figsize=(fig_width, row_height * n),
        sharex=True
    )
    if n == 1:
        axes = [axes]

    norm    = mcolors.Normalize(vmin=0, vmax=1)
    last_im = None

    # ── NEW: build display labels ────────────────────────────────────────────
    # Show the day prefix (e.g. 'D1') only on the first strip of each day
    # group; leave subsequent strips blank.
    # Works whether labels look like 'D1_M1' (no-averaging) or 'D1' (averaged).
    display_labels = []
    prev_day = None
    for label, _, _ in rows:
        day = label.split('_')[0]      # 'D1_M1' → 'D1'  |  'D1' → 'D1'
        if day != prev_day:
            display_labels.append(day)
            prev_day = day
        else:
            display_labels.append('')  # same day → blank
    # ────────────────────────────────────────────────────────────────────────

    for ax, (label, bin_centers, bin_vals), disp_label in zip(axes, rows, display_labels):
        half   = (bin_centers[1] - bin_centers[0]) / 2 if len(bin_centers) > 1 else 5
        extent = [bin_centers[0] - half, bin_centers[-1] + half, 0, 1]

        last_im = ax.imshow(
            bin_vals[np.newaxis, :],
            aspect='auto',
            cmap=cmap,
            norm=norm,
            extent=extent,
            origin='lower',
            interpolation='bilinear'
        )

        # ── CHANGED: use disp_label instead of label ─────────────────────────
        ax.set_yticks([0.5])
        ax.set_yticklabels([disp_label], fontsize=10, fontweight='bold', va='center')
        ax.tick_params(axis='y', length=0)
        # ─────────────────────────────────────────────────────────────────────

        ax.tick_params(axis='x', bottom=False, labelbottom=False)
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(False)

    axes[-1].tick_params(axis='x', bottom=True, labelbottom=True, labelsize=9)
    axes[-1].spines['bottom'].set_visible(True)
    axes[-1].set_xlabel(xlabel, fontsize=11)

    fig.subplots_adjust(hspace=0)

    cb = fig.colorbar(last_im, ax=axes, location='right', pad=0.015, fraction=0.035)
    cb.set_label(cbar_label, fontsize=10)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.ax.tick_params(labelsize=9)

    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.005)
    return fig

########################################################################################

def VelocityStackedHeatmap(file_sessions, video_dict_template,
                            bin_size=10, fps=None,
                            sg_window=11, sg_polyorder=3,
                            rotate=True, cmap='plasma',
                            row_height=0.55, fig_width=10):
    """
    -------------------------------------------------------------------------------------

    Stacked 1-D velocity heatmap — one strip per session, shared x-axis and colorbar.

    -------------------------------------------------------------------------------------
    Args:
        file_sessions :: [list of tuples]
            List of (filename, location_df, reference_array) in the order
            they should appear top → bottom.

        video_dict_template :: [dict]
            Any video_dict from the session; used only for scale information.
            'reference' is overridden per session internally.

        bin_size :: [int]
            Spatial bin width in pixels. Default 10.

        fps :: [float or None]
            Video frame rate. If None, speed is in units/frame.

        rotate :: [bool]
            True for a horizontal linear track (bins along the Y axis of the
            original frame, which becomes the horizontal display axis).

        cmap :: [str]
            Colormap. Default 'plasma'.

        row_height :: [float]
            Height of each strip row in inches.

        fig_width :: [float]
            Total figure width in inches.

    -------------------------------------------------------------------------------------
    Returns:
        fig :: matplotlib.figure.Figure

    -------------------------------------------------------------------------------------
    Notes:
        - Each session is normalised independently: its fastest bin = 1, slowest = 0.
        - Animal label is extracted from filename as the substring after the last '_',
          e.g.  'D1_M1.mp4' → 'M1'.

    """
    rows = []
    for filename, location, reference in file_sessions:
        vd              = {k: v for k, v in video_dict_template.items()}
        vd['reference'] = reference

        speed, _, speed_label, _ = _compute_kinematics(
            location, vd,
            fps=fps, sg_window=sg_window, sg_polyorder=sg_polyorder
        )
        bin_centers, bin_vals = _compute_binned_strip(
            vd, location, speed, bin_size, rotate)
        rows.append((os.path.splitext(filename)[0], bin_centers, bin_vals))
        #rows.append((_extract_animal_label(filename), bin_centers, bin_vals))

    return _plot_stacked_heatmap(
        rows,
        cmap=cmap,
        title='Velocity  |  {} px bins  |  normalised per session'.format(bin_size),
        cbar_label='Speed (norm. 0 – 1)',
        row_height=row_height,
        fig_width=fig_width
    )


########################################################################################

def AccelerationStackedHeatmap(file_sessions, video_dict_template,
                                bin_size=10, fps=None,
                                sg_window=11, sg_polyorder=3,
                                rotate=True, cmap='coolwarm',
                                row_height=0.55, fig_width=10):
    """
    -------------------------------------------------------------------------------------

    Stacked 1-D acceleration heatmap — one strip per session, shared x-axis and colorbar.

    -------------------------------------------------------------------------------------
    Args:
        file_sessions :: [list of tuples]
            List of (filename, location_df, reference_array) in the order
            they should appear top → bottom.

        video_dict_template :: [dict]
            Any video_dict from the session; used only for scale information.
            'reference' is overridden per session internally.

        bin_size :: [int]
            Spatial bin width in pixels. Default 10.

        fps :: [float or None]
            Video frame rate. If None, acceleration is in units/frame².

        rotate :: [bool]
            True for a horizontal linear track.

        cmap :: [str]
            Diverging colormap. Default 'coolwarm': blue = relatively low,
            red = relatively high, after per-session normalisation.

        row_height :: [float]
            Height of each strip row in inches.

        fig_width :: [float]
            Total figure width in inches.

    -------------------------------------------------------------------------------------
    Returns:
        fig :: matplotlib.figure.Figure

    -------------------------------------------------------------------------------------
    Notes:
        - Each session is normalised independently: max-acceleration bin = 1, min = 0.
        - The coolwarm midpoint (0.5) therefore marks the median acceleration level
          for that particular session, not absolute zero acceleration.
        - Animal label is extracted as the substring after '+' in the filename.

    """
    rows = []
    for filename, location, reference in file_sessions:
        vd              = {k: v for k, v in video_dict_template.items()}
        vd['reference'] = reference

        _, accel, _, accel_label = _compute_kinematics(
            location, vd,
            fps=fps, sg_window=sg_window, sg_polyorder=sg_polyorder
        )
        bin_centers, bin_vals = _compute_binned_strip(
            vd, location, accel, bin_size, rotate)
        rows.append((os.path.splitext(filename)[0], bin_centers, bin_vals))
        #rows.append((_extract_animal_label(filename), bin_centers, bin_vals))

    return _plot_stacked_heatmap(
        rows,
        cmap=cmap,
        title='Acceleration  |  {} px bins  |  normalised per session'.format(bin_size),
        cbar_label='Acceleration (norm. 0 – 1)',
        row_height=row_height,
        fig_width=fig_width
    )

###########################################################################################

def OccupancyStackedHeatmap(file_sessions, video_dict_template,
                             bin_size=10, sigma=None,
                             rotate=True, cmap='jet',
                             row_height=0.55, fig_width=10):
    """
    Stacked 1-D occupancy heatmap — one strip per session.
    Smoothing and colour identical to Heatmap(); orientation is 1-D along track.
    """
    rows = []
    for filename, location, reference in file_sessions:
        vd              = {k: v for k, v in video_dict_template.items()}
        vd['reference'] = reference

        bin_centers, bin_vals = _compute_binned_occupancy(
            vd, location, bin_size, rotate, sigma=sigma)
        rows.append((os.path.splitext(filename)[0], bin_centers, bin_vals))

    return _plot_stacked_heatmap(
        rows,
        cmap=cmap,
        title='Occupancy  |  {} px bins  |  normalised per session'.format(bin_size),
        cbar_label='Time spent (norm. 0 – 1)',
        row_height=row_height,
        fig_width=fig_width
    )

###################################################################################################


def OccupancyDayAveragedHeatmap(file_sessions, video_dict_template,
                                 bin_size=10, sigma=None,
                                 rotate=True, cmap='jet',
                                 row_height=0.55, fig_width=10):
    """
    Stacked 1-D occupancy heatmap averaged across animals per day.
    Smoothing and colour identical to Heatmap(); orientation is 1-D along track.
    """
    # --- 1. per-session smoothed + normalised strips ---
    session_strips = []
    for filename, location, reference in file_sessions:
        vd              = {k: v for k, v in video_dict_template.items()}
        vd['reference'] = reference

        bin_centers, bin_vals = _compute_binned_occupancy(
            vd, location, bin_size, rotate, sigma=sigma)
        session_strips.append((_extract_day_label(filename), bin_centers, bin_vals))

    # --- 2. shared x-grid across all sessions ---
    all_centers    = np.concatenate([bc for _, bc, _ in session_strips])
    x_min, x_max   = all_centers.min(), all_centers.max()
    common_centers = np.arange(x_min, x_max + bin_size, bin_size, dtype=float)

    # --- 3. interpolate onto common grid and group by day ---
    day_strips = {}
    for day, bin_centers, bin_vals in session_strips:
        interp = np.interp(common_centers, bin_centers, bin_vals,
                           left=np.nan, right=np.nan)
        day_strips.setdefault(day, []).append(interp)

    # --- 4. average across animals + re-normalise per day ---
    rows = []
    for day, strips in day_strips.items():
        mean_vals = np.nanmean(np.vstack(strips), axis=0)
        rows.append((day, common_centers, mean_vals))

    return _plot_stacked_heatmap(
        rows,
        cmap=cmap,
        title='Occupancy  |  {} px bins  |  mean across animals  |  normalised per day'.format(bin_size),
        cbar_label='Time spent (norm. 0 – 1)',
        row_height=row_height,
        fig_width=fig_width
    )
    
#################################################################################

def VelocityDayAveragedHeatmap(file_sessions, video_dict_template,
                                bin_size=10, fps=None,
                                sg_window=11, sg_polyorder=3,
                                rotate=True, cmap='hot',
                                row_height=0.55, fig_width=10):
    """
    -------------------------------------------------------------------------------------

    Stacked 1-D velocity heatmap averaged across all animals per day.
    Produces one strip per day (D1 … D7) instead of one strip per animal/session.

    -------------------------------------------------------------------------------------
    Args:
        file_sessions :: [list of tuples]
            Same structure as for VelocityStackedHeatmap:
            (filename, location_df, reference_array).
            Sessions belonging to the same day are identified automatically
            from the filename (e.g. 'D_1+M1.mp4' → day 'D1').

        video_dict_template :: [dict]
            Any video_dict from the batch; used only for scale information.
            'reference' is overridden per session internally.

        bin_size :: [int]
            Spatial bin width in pixels. Default 10.

        fps :: [float or None]
            Video frame rate. If None, speed is in units/frame.

        rotate :: [bool]
            True for a horizontal linear track (bins along the Y axis of the
            original frame, which becomes the horizontal display axis).

        cmap :: [str]
            Colormap. Default 'plasma'.

        row_height :: [float]
            Height of each strip row in inches.

        fig_width :: [float]
            Total figure width in inches.

    -------------------------------------------------------------------------------------
    Returns:
        fig :: matplotlib.figure.Figure

    -------------------------------------------------------------------------------------
    Notes:
        Pipeline per session
        --------------------
        1. Compute instantaneous speed via _compute_kinematics (SG filter).
        2. Bin along the track axis and normalise each session independently
           to [0, 1] via _compute_binned_strip (same as VelocityStackedHeatmap).

        Day averaging
        -------------
        3. Interpolate every session onto a single shared x-grid (union of all
           bin centres) so sessions with slightly different track extents align.
        4. Average the normalised strips across all animals within each day.
        5. Re-normalise the day-mean to [0, 1] so each day's strip uses the
           full colour range regardless of how many animals contributed.

        Day order in the plot follows the order in which each day first appears
        in file_sessions (i.e. the sorted FileNames order from Batch_LoadFiles).
    """
    # --- 1. Per-session normalised strips ---
    session_strips = []   # list of (day_label, bin_centers, bin_vals)
    for filename, location, reference in file_sessions:
        vd              = {k: v for k, v in video_dict_template.items()}
        vd['reference'] = reference

        speed, _, _, _ = _compute_kinematics(
            location, vd,
            fps=fps, sg_window=sg_window, sg_polyorder=sg_polyorder
        )
        bin_centers, bin_vals = _compute_binned_strip(
            vd, location, speed, bin_size, rotate
        )
        session_strips.append((_extract_day_label(filename), bin_centers, bin_vals))

    # --- 2. Shared x-grid spanning all sessions ---
    all_centers    = np.concatenate([bc for _, bc, _ in session_strips])
    x_min, x_max   = all_centers.min(), all_centers.max()
    common_centers = np.arange(x_min, x_max + bin_size, bin_size, dtype=float)

    # --- 3. Interpolate onto common grid and group by day (insertion order kept) ---
    day_strips = {}
    for day, bin_centers, bin_vals in session_strips:
        interp = np.interp(common_centers, bin_centers, bin_vals,
                           left=np.nan, right=np.nan)
        day_strips.setdefault(day, []).append(interp)

    # --- 4. Average + re-normalise per day ---
    rows = []
    for day, strips in day_strips.items():
        mean_vals = np.nanmean(np.vstack(strips), axis=0)   # mean of [0,1] values → stays in [0,1]
        rows.append((day, common_centers, mean_vals))

    return _plot_stacked_heatmap(
        rows,
        cmap=cmap,
        title='Velocity  |  {} px bins  |  mean across animals  |  normalised per day'.format(bin_size),
        cbar_label='Speed (norm. 0 – 1)',
        row_height=row_height,
        fig_width=fig_width
    )


########################################################################################

def AccelerationDayAveragedHeatmap(file_sessions, video_dict_template,
                                    bin_size=10, fps=None,
                                    sg_window=11, sg_polyorder=3,
                                    rotate=True, cmap='coolwarm',
                                    row_height=0.55, fig_width=10):
    """
    -------------------------------------------------------------------------------------

    Stacked 1-D acceleration heatmap averaged across all animals per day.
    Produces one strip per day (D1 … D7) instead of one strip per animal/session.

    -------------------------------------------------------------------------------------
    Args:
        (identical structure to VelocityDayAveragedHeatmap — see that docstring)

        cmap :: [str]
            Diverging colormap. Default 'coolwarm': blue = relatively low,
            red = relatively high, after per-day re-normalisation.

    -------------------------------------------------------------------------------------
    Returns:
        fig :: matplotlib.figure.Figure

    -------------------------------------------------------------------------------------
    Notes:
        Same four-step pipeline as VelocityDayAveragedHeatmap, but using the
        signed scalar acceleration from _compute_kinematics instead of speed.
        After averaging and re-normalising, the coolwarm midpoint (0.5) marks
        the median acceleration level for that day — not absolute zero.
    """
    # --- 1. Per-session normalised strips ---
    session_strips = []
    for filename, location, reference in file_sessions:
        vd              = {k: v for k, v in video_dict_template.items()}
        vd['reference'] = reference

        _, accel, _, _ = _compute_kinematics(
            location, vd,
            fps=fps, sg_window=sg_window, sg_polyorder=sg_polyorder
        )
        bin_centers, bin_vals = _compute_binned_strip(
            vd, location, accel, bin_size, rotate
        )
        session_strips.append((_extract_day_label(filename), bin_centers, bin_vals))

    # --- 2. Shared x-grid ---
    all_centers    = np.concatenate([bc for _, bc, _ in session_strips])
    x_min, x_max   = all_centers.min(), all_centers.max()
    common_centers = np.arange(x_min, x_max + bin_size, bin_size, dtype=float)

    # --- 3. Interpolate and group by day ---
    day_strips = {}
    for day, bin_centers, bin_vals in session_strips:
        interp = np.interp(common_centers, bin_centers, bin_vals,
                           left=np.nan, right=np.nan)
        day_strips.setdefault(day, []).append(interp)

    # --- 4. Average + re-normalise per day ---
    rows = []
    for day, strips in day_strips.items():
        mean_vals = np.nanmean(np.vstack(strips), axis=0)   # mean of [0,1] values → stays in [0,1]
        rows.append((day, common_centers, mean_vals))

    return _plot_stacked_heatmap(
        rows,
        cmap=cmap,
        title='Acceleration  |  {} px bins  |  mean across animals  |  normalised per day'.format(bin_size),
        cbar_label='Acceleration (norm. 0 – 1)',
        row_height=row_height,
        fig_width=fig_width
    )


########################################################################################

def VelocityDayLinePlot(file_sessions, video_dict_template,
                         bin_size=10, fps=None,
                         sg_window=11, sg_polyorder=3,
                         rotate=True,
                         cmap='tab10',
                         fig_width=10, fig_height=5,
                         save_path=None):
    """
    -------------------------------------------------------------------------------------

    Line plot of spatially-binned velocity, averaged across animals per day,
    with SEM shading. One line per day, shared x/y axes.

    -------------------------------------------------------------------------------------
    Args:
        file_sessions :: [list of tuples]
            (filename, location_df, reference_array) — same structure as
            VelocityStackedHeatmap.

        video_dict_template :: [dict]
            Any session video_dict; 'reference' is overridden per session.

        bin_size :: [int]
            Spatial bin width in pixels. Default 10.

        fps :: [float or None]
            Frame rate. If None, speed is in units/frame.

        sg_window :: [int]
            Savitzky-Golay window length (must be odd). Default 11.

        sg_polyorder :: [int]
            Savitzky-Golay polynomial order. Default 3.

        rotate :: [bool]
            True for a horizontal linear track (bins along Y axis of original
            frame → horizontal display axis). Must match what was used for the
            stacked heatmaps.

        cmap :: [str]
            Matplotlib qualitative colormap for day lines. Default 'tab10'.

        fig_width, fig_height :: [float]
            Figure dimensions in inches.

        save_path :: [str or None]
            Full path including extension to save the figure
            (e.g. '/data/velocity_days.png', '.svg', '.pdf').
            If None, figure is not saved. Format is inferred from the extension.

    -------------------------------------------------------------------------------------
    Returns:
        fig :: matplotlib.figure.Figure

    -------------------------------------------------------------------------------------
    Notes:
        Pipeline
        --------
        1. Per session: compute instantaneous speed via _compute_kinematics,
           then bin + normalise to [0, 1] via _compute_binned_strip
           (identical to VelocityDayAveragedHeatmap).
        2. All sessions are interpolated onto a single shared x-grid.
        3. Per day: mean and SEM are computed across animals.
           SEM = std / sqrt(n) using only non-NaN contributing animals per bin.
        4. Each day is drawn as a solid line (mean) with a translucent SEM band.

    """

    # --- 1. Per-session normalised strips -----------------------------------
    session_strips = []   # list of (day_label, bin_centers, bin_vals)
    for filename, location, reference in file_sessions:
        vd              = {k: v for k, v in video_dict_template.items()}
        vd['reference'] = reference

        speed, _, _, _ = _compute_kinematics(
            location, vd,
            fps=fps, sg_window=sg_window, sg_polyorder=sg_polyorder
        )
        bin_centers, bin_vals = _compute_binned_strip(
            vd, location, speed, bin_size, rotate
        )
        session_strips.append((_extract_day_label(filename), bin_centers, bin_vals))

    # --- 2. Common x-grid spanning all sessions -----------------------------
    all_centers    = np.concatenate([bc for _, bc, _ in session_strips])
    x_min, x_max   = all_centers.min(), all_centers.max()
    common_centers = np.arange(x_min, x_max + bin_size, bin_size, dtype=float)

    # --- 3. Interpolate onto common grid and group by day -------------------
    day_strips = {}   # {day_label: [strip_animal1, strip_animal2, ...]}
    for day, bin_centers, bin_vals in session_strips:
        interp = np.interp(
            common_centers, bin_centers, bin_vals,
            left=np.nan, right=np.nan
        )
        day_strips.setdefault(day, []).append(interp)

    # --- 4. Mean + SEM per day, then plot -----------------------------------
    days    = list(day_strips.keys())
    n_days  = len(days)
    cmap_fn = plt.get_cmap(cmap)
    colors  = [cmap_fn(i / max(n_days - 1, 1)) for i in range(n_days)]

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for (day, strips), color in zip(day_strips.items(), colors):
        mat      = np.vstack(strips)                             # (n_animals, n_bins)
        mean_v   = np.nanmean(mat, axis=0)
        n_valid  = (~np.isnan(mat)).sum(axis=0).astype(float)
        sem_v    = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(np.maximum(n_valid, 1))

        ax.plot(
            common_centers, mean_v,
            color=color, linewidth=2, label=day
        )
        ax.fill_between(
            common_centers,
            mean_v - sem_v,
            mean_v + sem_v,
            color=color, alpha=0.20
        )

    # --- axes cosmetics -----------------------------------------------------
    ax.set_xlabel('Track position (px)', fontsize=12)
    ax.set_ylabel('Velocity (norm. 0 – 1)', fontsize=12)
    ax.set_title(
        'Mean velocity along track per day  ±  SEM\n'
        '({} px bins, normalised per session)'.format(bin_size),
        fontsize=12, fontweight='bold'
    )
    ax.set_ylim(0, 1)
    ax.legend(title='Day', fontsize=10, title_fontsize=10,
              framealpha=0.8, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=10)

    fig.tight_layout()

    # --- save ---------------------------------------------------------------
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print('Figure saved → {}'.format(save_path))

    return fig



########################################################################################

def AccelerationDayLinePlot(file_sessions, video_dict_template,
                         bin_size=10, fps=None,
                         sg_window=11, sg_polyorder=3,
                         rotate=True,
                         cmap='tab10',
                         fig_width=10, fig_height=5,
                         save_path=None):
    """
    -------------------------------------------------------------------------------------

    Line plot of spatially-binned velocity, averaged across animals per day,
    with SEM shading. One line per day, shared x/y axes.

    -------------------------------------------------------------------------------------
    Args:
        file_sessions :: [list of tuples]
            (filename, location_df, reference_array) — same structure as
            VelocityStackedHeatmap.

        video_dict_template :: [dict]
            Any session video_dict; 'reference' is overridden per session.

        bin_size :: [int]
            Spatial bin width in pixels. Default 10.

        fps :: [float or None]
            Frame rate. If None, speed is in units/frame.

        sg_window :: [int]
            Savitzky-Golay window length (must be odd). Default 11.

        sg_polyorder :: [int]
            Savitzky-Golay polynomial order. Default 3.

        rotate :: [bool]
            True for a horizontal linear track (bins along Y axis of original
            frame → horizontal display axis). Must match what was used for the
            stacked heatmaps.

        cmap :: [str]
            Matplotlib qualitative colormap for day lines. Default 'tab10'.

        fig_width, fig_height :: [float]
            Figure dimensions in inches.

        save_path :: [str or None]
            Full path including extension to save the figure
            (e.g. '/data/velocity_days.png', '.svg', '.pdf').
            If None, figure is not saved. Format is inferred from the extension.

    -------------------------------------------------------------------------------------
    Returns:
        fig :: matplotlib.figure.Figure

    -------------------------------------------------------------------------------------
    Notes:
        Pipeline
        --------
        1. Per session: compute instantaneous speed via _compute_kinematics,
           then bin + normalise to [0, 1] via _compute_binned_strip
           (identical to VelocityDayAveragedHeatmap).
        2. All sessions are interpolated onto a single shared x-grid.
        3. Per day: mean and SEM are computed across animals.
           SEM = std / sqrt(n) using only non-NaN contributing animals per bin.
        4. Each day is drawn as a solid line (mean) with a translucent SEM band.

    """

    # --- 1. Per-session normalised strips -----------------------------------
    session_strips = []   # list of (day_label, bin_centers, bin_vals)
    for filename, location, reference in file_sessions:
        vd              = {k: v for k, v in video_dict_template.items()}
        vd['reference'] = reference

        _, accel, _, _ = _compute_kinematics(
            location, vd,
            fps=fps, sg_window=sg_window, sg_polyorder=sg_polyorder
        )
        bin_centers, bin_vals = _compute_binned_strip(
            vd, location, accel, bin_size, rotate
        )
        session_strips.append((_extract_day_label(filename), bin_centers, bin_vals))

    # --- 2. Common x-grid spanning all sessions -----------------------------
    all_centers    = np.concatenate([bc for _, bc, _ in session_strips])
    x_min, x_max   = all_centers.min(), all_centers.max()
    common_centers = np.arange(x_min, x_max + bin_size, bin_size, dtype=float)

    # --- 3. Interpolate onto common grid and group by day -------------------
    day_strips = {}   # {day_label: [strip_animal1, strip_animal2, ...]}
    for day, bin_centers, bin_vals in session_strips:
        interp = np.interp(
            common_centers, bin_centers, bin_vals,
            left=np.nan, right=np.nan
        )
        day_strips.setdefault(day, []).append(interp)

    # --- 4. Mean + SEM per day, then plot -----------------------------------
    days    = list(day_strips.keys())
    n_days  = len(days)
    cmap_fn = plt.get_cmap(cmap)
    colors  = [cmap_fn(i / max(n_days - 1, 1)) for i in range(n_days)]

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for (day, strips), color in zip(day_strips.items(), colors):
        mat      = np.vstack(strips)                             # (n_animals, n_bins)
        mean_a   = np.nanmean(mat, axis=0)
        n_valid  = (~np.isnan(mat)).sum(axis=0).astype(float)
        sem_a    = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(np.maximum(n_valid, 1))

        ax.plot(
            common_centers, mean_a,
            color=color, linewidth=2, label=day
        )
        ax.fill_between(
            common_centers,
            mean_a - sem_a,
            mean_a + sem_a,
            color=color, alpha=0.20
        )

    # --- axes cosmetics -----------------------------------------------------
    ax.set_xlabel('Track position (px)', fontsize=12)
    ax.set_ylabel('Acceleration (norm. 0 – 1)', fontsize=12)
    ax.set_title(
        'Mean acceleration along track per day  ±  SEM\n'
        '({} px bins, normalised per session)'.format(bin_size),
        fontsize=12, fontweight='bold'
    )
    ax.set_ylim(0, 1)
    ax.legend(title='Day', fontsize=10, title_fontsize=10,
              framealpha=0.8, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=10)

    fig.tight_layout()

    # --- save ---------------------------------------------------------------
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print('Figure saved → {}'.format(save_path))

    return fig
    
#Code to export svg
#conda install -c conda-forge selenium phantomjs

#import os
#from bokeh import models
#from bokeh.io import export_svgs

#bokeh_obj = hv.renderer('bokeh').get_plot(image).state
#bokeh_obj.output_backend = 'svg'
#export_svgs(bokeh_obj, dpath + '/' + 'Calibration_Frame.svg')