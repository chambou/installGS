#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Useful functions for different purposes
"""
import numpy as np
from PIL import Image
import scipy.ndimage
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import time
from numpy.fft import fft2,ifft2,fftshift
from scipy.optimize import leastsq
import scipy as scp
import os
from scipy.fft import dctn, idctn
from hcipy import *
from astropy.io import fits

#%% ====================================================================
#   ==================== USEFUL FUNCTIONS ==============================
#   ====================================================================

def makePupil(Rpx,nPx):
    
    # Cicular pupil
    pupil = np.zeros((nPx,nPx))
    for i in range(0,nPx):
        for j in range(0,nPx):
            if np.sqrt((i-nPx/2+0.5)**2+(j-nPx/2+0.5)**2)<Rpx:
                pupil[i,j]=1
                
    return pupil
    
def transformation_dm2wfs(alpha,beta,theta,x_wfs_0,y_wfs_0,x_dm_0,y_dm_0,xx_dm,yy_dm):
        # ---- Tramsformation ------
        xx_wfs = np.cos(theta)*alpha*(xx_dm-x_dm_0)-np.sin(theta)*beta*(yy_dm-y_dm_0)+x_wfs_0
        yy_wfs = -(np.sin(theta)*alpha*(xx_dm-x_dm_0)+np.cos(theta)*beta*(yy_dm-y_dm_0))+y_wfs_0
        
        return xx_wfs,yy_wfs
        
def gauss2d(A,x0,y0,sigma,n):
    # ------------ MASK -------------
    X = np.round(np.linspace(0,n-1,n))
    [x,y] = np.meshgrid(X,X)
    g = A*np.exp(-(1/(2*sigma**2))*((x-x0)**2+(y-y0)**2))
        
    return g

def build_poke_matrix_wfs(xx_wfs,yy_wfs,A,sigma,nPx,pupil):
        poke_matrix = np.zeros((nPx**2,xx_wfs.shape[0]))
        for k in range(0,xx_wfs.shape[0]):
            poke = gauss2d(A,xx_wfs[k],yy_wfs[k],sigma,nPx)
            poke = poke*pupil
            poke_matrix[:,k] = np.ravel(poke)
    
        return poke_matrix
    
def error_rms(A,B,pupil=None):
    if pupil is None:
        pupil = np.ones(A.shape)
    err = np.sqrt(np.sum(np.sum((A*pupil-B*pupil)**2)))/np.sqrt(np.sum(pupil))
    return err

def fig(mat,vmin=None,vmax=None):
   plt.figure()
   if vmin is not None and vmax is not None:
      plt.imshow(mat,vmin=vmin,vmax=vmax)
   elif vmin is not None and vmax is None:
      plt.imshow(mat,vmin=vmin)
   elif vmin is None and vmax is not None:
      plt.imshow(mat,vmax=vmax)
   else:
      plt.imshow(mat)
   plt.colorbar()
   plt.show(block=False)


#%% ====================================================================
#   ================ TAKE off mask pupil on PWFS =======================
#   ====================================================================	

def take_image_pupils(cam,tt_mirror,amp):
    img = 0
    for k in range(0,4):
        if k == 0:
            tt_mirror.move([amp,amp])
        elif k == 1:
            tt_mirror.move([-amp,amp])
        elif k == 2:
            tt_mirror.move([-amp,-amp])
        elif k == 3:
            tt_mirror.move([amp,-amp])
        time.sleep(1)
        img = img+cam.get()
    tt_mirror.moveRef()
    return img
  

def zernikeBasis(N,nPx):
    """ Define a Zernike Basis using HCIpy function (faster than previous function)"""
    pupil_grid = make_pupil_grid(nPx,10) # Diameter size = 10m but doesn't matter
    a=make_zernike_basis(N,10,pupil_grid)
    zernike = a.transformation_matrix
    return zernike

#%% ====================================================================
#   ============= Custom FFT to be centered on 4 pixels ================
#   ====================================================================

############# CHANGE FFT DEFINITION TO FIX ISSUE IN PWFS CODE ###########
def fft_centre(X):
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(X)))

def ifft_centre(X):
    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(X)))

#%% ====================================================================
#   ============== LOCATE PUPIL in WFS images     =====================
#   ====================================================================

def findPWFS_param(img):
    # -------------
    #  - Use img (cumulative img of PWFS camera with PSF displaced in each quasrant)
    # -------------
    
    # --------------- Track -----------------------------
    nPx_img = img.shape[0]
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 6)
    plt.subplots_adjust(left=0.3,right=0.7,bottom=0.15)
    #ax.axis([0, nPx_img, 0, nPx_img])
    ax.set_aspect("equal")
    plt.imshow(img)
    plt.title('Find pupils positions')
    #======================= SLIDERS ==============================
    axcolor = 'skyblue'
    # --- LEFT TOP PUPIL -----
    sl_lt_1 = plt.axes([0.05, 0.25, 0.15, 0.03], facecolor=axcolor)
    slider_lt_1 = Slider(sl_lt_1 , 'X', 0.0*nPx_img, 0.5*nPx_img, 0.2922*nPx_img)# big pupil 0.2908
    sl_lt_2 = plt.axes([0.05, 0.2, 0.15, 0.03], facecolor=axcolor)
    slider_lt_2 = Slider(sl_lt_2, 'Y', 0.5*nPx_img, nPx_img, 0.7481*nPx_img) # 0.7258
    # --- LEFT BOTTOM PUPIL -----
    sl_lb_1 = plt.axes([0.05, 0.8, 0.15, 0.03], facecolor=axcolor)
    slider_lb_1 = Slider(sl_lb_1, 'X', 0.0*nPx_img, 0.5*nPx_img, 0.2956*nPx_img) #0.2958
    sl_lb_2 = plt.axes([0.05, 0.75, 0.15, 0.03], facecolor=axcolor)
    slider_lb_2 = Slider(sl_lb_2 , 'Y', 0.0*nPx_img, 0.5*nPx_img, 0.3133*nPx_img) # 0.2958
    # --- RIGHT TOP PUPIL -----
    sl_rt_1 = plt.axes([0.75, 0.25, 0.15, 0.03], facecolor=axcolor)
    slider_rt_1 = Slider(sl_rt_1, 'X', 0.5*nPx_img, nPx_img, 0.7372*nPx_img) #0.7292
    sl_rt_2 = plt.axes([0.75, 0.2, 0.15, 0.03], facecolor=axcolor)
    slider_rt_2 = Slider(sl_rt_2 , 'Y', 0.5*nPx_img, nPx_img, 0.7498*nPx_img) #0.7292
    # --- RIGHT BOTTOM PUPIL -----
    sl_rb_1 = plt.axes([0.75, 0.8, 0.15, 0.03], facecolor=axcolor)
    slider_rb_1 = Slider(sl_rb_1, 'X', 0.5*nPx_img, nPx_img, 0.7422*nPx_img) #0.7358
    sl_rb_2 = plt.axes([0.75, 0.75, 0.15, 0.03], facecolor=axcolor)
    slider_rb_2 = Slider(sl_rb_2, 'Y', 0.0*nPx_img, 0.5*nPx_img, 0.3158*nPx_img) #0.2942
    # ---- Pupils Diameter -----
    start_radius = 0.1792*nPx_img #0.2008
    sl3 = plt.axes([0.35, 0.05, 0.3, 0.03], facecolor=axcolor)
    slider_r = Slider(sl3, 'Radius', 0.01*nPx_img, 0.25*nPx_img,start_radius)
    
    #======================= CIRCLES ==============================
    circ_lt = plt.Circle((0.2922*nPx_img,0.7481*nPx_img),start_radius,facecolor = [0,0,0,0],ec="w")
    circ_lb = plt.Circle((0.2956*nPx_img,0.3133*nPx_img),start_radius,facecolor = [0,0,0,0],ec="w")
    
    circ_rt = plt.Circle((0.7372*nPx_img,0.7498*nPx_img),start_radius,facecolor = [0,0,0,0],ec="w")
    circ_rb = plt.Circle((0.7422*nPx_img,0.315*nPx_img),start_radius,facecolor = [0,0,0,0],ec="w")
    
    ax.add_patch(circ_lt)
    ax.add_patch(circ_lb)
    ax.add_patch(circ_rt)
    ax.add_patch(circ_rb)
    
    #======================= PLOT ==============================
    def update(val):
        # ----- UPDATE X and Y position -----
        r_lt_1 = slider_lt_1.val
        r_lt_2 = slider_lt_2.val
        r_lb_1 = slider_lb_1.val
        r_lb_2 = slider_lb_2.val
        r_rt_1 = slider_rt_1.val
        r_rt_2 = slider_rt_2.val
        r_rb_1 = slider_rb_1.val
        r_rb_2 = slider_rb_2.val
        circ_lt.center = r_lt_1 , r_lt_2
        circ_lb.center = r_lb_1 , r_lb_2
        circ_rt.center = r_rt_1 , r_rt_2
        circ_rb.center = r_rb_1 , r_rb_2  
        # ----- UPDATE Radius -----
        circ_lt.set_radius(slider_r.val) 
        circ_lb.set_radius(slider_r.val) 
        circ_rt.set_radius(slider_r.val) 
        circ_rb.set_radius(slider_r.val) 
        fig.canvas.draw_idle()
        
    slider_lt_1.on_changed(update)
    slider_lt_2.on_changed(update)
    slider_lb_1.on_changed(update)
    slider_lb_2.on_changed(update)
    slider_rt_1.on_changed(update)
    slider_rt_2.on_changed(update)
    slider_rb_1.on_changed(update)
    slider_rb_2.on_changed(update)
    slider_r.on_changed(update)
    
    plt.show(block=True)
    
    r_lt_1 = int(slider_lt_1.val)
    r_lt_2 = int(slider_lt_2.val)
    r_lb_1 = int(slider_lb_1.val)
    r_lb_2 = int(slider_lb_2.val)
    r_rt_1 = int(slider_rt_1.val)
    r_rt_2 = int(slider_rt_2.val)
    r_rb_1 = int(slider_rb_1.val)
    r_rb_2 = int(slider_rb_2.val)
    r = int(slider_r.val)
    
    #  ----- BUILD PYRAMID MASK --------
    nPx_pup = 2*r
    x_center = int(nPx_img/2+1)#nPx_pup*shannon
    y_center = int(nPx_img/2+1)#nPx_pup*shannon
    
    # angle for all faces
    position_pup = np.zeros((8,1))
    # left top pupil -----
    position_pup[0] = (x_center-r_lt_1)/nPx_pup 
    position_pup[1] = (y_center-r_lt_2)/nPx_pup
    # left bottom pupil -----
    position_pup[2] = (x_center-r_lb_1)/nPx_pup 
    position_pup[3] = (y_center-r_lb_2)/nPx_pup 
    # right top pupil -----
    position_pup[4] = (x_center-r_rt_1)/nPx_pup 
    position_pup[5] = (y_center-r_rt_2)/nPx_pup 
    # right bottom pupil -----
    position_pup[6] = (x_center-r_rb_1)/nPx_pup 
    position_pup[7] = (y_center-r_rb_2)/nPx_pup 

    pup_lt = img[r_lt_2-r+1:r_lt_2+r+1,r_lt_1-r+1:r_lt_1+r+1]
    pup_lb = img[r_lb_2-r+1:r_lb_2+r+1,r_lb_1-r+1:r_lb_1+r+1]
    pup_rt = img[r_rt_2-r+1:r_rt_2+r+1,r_rt_1-r+1:r_rt_1+r+1]
    pup_rb = img[r_lb_2-r+1:r_lb_2+r+1,r_lb_1-r+1:r_lb_1+r+1]
    pupil = 1/4*(pup_lt+pup_lb+pup_rt+pup_lt+pup_rb)
    
    # Remove outside pixel
    pupil_footprint = makePupil(int(pupil.shape[0]/2),pupil.shape[0])
    pupil = pupil*pupil_footprint
    
    plt.figure()
    plt.imshow(pupil)
    plt.colorbar()
    plt.show(block=False)
    
    return pupil,position_pup


def maskFineTuning(img,PWFS):
    # -------------
    #  - Use img (cumulative img of PWFS camera with PSF displaced in each quasrant)
    # -------------
    
    # --------------- Track -----------------------------
    nPx_img = img.shape[0]
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 6)
    plt.subplots_adjust(left=0.3,right=0.7,bottom=0.15)
    ax.set_aspect("equal")
    imageDiff = plt.imshow(PWFS.img0-img)
    plt.title('Adjust PWFS mask angles')
    #======================= SLIDERS ==============================
    axcolor = 'skyblue'
    # --- LEFT TOP PUPIL -----
    sl_lt_1 = plt.axes([0.05, 0.25, 0.15, 0.03], facecolor=axcolor)
    slider_lt_1 = Slider(sl_lt_1 , 'X', 0.7, 1.3, 1)
    sl_lt_2 = plt.axes([0.05, 0.2, 0.15, 0.03], facecolor=axcolor)
    slider_lt_2 = Slider(sl_lt_2, 'Y', 0.7, 1.3, 1)
    # --- LEFT BOTTOM PUPIL -----
    sl_lb_1 = plt.axes([0.05, 0.8, 0.15, 0.03], facecolor=axcolor)
    slider_lb_1 = Slider(sl_lb_1, 'X', 0.7, 1.3, 1)
    sl_lb_2 = plt.axes([0.05, 0.75, 0.15, 0.03], facecolor=axcolor)
    slider_lb_2 = Slider(sl_lb_2 , 'Y', 0.7, 1.3, 1)
    # --- RIGHT TOP PUPIL -----
    sl_rt_1 = plt.axes([0.75, 0.25, 0.15, 0.03], facecolor=axcolor)
    slider_rt_1 = Slider(sl_rt_1, 'X', 0.7, 1.3, 1)
    sl_rt_2 = plt.axes([0.75, 0.2, 0.15, 0.03], facecolor=axcolor)
    slider_rt_2 = Slider(sl_rt_2 , 'Y', 0.7, 1.3, 1)
    # --- RIGHT BOTTOM PUPIL -----
    sl_rb_1 = plt.axes([0.75, 0.8, 0.15, 0.03], facecolor=axcolor)
    slider_rb_1 = Slider(sl_rb_1, 'X', 0.7, 1.3, 1)
    sl_rb_2 = plt.axes([0.75, 0.75, 0.15, 0.03], facecolor=axcolor)
    slider_rb_2 = Slider(sl_rb_2, 'Y', 0.7, 1.3, 1)

    #======================= PLOT ==============================
    def update(val):
        # ----- UPDATE X and Y position -----
        r_lt_1 = slider_lt_1.val
        r_lt_2 = slider_lt_2.val
        r_lb_1 = slider_lb_1.val
        r_lb_2 = slider_lb_2.val
        r_rt_1 = slider_rt_1.val
        r_rt_2 = slider_rt_2.val
        r_rb_1 = slider_rb_1.val
        r_rb_2 = slider_rb_2.val
        PWFS.offset_angle = np.array([r_lt_1,r_lt_2,r_lb_1,r_lb_2,r_rt_1,r_rt_2,r_rb_1,r_rb_2])
        new_mask = PWFS.build_mask(PWFS.position_pup)
        PWFS.changeMask(new_mask)
        imageDiff.set_data(PWFS.img0-img)
        #plt.imshow(PWFS.img0-img)
        fig.canvas.draw_idle()
        
    slider_lt_1.on_changed(update)
    slider_lt_2.on_changed(update)
    slider_lb_1.on_changed(update)
    slider_lb_2.on_changed(update)
    slider_rt_1.on_changed(update)
    slider_rt_2.on_changed(update)
    slider_rb_1.on_changed(update)
    slider_rb_2.on_changed(update)
    
    plt.show(block=True)
    
    
    return PWFS.offset_angle

def extract_pupil(img,position_pup):
    nPx_img = img.shape[0]
    r = int(0.1792*nPx_img)
    nPx_pup = 2*r
    x_center = int(nPx_img/2+1)#nPx_pup*shannon
    y_center = int(nPx_img/2+1)#nPx_pup*shannon
    # left top pupil -----
    r_lt_1 = int(x_center-nPx_pup*position_pup[0])
    r_lt_2 = int(y_center-nPx_pup*position_pup[1])
    # left bottom pupil -----
    r_lb_1 = int(x_center-nPx_pup*position_pup[2])
    r_lb_2 = int(y_center-nPx_pup*position_pup[3])
    # right top pupil -----
    r_rt_1 = int(x_center-nPx_pup*position_pup[4])
    r_rt_2 = int(y_center-nPx_pup*position_pup[5])
    # right bottom pupil -----
    r_rb_1 = int(x_center-nPx_pup*position_pup[6])
    r_rb_2 = int(y_center-nPx_pup*position_pup[7])
    # pupil crop
    pup_lt = img[r_lt_2-r+1:r_lt_2+r+1,r_lt_1-r+1:r_lt_1+r+1]
    pup_lb = img[r_lb_2-r+1:r_lb_2+r+1,r_lb_1-r+1:r_lb_1+r+1]
    pup_rt = img[r_rt_2-r+1:r_rt_2+r+1,r_rt_1-r+1:r_rt_1+r+1]
    pup_rb = img[r_lb_2-r+1:r_lb_2+r+1,r_lb_1-r+1:r_lb_1+r+1]
    pupil = 1/4*(pup_lt+pup_lb+pup_rt+pup_lt+pup_rb)
    # Remove outside pixel
    pupil_footprint = makePupil(int(pupil.shape[0]/2),pupil.shape[0])
    pupil = pupil*pupil_footprint
    return pupil

#%% ====================================================================
#   ============  Actuators registration with Pyramid WFS===============
#   ====================================================================


def build_pokeMatrix(wfs,dm,nRecPoke = None):
    """
    This fonction allows to register DM actuators w.r.t to reconstructed phase maps
     - Takes push-pull of 2 actuators and fit gaussian to determine position
     - Take waffle image and other pokes images to check results
    """
    #------------------------------ PARAMETERS --------------------------------------------
    if nRecPoke is None:
        nRecPoke = wfs.nIterRec # NUMBER OF ITERATION to reconstruct 2 main pokes used for registration
    nRec_offset = 5 # for other pokes
    nRec_waffle = 5 # FOR WAFFLE
    threshold_act = 10 # to select actuators only in a circle in percent of diameter
    amp = 0.05 # AMPLITUDE POKE
    amp_waffle = 0.2 # AMPLITUDE WAFFLE
    x_dm_poke = np.array([15,35,35,15])
    y_dm_poke = np.array([15,35,15,35])
    number_valid_actuators = np.copy(dm.valid_actuators_number)
    # -------------------------------------------------------------------------------------
    
    
    #%% =========================== POKES ================================
    
    # Send poke, save signal, reconstruct, and fit with Gaussian
    poke = np.zeros((wfs.nPx,wfs.nPx,x_dm_poke.shape[0]))
    # -------  RECORD POKES ------
    for i in range(0,x_dm_poke.shape[0]):
        # ---- Send poke -------
        dm.pokeAct(amp,[x_dm_poke[i],y_dm_poke[i]])
        time.sleep(1)
        img_push = wfs.getImage()
        img_push[img_push<0] = 0
        dm.pokeAct(-amp,[x_dm_poke[i],y_dm_poke[i]])
        time.sleep(1)
        img_pull = wfs.getImage()
        img_pull[img_pull<0] = 0
        dm.setFlatSurf()
        # ---- Reconstruct poke -------
        print('Reconstruction of the poke number: ',i)
        if i<2:
            nRec = nRecPoke
        else:
            nRec = nRec_offset
            
        wfs.reconNonLinear(img_push,nRec)
        push = wfs.phase_rec
        wfs.reconNonLinear(img_pull,nRec)
        pull = wfs.phase_rec
        poke[:,:,i] = (push-pull)/(2*amp)
        
    # ------- Gaussian Fit 2 first pokes ------
    poke_rec = np.zeros((wfs.nPx,wfs.nPx,2))
    A_poke = []
    pos_act_x = []
    pos_act_y = []
    sigma_poke = []
    for i in range(0,2):
        # ----- FITTING A GAUSSIAN on first 2 pokes---------
        nPx = poke.shape[0]
        func = lambda param: np.ravel(gauss2d(param[0],param[1],param[2],param[3],nPx)-poke[:,:,i])
        if i == 0:
            post_x_guess = nPx/2
            post_y_guess = nPx/2
        elif i == 1:
            post_x_guess = nPx/4
            post_y_guess = nPx/4
        param0 = np.array([-20,post_x_guess,post_y_guess,10])
        param_poke = leastsq(func,param0)
        A_poke.append(param_poke[0][0])
        pos_act_x.append(param_poke[0][1])
        pos_act_y.append(param_poke[0][2])
        sigma_poke.append(param_poke[0][3])
        poke_rec[:,:,i] = gauss2d(A_poke[i],pos_act_x[i],pos_act_y[i],sigma_poke[i],nPx)
    
    print(A_poke)
    # --- Show Reconstructed poke and Gaussian Fit results --------
    plt.figure()
    plt.subplot(121)
    plt.imshow(poke[:,:,0]+poke[:,:,1])
    plt.subplot(122)
    plt.imshow(poke_rec[:,:,0]+poke_rec[:,:,1])
    plt.show(block=True)
    
    #%% ======================== WAFFLE MODE TO CHECK GRID ================================
    # ----- Send WAFFLE ------------
    if nPx > 140:
        dm.pokeWaffle(amp_waffle)
        time.sleep(1)
        img_push = wfs.getImage()
        dm.pokeWaffle(-amp_waffle)
        time.sleep(1)
        img_pull = wfs.getImage()
        dm.setFlatSurf()
    else:
        dm.pokeWaffleLarge(amp_waffle)
        time.sleep(1)
        img_push = wfs.getImage()
        dm.pokeWaffleLarge(-amp_waffle)
        time.sleep(1)
        img_pull = wfs.getImage()
        dm.setFlatSurf()
    #  ------ RECONSTRUCT WAFFLE -------
    wfs.reconNonLinear(img_push,nRec_waffle)
    push = wfs.phase_rec
    wfs.reconNonLinear(img_pull,nRec_waffle)
    pull = wfs.phase_rec
    phi_waffle = (push-pull)/(2*amp_waffle)
        
    #%% =========================== TRANSFORMATION DM -> WFS ================================
    # central pixel in wfs map
    y_dm_0 = x_dm_poke[0]
    x_dm_0 = y_dm_poke[0]
    x_wfs_0 = pos_act_x[0]
    y_wfs_0 = pos_act_y[0]

    # COMPUTE TRANSFORMATION PARAMETERS
    v_dm = np.array([x_dm_poke[0]-x_dm_poke[1],y_dm_poke[0]-y_dm_poke[1]])
    v_wfs = np.array([pos_act_y[0]-pos_act_y[1],pos_act_x[0]-pos_act_x[1]])
    print(pos_act_x)
    print(pos_act_y)
    # scaling along X ---------
    alpha = v_wfs[1]/v_dm[1]
    print('ALPHA',alpha)
    # scaling along Y ---------
    beta = -v_wfs[0]/v_dm[0]
    print('BETA',beta)
    # rotation ---------
    ps = np.dot(v_dm,v_wfs)/(np.linalg.norm(v_dm)*np.linalg.norm(v_wfs)) # normalized scalar product
    theta = 0*np.arccos(ps)
    print('THETA',theta)
    xx_wfs,yy_wfs = transformation_dm2wfs(alpha,beta,theta,x_wfs_0,y_wfs_0,x_dm_0,y_dm_0,dm.xx_dm,dm.yy_dm)


    #%% ======================== FINAL SELECTION ON PLOT ================================
    time.sleep(1)
    # ---- Waffle ----
    fig, ax = plt.subplots()
    fig.set_size_inches(14, 8)
    plt.subplots_adjust(left=0.1,right=0.8,bottom=0.1)
    ax.set_aspect("equal")
    axcolor = 'skyblue'
    plt.subplot(121)
    plt.imshow(phi_waffle)
    grid_wfs = plt.scatter(xx_wfs,yy_wfs,edgecolor ="orange",facecolors='none')
    #ax.scatter(xx_wfs_valid,yy_wfs_valid,edgecolor ="yellow",facecolors='none')
    # ---- POKE ---
    plt.subplot(122)
    plt.imshow(np.sum(poke,axis=2))
    grid_wfs_2 = plt.scatter(xx_wfs,yy_wfs,edgecolor ="orange",facecolors='none')
    #plt.scatter(xx_wfs_valid,yy_wfs_valid,edgecolor ="yellow",facecolors='none')
    poke_pos = plt.scatter(xx_wfs[int(number_valid_actuators[x_dm_poke[0],y_dm_poke[0]]-1)],yy_wfs[int(number_valid_actuators[x_dm_poke[0],y_dm_poke[0]]-1)],edgecolor ="red",facecolors='none')
    poke_2_pos = plt.scatter(xx_wfs[int(number_valid_actuators[x_dm_poke[1],y_dm_poke[1]]-1)],yy_wfs[int(number_valid_actuators[x_dm_poke[1],y_dm_poke[1]]-1)],edgecolor ="red",facecolors='none')
    poke_3_pos = plt.scatter(xx_wfs[int(number_valid_actuators[x_dm_poke[2],y_dm_poke[2]]-1)],yy_wfs[int(number_valid_actuators[x_dm_poke[2],y_dm_poke[2]]-1)],edgecolor ="red",facecolors='none')
    poke_4_pos = plt.scatter(xx_wfs[int(number_valid_actuators[x_dm_poke[3],y_dm_poke[3]]-1)],yy_wfs[int(number_valid_actuators[x_dm_poke[3],y_dm_poke[3]]-1)],edgecolor ="red",facecolors='none')

    # ------- SLIDERS --------
    slider_alpha_ax = plt.axes([0.2, 0.9, 0.15, 0.03], facecolor=axcolor)
    slider_alpha = Slider(slider_alpha_ax, 'Alpha', alpha-2, alpha+2, alpha)# big pupil 0.2908

    slider_beta_ax = plt.axes([0.2, 0.8, 0.15, 0.03], facecolor=axcolor)
    slider_beta = Slider(slider_beta_ax, 'Beta', beta-2, beta+2, beta)# big pupil 0.2908

    slider_theta_ax = plt.axes([0.4, 0.85, 0.15, 0.03], facecolor=axcolor)
    slider_theta = Slider(slider_theta_ax, 'Theta', -np.pi, np.pi, theta)

    slider_xwfs0_ax = plt.axes([0.6, 0.9, 0.15, 0.03], facecolor=axcolor)
    slider_xwfs0 = Slider(slider_xwfs0_ax, 'X0_WFS', 0.8*x_wfs_0, 1.2*x_wfs_0, x_wfs_0)# big pupil 0.2908
    
    slider_ywfs0_ax = plt.axes([0.6, 0.8, 0.15, 0.03], facecolor=axcolor)
    slider_ywfs0 = Slider(slider_ywfs0_ax, 'Y0_WFS',0.8*y_wfs_0, 1.2*y_wfs_0, y_wfs_0)# big pupil 0.2908
    
    def update(val):
        # ----- UPDATE X and Y position -----
        alpha = slider_alpha.val
        beta = slider_beta.val
        theta = slider_theta.val
        x_wfs_0 = slider_xwfs0.val
        y_wfs_0 = slider_ywfs0.val
        xx_wfs,yy_wfs = transformation_dm2wfs(alpha,beta,theta,x_wfs_0,y_wfs_0,x_dm_0,y_dm_0,dm.xx_dm,dm.yy_dm)
        grid_wfs.set_offsets(np.transpose(np.array([xx_wfs,yy_wfs])))
        grid_wfs_2.set_offsets(np.transpose(np.array([xx_wfs,yy_wfs])))
        poke_pos.set_offsets(np.array([xx_wfs[int(number_valid_actuators[x_dm_poke[0],y_dm_poke[0]]-1)],yy_wfs[int(number_valid_actuators[x_dm_poke[0],y_dm_poke[0]]-1)]]))
        poke_2_pos.set_offsets(np.array([xx_wfs[int(number_valid_actuators[x_dm_poke[1],y_dm_poke[1]]-1)],yy_wfs[int(number_valid_actuators[x_dm_poke[1],y_dm_poke[1]]-1)]]))
        poke_3_pos.set_offsets(np.array([xx_wfs[int(number_valid_actuators[x_dm_poke[2],y_dm_poke[2]]-1)],yy_wfs[int(number_valid_actuators[x_dm_poke[2],y_dm_poke[2]]-1)]]))
        poke_4_pos.set_offsets(np.array([xx_wfs[int(number_valid_actuators[x_dm_poke[3],y_dm_poke[3]]-1)],yy_wfs[int(number_valid_actuators[x_dm_poke[3],y_dm_poke[3]]-1)]]))
        fig.canvas.draw_idle()
    
    slider_alpha.on_changed(update)
    slider_beta.on_changed(update)
    slider_theta.on_changed(update)
    slider_xwfs0.on_changed(update)
    slider_ywfs0.on_changed(update)
    
    plt.show(block = True)
    
    #%% ======================== SELECTING WFS VALID ACTUATORS ================================
    xx_wfs_valid = []
    yy_wfs_valid = []
    validWFS_map = np.zeros((dm.nAct,dm.nAct))
    for i in range(0,dm.nAct):
        for j in range(0,dm.nAct):
            if dm.valid_actuators_map[i,j] == 1:
                if ((xx_wfs[int(dm.valid_actuators_number[i,j]-1)]-nPx/2)**2+(yy_wfs[int(dm.valid_actuators_number[i,j]-1)]-nPx/2)**2)< (threshold_act*nPx/2)**2:
                    # Add in valid list
                    xx_wfs_valid.append(xx_wfs[int(dm.valid_actuators_number[i,j]-1)])
                    yy_wfs_valid.append(yy_wfs[int(dm.valid_actuators_number[i,j]-1)])
                    # add in Valid MAP
                    validWFS_map[i,j] = 1
    xx_wfs_valid = np.array(xx_wfs_valid)	
    yy_wfs_valid = np.array(yy_wfs_valid)
    
    #%% ======================== CREATE POKE MATRIX ================================
    poke_matrix = build_poke_matrix_wfs(xx_wfs_valid,yy_wfs_valid,A_poke[0],sigma_poke[0],wfs.nPx,wfs.pupil_footprint)
    return poke_matrix,validWFS_map

#%% ====================================================================
#   ============  unwrapping PHASE (code from internet)  ===============
#   ====================================================================


def phase_unwrap_ref(psi, weight, kmax=100):    
    """
    A weighed phase unwrap algorithm implemented in pure Python
    author: Tobias A. de Jong
    Based on:
    Ghiglia, Dennis C., and Louis A. Romero. 
    "Robust two-dimensional weighted and unweighted phase unwrapping that uses 
    fast transforms and iterative methods." JOSA A 11.1 (1994): 107-117.
    URL: https://doi.org/10.1364/JOSAA.11.000107
    and an existing MATLAB implementation:
    https://nl.mathworks.com/matlabcentral/fileexchange/60345-2d-weighted-phase-unwrapping
    Should maybe use a scipy conjugate descent.
    """
    # vector b in the paper (eq 15) is dx and dy
    dx = _wrapToPi(np.diff(psi, axis=1))
    dy = _wrapToPi(np.diff(psi, axis=0))
    
    # multiply the vector b by weight square (W^T * W)
    WW = weight**2
    
    # See 3. Implementation issues: eq. 34 from Ghiglia et al.
    # Improves number of needed iterations. Different from matlab implementation
    WWx = np.minimum(WW[:,:-1], WW[:,1:])
    WWy = np.minimum(WW[:-1,:], WW[1:,:])
    WWdx = WWx * dx
    WWdy = WWy * dy

    # applying A^T to WWdx and WWdy is like obtaining rho in the unweighted case
    WWdx2 = np.diff(WWdx, axis=1, prepend=0, append=0)
    WWdy2 = np.diff(WWdy, axis=0, prepend=0, append=0)

    rk = WWdx2 + WWdy2
    normR0 = np.linalg.norm(rk);

    # start the iteration
    eps = 1e-9
    k = 0
    phi = np.zeros_like(psi);
    while (~np.all(rk == 0.0)):
        zk = solvePoisson(rk);
        k += 1
        
        # equivalent to (rk*zk).sum()
        rkzksum = np.tensordot(rk, zk)
        if (k == 1):
            pk = zk
        else:
            betak = rkzksum  / rkzkprevsum
            pk = zk + betak * pk;

        # save the current value as the previous values
        rkzkprevsum = rkzksum

        # perform one scalar and two vectors update
        Qpk = applyQ(pk, WWx, WWy)
        alphak = rkzksum / np.tensordot(pk, Qpk)
        phi +=  alphak * pk;
        rk -=  alphak * Qpk;

        # check the stopping conditions
        if ((k >= kmax) or (np.linalg.norm(rk) < eps * normR0)):
            break;
        #print(np.linalg.norm(rk), normR0)
    print(k, rk.shape)
    return phi

def solvePoisson(rho):
    """Solve the poisson equation "P phi = rho" using DCT
    """
    dctRho = dctn(rho);
    N, M = rho.shape;
    I, J = np.ogrid[0:N,0:M]
    with np.errstate(divide='ignore'):
        dctPhi = dctRho / 2 / (np.cos(np.pi*I/M) + np.cos(np.pi*J/N) - 2)
    dctPhi[0, 0] = 0 # handling the inf/nan value
    # now invert to get the result
    phi = idctn(dctPhi);
    return phi

def solvePoisson_precomped(rho, scale):
    """Solve the poisson equation "P phi = rho" using DCT
    Uses precomputed scaling factors `scale`
    """
    dctPhi = dctn(rho) / scale
    # now invert to get the result
    phi = idctn(dctPhi, overwrite_x=True)
    return phi

def precomp_Poissonscaling(rho):
    N, M = rho.shape;
    I, J = np.ogrid[0:N,0:M]
    scale = 2 * (np.cos(np.pi*I/M) + np.cos(np.pi*J/N) - 2)
    # Handle the inf/nan value without a divide by zero warning:
    # By Ghiglia et al.:
    # "In practice we set dctPhi[0,0] = dctn(rho)[0, 0] to leave
    #  the bias unchanged"
    scale[0, 0] = 1. 
    return scale

def applyQ(p, WWx, WWy):
    """Apply the weighted transformation (A^T)(W^T)(W)(A) to 2D matrix p"""
    # apply (A)
    dx = np.diff(p, axis=1)
    dy = np.diff(p, axis=0)

    # apply (W^T)(W)
    WWdx = WWx * dx;
    WWdy = WWy * dy;
    
    # apply (A^T)
    WWdx2 = np.diff(WWdx, axis=1, prepend=0, append=0)
    WWdy2 = np.diff(WWdy, axis=0, prepend=0, append=0)
    Qp = WWdx2 + WWdy2
    return Qp


def _wrapToPi(x):
    r = (x+np.pi)  % (2*np.pi) - np.pi
    return r

def phase_unwrap(psi, weight=None, kmax=100):
    """
    Unwrap the phase of an image psi given weights weight
    This function uses an algorithm described by Ghiglia and Romero
    and can either be used with or without weight array.
    It is especially suited to recover a unwrapped phase image
    from a (noisy) complex type image, where psi would be 
    the angle of the complex values and weight the absolute values
    of the complex image.
    """

    # vector b in the paper (eq 15) is dx and dy
    dx = _wrapToPi(np.diff(psi, axis=1))
    dy = _wrapToPi(np.diff(psi, axis=0))
    
    # multiply the vector b by weight square (W^T * W)
    if weight is None:
        # Unweighed case. will terminate in 1 round
        WW = np.ones_like(psi)
    else:
        WW = weight**2
    
    # See 3. Implementation issues: eq. 34 from Ghiglia et al.
    # Improves number of needed iterations. Different from matlab implementation
    WWx = np.minimum(WW[:,:-1], WW[:,1:])
    WWy = np.minimum(WW[:-1,:], WW[1:,:])
    WWdx = WWx * dx
    WWdy = WWy * dy

    # applying A^T to WWdx and WWdy is like obtaining rho in the unweighted case
    WWdx2 = np.diff(WWdx, axis=1, prepend=0, append=0)
    WWdy2 = np.diff(WWdy, axis=0, prepend=0, append=0)

    rk = WWdx2 + WWdy2
    normR0 = np.linalg.norm(rk);

    # start the iteration
    eps = 1e-9
    k = 0
    phi = np.zeros_like(psi)
    scaling = precomp_Poissonscaling(rk)
    while (~np.all(rk == 0.0)):
        zk = solvePoisson_precomped(rk, scaling);
        k += 1
        
        # equivalent to (rk*zk).sum()
        rkzksum = np.tensordot(rk, zk)
        if (k == 1):
            pk = zk
        else:
            betak = rkzksum  / rkzkprevsum
            pk = zk + betak * pk;

        # save the current value as the previous values
        
        rkzkprevsum = rkzksum

        # perform one scalar and two vectors update
        Qpk = applyQ(pk, WWx, WWy)
        alphak = rkzksum / np.tensordot(pk, Qpk)
        phi +=  alphak * pk;
        rk -=  alphak * Qpk;

        # check the stopping conditions
        if ((k >= kmax) or (np.linalg.norm(rk) < eps * normR0)):
            break;
    return phi
