#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. code-block:: python

    run /home/lab/libSEAL/debugStartup.py
    
Script to debug closed loop on SEAL static aberrations.
To be used if **quickStartup** script does not work. Usually it is because:
    * New pupil stop
    * Pupil moved on WFS
    * DMs moved w.r.t pupil 

.. warning::
    Code to be excuted with each section at a time
"""
from pyMilk.interfacing.shm import SHM
from pwfs import *
from hardware import *
from tools_pwfs import *

if __name__ == "__main__":

    #%% ############### TIP-TILT MIRROR ##############
    tt_mirror = mirror_TT()

    #%% ############### DEFORMABLE MIRRORS ##############
    dm06 = SHM('dm00disp06')
    dm = DM(dm06)

    #%% ################ PYRAMID CAMERA ##############
    ocam = SHM('ocam2dhr')
    cam = Camera(ocam)
    cam.getDark()

    1/0

    #%% ################ PYRAMID WFS MODEL ##############
    img = take_image_pupils(cam,tt_mirror,1.5)
    pupil_im,position_pup = findPWFS_param(img)
    x = np.arange(pupil_im.shape[0]) - (pupil_im.shape[0] / 2 - 0.5)
    r = ((x[None,:]-2.0)**2 + (x[:, None] - 0)**2)**.5
    cobs = r > 13
    pupil_im_obs = pupil_im*cobs
    fig(pupil_im_obs)
    # ----- CREATE PWFS object ----s
    shannon = 2
    wavelength = 900 # in nm
    PWFS = pyramidWFS(pupil_im_obs,position_pup,shannon,wavelength,cam)

    img = PWFS.getImage()
    PWFS.reconNonLinear(img)
    fig(PWFS.phase_rec)

    #%% ################ DM / WFS registration ##############
    nIterRec = 5
    pokeMatrix,validActuators = build_pokeMatrix(PWFS,dm,nIterRec)

    #%% ################ POKE MATRIX ##############
    thres = 1/20 # cutoff for linear reconstructor
    PWFS.load_pokeMatrix(pokeMatrix,0)

    #%% ############### Close Loop ################
    loopgain = 0.5
    leak = 1
    nIter = 10
    timeStop = 0.1

    C = np.zeros((dm.nAct,dm.nAct)).astype('float32')
    for iter_k in range(0,nIter):
        print('Loop iteration:',iter_k)
        time.sleep(timeStop)
        # ---- Record image -----
        img = PWFS.getImage()
        # ---- Reconstruction -----
        c = PWFS.img2cmd(img)
        c_map = dm.mapCmd(c)
        C = leak*C-loopgain*c_map
        dm.setSurf(dm.flat_surf+C)

