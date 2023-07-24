#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes for wavefront sensors:
    * 4-sided Pyramid WFS
    * Zernike WFS
    * Shhack-Hartmann WFS

.. warning::
You can add any new object, but you will need to have this 2 methods in your class:
     * wfs.getImage() : record WFS signal (can be already processed)
     * wfs.img2cmd(img) : reconstruct command (as vector) to send to DM from measurement

.. note::
     More sensors should be added: FAST, focal plane WFS (EFC)

"""

from tools import *
from numpy import matlib
           
#%% ====================================================================
#   ============== PYRAMID WAVEFRONT SENSOR OBJECT =====================
#   ====================================================================

class pyramidWFS:
    """
    This is a class to use the 4-sided PWFS

    Attributes
    ----------
    pupil_tel: array
        pupil intensities
    pupil_footprint: array
        pupil footprint (0 and 1)
    pupil_pad: array
        shannon-padded pupil
    pupil_pad_footprint: array
        shannon-padded pupil footprint
    cam: cameras Object
        camera used for this sensor
    nPx: int
        pupil resolution
    pad: int
        integer to pad array to shannon
    mask_phase: array
        pyramid mask in phase
    mask: array
        pyramid mask (exp(1j*mask_phase))
    nIterRec: int
        number of iterations for GS algorithm
    nonLinRec: bool
        choose if using non-linear reconstructor while reconstructing DM commands - 1 by default
    doUnwrap: bool
        choose if reconstructed phase is unwrapped - 0 by default
    """

    def __init__(self,pupil_tel,position_pup,shannon,wavelength,cam = 0):  
        """ CONSTRUCTOR """
        self.model = 'PWFS'
        # ---------- Pupil and resolution -----------------
        pupil_tel[pupil_tel<0] = 0
        self.pupil = pupil_tel/np.sum(pupil_tel)
        self.wavelength = wavelength # useful only for DIPSLAYING phase in NANOMETERS
        self.pupil_footprint = np.copy(self.pupil)
        self.pupil_footprint[self.pupil_footprint>0] = 1
        self.nPx = self.pupil.shape[0] # Resolution in our pupil
        self.shannon = shannon # Shannon sampling : 1 = 2px per lambda/D - Recommended parameter: shannon = 2
        self.pad = int((2*self.shannon-1)*self.nPx/2) # padding in pupil plane to reach Shannon
        self.pupil_pad = np.pad(self.pupil,((self.pad,self.pad),(self.pad,self.pad)), 'constant') # padded pupil
        self.pupil_pad_footprint = np.copy(self.pupil_pad)
        self.pupil_pad_footprint[self.pupil_pad_footprint>0] = 1
        self.modu = 0 # no modulation by default
        # ----- Non linear Reconstructor by default ----
        self.nonLinRec = 1
        self.nIterRec = 5
        self.doUnwrap = 0
        self.startPhase = np.zeros((self.nPx,self.nPx))
        # --- Camera for the WFS --
        if cam == 0: # case for simulation only
            self.cam = []
            self.nPx_img = 2*self.shannon*self.nPx
        else:
            self.cam = cam
            self.nPx_img = int(self.cam.nPx)
        # --- data for display ----
        self.img_simulated = [] # Simulated image from phase estimation - to compare with true data
        self.phase_rec = []
        self.phase_rec_unwrap = []
        self.opd_rec = []
        self.img = [] # image
        self.stopping_criteria = 0.01
        # ------- Define MASK SHAPE from position_pup -------
        self.position_pup = position_pup
        #self.offset_angle =np.ones((8,1))
        self.offset_angle = np.array([0.96736842, 1., 0.95684211,0.96947368, 0.99473684,1., 1.01157895,0.94])
        mask = self.build_mask(self.position_pup)
        self.mask_phase = mask
        self.mask = np.exp(1j*mask)
        # Reference intensities
        self.img0 = self.cropImg(self.getImageSimu(np.zeros((self.nPx,self.nPx))))

    ################################ BUILD PYRAMIDAL MASK ####################### 

    def changeMask(self,mask_new):
        self.mask_phase = mask_new
        self.mask = np.exp(1j*self.mask_phase)
        # Reference intensities
        self.img0 = self.cropImg(self.getImageSimu(np.zeros((self.nPx,self.nPx))))

    def build_mask(self,position_pup):
        """ Take pupil position and build mask """
        # position_pup in percent of Diameter along X and Y axis
        # Angle for all faces
        l = np.linspace(-int(self.nPx*self.shannon),int(self.nPx*self.shannon)-1,int(2*self.nPx*self.shannon))
        [tip_x,tilt_y] = np.meshgrid(l,l)
        # left top pupil -----
        x_lt = position_pup[0]*self.offset_angle[0]
        y_lt = position_pup[1]*self.offset_angle[1]
        angle_lt = tip_x*np.pi*x_lt/self.shannon+tilt_y*np.pi*y_lt/self.shannon
        # left bottom pupil -----
        x_lb = position_pup[2]*self.offset_angle[2]
        y_lb = position_pup[3]*self.offset_angle[3]
        angle_lb = tip_x*np.pi*x_lb/self.shannon+tilt_y*np.pi*y_lb/self.shannon
        # right top pupil -----
        x_rt = position_pup[4]*self.offset_angle[4]
        y_rt = position_pup[5]*self.offset_angle[5]
        angle_rt = tip_x*np.pi*x_rt/self.shannon+tilt_y*np.pi*y_rt/self.shannon
        # right bottom pupil -----
        x_rb = position_pup[6]*self.offset_angle[6]
        y_rb = position_pup[7]*self.offset_angle[7]
        angle_rb = tip_x*np.pi*x_rb/self.shannon+tilt_y*np.pi*y_rb/self.shannon
        # Create mask
        mask = np.zeros((int(2*self.nPx*self.shannon),int(2*self.nPx*self.shannon)))
        mask[0:int(self.nPx*self.shannon),0:int(self.nPx*self.shannon)] = angle_rt[0:int(self.nPx*self.shannon),0:int(self.nPx*self.shannon)]
        mask[0:int(self.nPx*self.shannon),int(self.nPx*self.shannon):] = angle_lt[0:int(self.nPx*self.shannon),int(self.nPx*self.shannon):]
        mask[int(self.nPx*self.shannon):,0:int(self.nPx*self.shannon)] = angle_rb[int(self.nPx*self.shannon):,0:int(self.nPx*self.shannon)]
        mask[int(self.nPx*self.shannon):,int(self.nPx*self.shannon):] = angle_lb[int(self.nPx*self.shannon):,int(self.nPx*self.shannon):]
        return mask

    ################################ CALIBRATION PROCESS #######################  
    
    def load_pokeMatrix(self,pokeMatrix,calib=0):
        """
        Load a poke matrix computed through calibration process, and compute its pseudo-inverse.
        It is useful to project reconstructed phase onto DM actuators.(This poke matrix is therefore associated with a diven dm.)

            Parameters:
                pokeMatrix (np.array): poke phase in WFS space
                calib (bool): launching or not the synthetic calibration
        
        """
        self.mode2phase = pokeMatrix
        self.phase2mode = np.linalg.pinv(self.mode2phase,1/30)
        # --- End-to-End calibration matrix ------
        if calib == 1:
            self.calibrateSimu(1/30)

    def calibrate(self,dm,modal=0,amp_calib=None):
        """
        Push-pull calibration for the given DM. It can be zonal or modal (only Zernike modes for now).

            Parameters:
                dm (DM_seal object): DM for used for calibration
                modal (bool): Set if the calibration is done on a modal basis - Zernike modes (=1) - or not (=0)
        """
        if amp_calib is None:
            amp_calib = 0.05

        if modal == 0:
            self.modal = 0
            # ---- ZONAL ---------
            self.validActuators = dm.valid_actuators_map
            self.intMat = np.zeros((self.nPx_img**2,int(np.sum(dm.valid_actuators_map))))
            compt = 0
            for i in range(dm.nAct):
                for j in range(dm.nAct):
                    if dm.valid_actuators_map[i,j] == 1:
                        # --------- PUSH --------
                        dm.pokeAct(amp_calib,[i,j])
                        time.sleep(0.1)
                        s_push = self.getImage()
                        # --------- PULL --------
                        dm.pokeAct(-amp_calib,[i,j])
                        time.sleep(0.1)
                        s_pull = self.getImage()
                        # -------- Push-pull ------
                        s = (s_push-s_pull)/(2*amp_calib)
                        self.intMat[:,compt] = s.ravel()
                        compt = compt + 1
                        print('Percentage done: ',compt/int(np.sum(dm.valid_actuators_map)))
        elif modal == 1:
            self.modal = 1
            nModes = dm.Z2C.shape[1]
            self.Z2C = dm.Z2C
            self.intMat = np.zeros((self.nPx_img*self.nPx_img,nModes))
            for k in range(nModes):
                    # --------- PUSH --------
                    dm.pokeZernike(amp_calib,k+2)
                    time.sleep(0.1)
                    s_push = self.getImage()
                    # --------- PULL --------
                    dm.pokeZernike(-amp_calib,k+2)
                    time.sleep(0.1)
                    s_pull = self.getImage()
                    # -------- Push-pull ------
                    s = (s_push-s_pull)/(2*amp_calib)
                    self.intMat[:,k] = s.ravel()
                    print('Percentage done: ',100*k/nModes)
        dm.setFlatSurf()
        self.compute_cmdMat(self.intMat)

    def compute_cmdMat(self,intMat,thres=None):
        """
        Compute pseudo inverse of the interaction matrix.

            Parameters:
                threshold (float - optional): conditionning for pseudo-inverse computation.
        """
        if thres is None:
            thres = 1/30 # by default
        self.cmdMat = np.linalg.pinv(intMat,thres)
    
    def calibrateSimu(self,mode2phase=None,phaseRef=None):
        """ 
        Compute synthetic interaction matrix.

            Parameters:
                mode2phase (np.array): mode to calibrate. By default, it is the pokeMatrix that was loaded.
        """
        if not(mode2phase is None):
            self.modal = 1
            self.Z2C = 0
            self.mode2phase = mode2phase
            self.phase2mode = np.linalg.pinv(self.mode2phase,1/30)
        else:
            self.modal = 0
        if phaseRef is None:
            phaseRef = np.zeros((self.nPx,self.nPx))
        # -- Linear Calibration -----
        amp_calib = 0.00001
        self.intMat_simu = np.zeros((self.nPx_img*self.nPx_img,self.mode2phase.shape[1]))
        for k in range(self.mode2phase.shape[1]):
                poke_calib = self.mode2phase[:,k].reshape(int(np.sqrt(self.mode2phase.shape[0])),int(np.sqrt(self.mode2phase.shape[0])))
                # --------- PUSH --------
                I_push = self.cropImg(self.getImageSimu(phaseRef+amp_calib*poke_calib))
                I_push = I_push/np.sum(I_push) # normalisation
                # --------- PULL --------
                I_pull = self.cropImg(self.getImageSimu(phaseRef-amp_calib*poke_calib))
                I_pull = I_pull/np.sum(I_pull) # normalisation
                # -------- Push-pull ------
                s = (I_push-I_pull)/(2*amp_calib)
                self.intMat_simu[:,k] = s.ravel()
        self.compute_cmdMat(self.intMat_simu)

    def load_intMat(self,intMat,dm,modal=0):
        """ 
        Load true interaction matrix already computed before.
        Precise if it is a modal or a zonal matrix (zonal by default).

            Parameters:
                intMat (np.array): Interaction Matrix to be loaded
                dm (DM_seal object): DM associated with this interaction matrix
                modal (bool): if it is modal or not (=0 by default)
        """
        self.modal=modal
        if modal == 1:
            self.Z2C = dm.Z2C
        self.intMat = intMat
        self.compute_cmdMat(self.intMat)


    ################################ NOISE and SENSITIVITY #######################
     
    def noisePropag(self,sigma_ron,sigma_dark,Nph):
        """ Noise propagation model in action """
        # Uniform noise
        self.S_uniform = np.sqrt(np.diag(np.dot(np.transpose(self.intMat),self.intMat)))
        # Photon Noise
        I0 = np.transpose(matlib.repmat(self.img0.ravel(),self.intMat.shape[1],1))
        D = self.intMat/np.sqrt(I0)
        self.S_photon = np.sqrt(np.diag(np.dot(np.transpose(D),D)))        
        # ---- Compute noise propagation for all modes ----
        # RON
        sigma_phi_ron = sigma_ron/(self.S_uniform*Nph)
        # Dark
        sigma_phi_dark = sigma_dark/(self.S_uniform*Nph)
        # photon noise
        sigma_phi_photon = 1/(self.S_photon*np.sqrt(Nph))
        # Sum on all modes
        sigma_phi = np.sqrt(sigma_phi_ron**2+sigma_phi_dark**2+sigma_phi_photon**2)
        #sigma_phi = sum(sigma_phi)/np.sqrt(self.intMat.shape[1])
        return sigma_phi

    ################################ PROPAGATION ####################### 
     
    def propag(self,phi,psf_img=None):
        """ PROPAGATION of the EM field """
        phi_pad =  np.pad(phi,((self.pad,self.pad),(self.pad,self.pad)), 'constant') # padded pupil
        # To first focal plane
        amp = np.sqrt(self.pupil_pad) # amplitude = sqrt(intensities)
        Psi_FP = fft_centre(amp*np.exp(1j*phi_pad))
        # Multiply by Zernike Phase mask
        if psf_img is None:
            Psi_FP = self.mask*Psi_FP
        else:
            Psi_FP = self.mask*np.sqrt(psf_img)*np.exp(1j*np.angle(Psi_FP))
        # Back to pupil plane
        Psi_PP = ifft_centre(Psi_FP)
        return Psi_PP


    def backPropag(self,amp,phi,psf_img=None):
        """ BACKWARD PROPAGATION of the EM field for GS algorithm """ 
        # To first focal plane
        Psi_FP = fft_centre(amp*np.exp(1j*phi))
        # Multiply by conjugate of Zernike Phase mask
        if psf_img is None:
            Psi_FP = np.conj(self.mask)*Psi_FP
        else:
            Psi_FP = np.conj(self.mask)*np.sqrt(psf_img)*np.exp(1j*np.angle(Psi_FP))
        # Back to pupil plane
        Psi_PP = ifft_centre(Psi_FP)
        
        return Psi_PP

    def getImage(self):
        """ Record True image """
        img = self.cam.get()
        img[img<0]=0
        img = img/np.sum(img)
        return img
    
    def getImageSimu(self,phi):
        """ Simulation of Pyramid WFS image - include a modulated case if needed """
        # ========= Non-modulated case =========
        if self.modu == 0:
            Psi_PP = self.propag(phi)
            # Intensities
            img = np.abs(Psi_PP)**2
            img = img/np.sum(img)
        # ========= Modulated case =========
        else:
            img = np.zeros((self.shannon*self.nPx*2,self.shannon*self.nPx*2))
            for k in range(0,self.TTmodu.shape[2]):
                Psi_PP = self.propag(phi+self.TTmodu[:,:,k])
                # Intensities
                img_modu = np.abs(Psi_PP)**2
                img = img+img_modu	
            img = img/np.sum(img)
        return img

    def getImageSimuMultiLambda(self,phi,lambda_list):
        """ Simulation of BROADBAND Pyramid WFS image - include a modulated case if needed """
        img = np.zeros((self.shannon*self.nPx*2,self.shannon*self.nPx*2))
        for l in lambda_list:
            phi_l = phi*self.wavelength/l # scale phase to right value
            img = img + self.getImageSimu(phi_l)
        img = img/np.sum(img)
        return img

    def setModu(self,modu):
        if modu != 0:
            self.modu = modu
            # ======== Create modulation ring ========
            w = np.zeros((self.shannon*self.nPx*2,self.shannon*self.nPx*2))
            Rmod_px = 2*self.shannon*self.modu
            for i in range(0,self.shannon*self.nPx*2):
                for j in range(0,self.shannon*self.nPx*2):
                    if np.sqrt((i-(self.shannon*self.nPx*2-1)/2)**2+(j-(self.shannon*self.nPx*2-1)/2)**2)< Rmod_px+1 and np.sqrt((i-(self.shannon*self.nPx*2-1)/2)**2+(j-(self.shannon*self.nPx*2-1)/2)**2)>=Rmod_px:
                        w[i,j]=1
            self.w = w/sum(sum(w))
            # ===== Create Cube with all modulation tip-tilt
            TTmodu = None
            l = np.linspace(-self.shannon*self.nPx,self.shannon*self.nPx,self.shannon*self.nPx*2)
            [xx,yy] = np.meshgrid(l,l)
            for i in range(0,self.shannon*self.nPx*2):
                for j in range(0,self.shannon*self.nPx*2):
                    if self.w[i,j]!=0:
                        ls = 2*np.pi/(self.shannon*self.nPx*2)*((i-(self.shannon*self.nPx*2-1)/2)*xx+(j-(self.shannon*self.nPx*2-1)/2)*yy)
                        ls = ls[int(self.shannon*self.nPx+1-self.nPx/2-1):int(self.shannon*self.nPx+self.nPx/2),int(self.shannon*self.nPx+1-self.nPx/2-1):int(self.shannon*self.nPx+self.nPx/2)] 
                        if TTmodu is None:
                            TTmodu = ls
                        else:
                            TTmodu = np.dstack([TTmodu,ls])
            self.TTmodu = TTmodu
        # ==== Update Reference ========
        self.img0 = self.cropImg(self.getImageSimu(np.zeros((self.nPx,self.nPx))))

    def reconLinear(self,img):
        """ Linear reconstructor using synthetic interaction matrix """
        self.img = img
        dI = img.ravel()/np.sum(img)-self.img0.ravel()/np.sum(self.img0) # reduced intensities
        cmd = np.dot(self.cmdMat,dI)
        if self.modal == 1 and self.Z2C != 0:
            cmd = np.dot(self.Z2C,cmd)
        self.phase_rec = np.dot(self.mode2phase,cmd)
        self.phase_rec = np.reshape(self.phase_rec,(self.nPx,self.nPx))
        self.img_simulated = self.cropImg(self.getImageSimu(self.phase_rec))
        return cmd

    def reconNonLinear(self,img,nIterRec=None,psf_img=None):
        """ Non-linear reconstructor using GS algorithm
        img: PWFS recorded image (dark removed)
        nIter: number of iteration for the reconstructor
        """
        if nIterRec is None:
            nIterRec = self.nIterRec
        img = np.copy(img)
        img[img<0] = 0 #killing negative values
        # Pad image if true image
        if img.shape[0] < 2*self.nPx*self.shannon:
                img = np.pad(img,((int(self.nPx*self.shannon-self.nPx_img/2),int(self.nPx*self.shannon-self.nPx_img/2)),(int(self.nPx*self.shannon-self.nPx_img/2),int(self.nPx*self.shannon-self.nPx_img/2))), 'constant') # padded pupil
        #  ========= GS algortihm to reconstruct phase ==========
        # --- 0 point for phase in detector plane ----
        Psi_0 = self.propag(self.startPhase,psf_img)
        phi_0 = np.angle(Psi_0)
        # --- 0 point for amplitude in detector plane ---
        frame = np.copy(img)
        amp_0 = np.sqrt(frame) # SQRT because img is the intensity
        # --- First BACK PROPAGATION ----
        Psi_p = self.backPropag(amp_0,phi_0,psf_img)
        # First phase estimate
        phi_k = np.angle(Psi_p)
        phi_k = phi_k*self.pupil_pad_footprint
        phi_k[np.isnan(phi_k)] = 0 # avoid NaN values
        phi_k = phi_k[int(self.shannon*self.nPx+1-self.nPx/2-1):int(self.shannon*self.nPx+self.nPx/2),int(self.shannon*self.nPx+1-self.nPx/2-1):int(self.shannon*self.nPx+self.nPx/2)] 
        err_k_previous = float('inf') # for stopping criteria
        for k in range(0,nIterRec):
                print('GS algorithm iteration number: ',k)
                # ---- Direct propagation ----
                Psi_d_k = self.propag(phi_k,psf_img)
                phi_d_k = np.angle(Psi_d_k) # record phase in WFS camera plane
                # ---- BACK PROPAGATION ----
                Psi_p = self.backPropag(amp_0,phi_d_k,psf_img)
                phi_k = np.angle(Psi_p)
                phi_k = phi_k*self.pupil_pad_footprint
                phi_k[np.isnan(phi_k)] = 0 # avoid NaN values
                phi_k = phi_k[int(self.shannon*self.nPx+1-self.nPx/2-1):int(self.shannon*self.nPx+self.nPx/2),int(self.shannon*self.nPx+1-self.nPx/2-1):int(self.shannon*self.nPx+self.nPx/2)]
                # STOPPING CRITERIA -----------
                #err_k = error_rms(image_k,img_red)
        # ----- Record last phase --------
        phi = phi_k
        self.img_simulated = self.cropImg(self.getImageSimu(phi_k))
        self.img = img
        self.phase_rec = phi
        self.opd_rec = self.phase_rec*self.wavelength/2*np.pi
        self.phase_rec_unwrap = phase_unwrap(phi)*self.pupil_footprint
             

    def getPSF(self,phi = None):
        """ Get PSF from estimated phase """
        if phi is None:
                phi = self.phase_rec
        phi_pad =  np.pad(phi,((self.pad,self.pad),(self.pad,self.pad)), 'constant') # padded pupil
        # ---- PROPAGATION of the EM field ----
        # To first focal plane
        amp = np.sqrt(self.pupil_pad) # amplitude = sqrt(intensities)
        Psi_FP = fft_centre(amp*np.exp(1j*phi_pad))
        psf = np.abs(Psi_FP)**2
        return psf

    def cropImg(self,img):
        """ crop Image to have same size as true image """
        img = img[int(self.nPx*self.shannon-self.nPx_img/2):int(self.nPx*self.shannon+self.nPx_img/2),int(self.nPx*self.shannon-self.nPx_img/2):int(self.nPx*self.shannon+self.nPx_img/2)]
        return img

    ################################ DM Command from images #######################
    def img2cmd(self,img):
        """ 
        Full reconstruction of the signal: image to DM commands.
        """
        if self.nonLinRec == 0:
            self.img = img
            cmd = self.reconLinear(img)
        else:
            self.reconNonLinear(img,self.nIterRec)
            if self.doUnwrap == 0:
                cmd = np.dot(self.phase2mode,self.phase_rec.ravel())
                self.opd_rec = self.phase_rec*self.wavelength/2*np.pi
            else:
                cmd = np.dot(self.phase2mode,self.phase_rec_unwrap.ravel())
                self.opd_rec = self.phase_rec_unwrap*self.wavelength/2*np.pi			
        return cmd

