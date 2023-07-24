
from tools import *
import os

#%% ====================================================================
#   ================= TT mirror =========================
#   ====================================================================


class mirror_TT:

	def __init__(self):

	def move(self,[ampX,ampY]):
		os.system('')

	def moveRef(self):
		os.system('')





#%% ====================================================================
#   ================= CAMERA =========================
#   ====================================================================


class camera:
	"""
	This is a class to communicate with BlackFly cameras

	Attributes
	----------

		model: string
			choose which camera: PWFS, SHWFS-PSF, ZWFS, or Ben
		binning: int
			choose camera binning
		imShm: ktrc object
		dit: krtc object
		dark: array
			dark for the given exposure time. Automatically removed from images.
		nPx: int
			resolution (nPx*nPx)

	Methods
	----------
	"""
	def __init__(self,ocam):
		""" CONSTRUCTOR """ 

		self.ocam = ocam
		self.nPx = 240
		self.nFrames = 100
		self.dark = np.zeros((self.nPx,self.nPx))
		
	def getDark(self,nFrames=None):
		""" Take Dark image """ 
		if nFrames is None:
			nFrames = self.nFrames
		self.dark = self.get(nFrames)
		print('Dark Frame taken')

	def get(self,nFrames=None):
		""" Get image from camera """ 
		if nFrames is None:
			nFrames = self.nFrames
		image = 0
		for k in range(0,nFrames):
			im = self.ocam.get_data(True)
			image = image+im
		image = image/nFrames
		# remove dark
		image = image-self.dark
		return image

#%% ====================================================================
#   ================= DEFOMABLE MIRRORS OBJECT =========================
#   ====================================================================

class DM:

	
	def __init__(self,dm06):
		""" CONSTRUCTOR """ 	
		self.dm06 = dm06
		self.flat_surf = np.zeros((self.nAct,self.nAct))
		# -------- Valid Actuator grid -----
		self.nAct = 50
		self.valid_actuators_map = load('pup.npy')
		# ---- Valid acuator with referencing number map ------
		self.valid_actuators_number = np.copy(self.valid_actuators_map)
		k = 1
		for i in range(0,self.nAct):
			for j in range(0,self.nAct):
				if self.valid_actuators_number[i][j] == 1:
					self.valid_actuators_number[i][j] = k
					k = k + 1
		# ----- Valid acutators position -----------
		X = np.round(np.linspace(0,self.nAct-1,self.nAct))
		[xx_dm,yy_dm] = np.meshgrid(X,X) # actuators grid
		#valid actuators
		self.xx_dm = xx_dm[self.valid_actuators_map==1]
		self.yy_dm = yy_dm[self.valid_actuators_map==1]


	def pokeAct(self,amplitude,position,bias=None):
		""" Poke actuator: take its position in X and Y """
		if bias is None:
			bias = np.copy(self.flat_surf)
		dm_shape = np.copy(bias)
		if len(position) == 2:
			dm_shape[position[0]][position[1]] = dm_shape[position[0]][position[1]] + amplitude
		else:
			print('please enter valid position in x and y according to obj.valid_actuators_map')
		# --- Send commands ---
		self.setSurf(dm_shape)
		return dm_shape
	
	def pokeWaffle(self,amplitude=0.1,bias=None):
		""" Poking Waffle patern - Amplitude in PtV """
		if bias is None:
			bias = np.copy(self.flat_surf)
		# Maximun PtV: 0.5
		if amplitude > 0.5:
			amplitude = 0.5
			print('To high amplitude for Waffle: set to 0.5 PtV')
		# Create waffle patern
		a = -np.ones((self.nAct,self.nAct)).astype(np.float32)
		a[::2,:] = 1
		a[:,::2] = 1
		b = -np.ones((self.nAct,self.nAct)).astype(np.float32)
		b[1::2,:] = 1
		b[:,1::2] = 1
		waffle = bias+amplitude/2*a*b
		self.setSurf(waffle)
		return waffle	
		
	def pokeWaffleLarge(self,amplitude=0.1,bias=None):
		""" Poking Large Waffle patern - Amplitude in PtV """
		if bias is None:
			bias = np.copy(self.flat_surf)
		# Maximun PtV: 0.5
		if amplitude > 0.5:
			amplitude = 0.5
			print('To high amplitude for Waffle: set to 0.5 PtV')
		# Create waffle patern
		a = -np.zeros((self.nAct,self.nAct)).astype(np.float32)
		for i in range(0,self.nAct-2):
			for j in range(0,self.nAct):
				if np.mod(i,4)==0 and np.mod(j,2)==0:
					a[i,j] = 1
				if np.mod(i,4)==0 and np.mod(j,2)==1:
					a[i+2,j] = 1
		waffle = bias+amplitude*a
		self.setSurf(waffle)
		return waffle		
				
	def setFlatSurf(self):
		""" Apply flat """
		self.setSurf(self.flat_surf)
	
	def newFlat(self,dm_map):
		""" Change flat and apply it """
		self.flat_surf = dm_map
		self.setFlatSurf()
		
	def setSurf(self,cmd):
		""" Apply DM map command """
		self.dm06.set_data(cmd*self.valid_actuators_map)
		return cmd
		