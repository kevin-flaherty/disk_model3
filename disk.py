# Define the Disk class. This class of objects can be used to create a disk structure. Given parameters defining the disk, it calculates the desnity structure under hydrostatic equilibrium and defines the grid used for radiative transfer. This object can then be fed into the modelling code which does the radiative transfer given this structure.

#two methods for creating an instance of this class

# from disk import *
# x=Disk()

# import disk
# x = disk.Disk()

#to reload the package, after making changes
#from importlib import reload
#reload(disk)

#under python 3.7, creating a disk object takes 63.7 seconds
#under python 2.7, creating a disk object takes 64.5 seconds
#disk_pow also takes one less second. disk_ecc runs ten seconds faster under python 3.7. debris_disk_ecc runs 6 seconds faster under python 3.7.

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy import ndimage
from astropy import constants as const
from scipy.special import ellipk,ellipe
from scipy.integrate import trapz

class Disk:

    'Common class for circumstellar disk structure'

    #Define useful constants
    AU      = const.au.cgs.value      # - astronomical unit (cm)
    Rsun    = const.R_sun.cgs.value   # - radius of the sun (cm)
    c       = const.c.cgs.value       # - speed of light (cm/s)
    h       = const.h.cgs.value       # - Planck's constant (erg/s)
    kB      = const.k_B.cgs.value     # - Boltzmann's constant (erg/K)
    sigmaB  = const.sigma_sb.cgs.value # - Stefan-Boltzmann constant (erg m^-1 s^-1 K^-4)
    pc      = const.pc.cgs.value      # - parsec (cm)
    Jy      = 1.e23                   # - cgs flux density (Janskys)
    Lsun    = const.L_sun.cgs.value   # - luminosity of the sun (ergs)
    Mearth  = const.M_earth.cgs.value # - mass of the earth (g)
    mh      = const.m_p.cgs.value     # - proton mass (g)
    Da      = mh                      # - atmoic mass unit (g)
    Msun    = const.M_sun.cgs.value   # - solar mass (g)
    G       = const.G.cgs.value       # - gravitational constant (cm^3/g/s^2)
    rad     = 206264.806              # - radian to arcsecond conversion
    kms     = 1e5                     # - convert km/s to cm/s
    GHz     = 1e9                     # - convert from GHz to Hz
    mCO     = 12.011+15.999           # - CO molecular weight
#    mDCO    = mCO+2.014               # - HCO molecular weight
    mu      = 2.37                    # - gas mean molecular weight
    m0      = mu*mh                   # - gas mean molecular opacity
    Hnuctog = 0.706*mu                # - H nuclei abundance fraction (H nuclei:gas)
    sc      = 1.59e21                 # - Av --> H column density (C. Qi 08,11)
    H2tog   = 0.8                     # - H2 abundance fraction (H2:gas)
    Tco     = 19.                     # - freeze out
    sigphot = 0.79*sc                 # - photo-dissociation column

    #def __init__(self,params=[-0.5,0.09,1.,10.,1000.,150.,51.5,2.3,1e-4,0.01,33.9,19.,69.3,
    #             [.79,1000],[10.,1000],-1],obs=[180,131,300,170],rtg=True,vcs=True,
    #             exp_temp=False,line='co',ring=None,include_selfgrav=False):
    def __init__(self,q=-0.5,McoG=0.09,pp=1.,Rin=1.,Rout=1000,Rc=150.,incl=51.5,
                 Mstar=2.3,Xco=1e-4,vturb=0.01,Zq0=33.9,Tmid0=19,Tatm0=69.3,sigbound=[.79,1000],
                 Rabund=[10,1000],handed=-1,vtalpha=0,vtsig=0,
                 nr=180,nphi=131,nz=300,zmax=170,rtg=True,vcs=True,
                 exp_temp=False,line='co',ring=None,include_selfgrav=False):

        ### Changed the init function so that it is easier to specify a subset of the parameters
        ### while keeping the rest as defaults.
        params = [q,McoG,pp,Rin,Rout,Rc,incl,Mstar,Xco,vturb,Zq0,Tmid0,Tatm0,sigbound,Rabund,handed,vtalpha,vtsig]
        obs = [nr,nphi,nz,zmax]
        self.ring=ring
        self.set_obs(obs)   # set the observational parameters
        self.set_params(params) # set the structure parameters
        self.include_selfgrav=include_selfgrav

        self.set_structure(exp_temp=exp_temp)  # use obs and params to create disk structure
        if rtg:
            self.set_rt_grid(vcs=vcs)
            self.set_line(line=line,vcs=vcs)

    def set_params(self,params):
        'Set the disk structure parameters'
        self.qq = params[0]                 # - temperature index
        self.McoG = params[1]*Disk.Msun     # - gas mass
        self.pp = params[2]                 # - surface density index
        self.Rin = params[3]*Disk.AU        # - inner edge in cm
        self.Rout = params[4]*Disk.AU       # - outer edge in cm
        self.Rc = params[5]*Disk.AU         # - critical radius in cm
        self.thet = math.radians(params[6]) # - convert inclination to radians
        self.Mstar = params[7]*Disk.Msun    # - convert mass of star to g
        self.Xco = params[8]                # - CO gas fraction
        self.vturb = params[9]*Disk.kms     # - turbulence velocity
        self.zq0 = params[10]               # - Zq, in AU, at 150 AU
        self.tmid0 = params[11]             # - Tmid at 150 AU
        self.tatm0 = params[12]             # - Tatm at 150 AU
        self.sigbound = [params[13][0]*Disk.sc,params[13][1]*Disk.sc]
              # - upper and lower column density boundaries
        if len(params[14]) ==2:
            # - inner and outer abundance boundaries
            self.Rabund = [params[14][0]*Disk.AU,params[14][1]*Disk.AU]
        else:
            # - inner/outer ring, width of inner/outer ring
            self.Rabund=[params[14][0]*Disk.AU,params[14][1]*Disk.AU,params[14][2]*Disk.AU,params[14][3]*Disk.AU,params[14][4]*Disk.AU,params[14][5]*Disk.AU]
        self.handed = params[15]            #

        #vturb power law radially and Gaussian with height (from Ashraf Dhabhi's thesis)
        self.vtalpha = params[16] # power law for radial dependency
        self.vtsig = params[17] #scale factory by which we multiply width for the height dependency Gaussian
        self.vvtr = (self.vtalpha != 0) #Boolean describing whether there is a radial dependency
        self.vvtz = (self.vtsig != 0) #Boolean describing whether there is a height dependency
        self.vvt = self.vvtr | self.vvtz # Boolean describing whether there is any spatial dependency

        self.costhet = np.cos(self.thet)  # - cos(i)
        self.sinthet = np.sin(self.thet)  # - sin(i)
        if self.ring is not None:
            self.Rring = self.ring[0]*Disk.AU # location of the ring
            self.Wring = self.ring[1]*Disk.AU #width of ring
            self.sig_enhance = self.ring[2] #power law slope of inner temperature structure

    def set_obs(self,obs):
        'Set the observational parameters. These parameters are the number of r, phi, S grid points in the radiative transer grid, along with the maximum height of the grid.'
        self.nr = obs[0]
        self.nphi = obs[1]
        self.nz = obs[2]
        self.zmax = obs[3]*Disk.AU


    def set_structure(self,exp_temp=False):
        '''Calculate the disk density and temperature structure given the specified parameters'''
        # Define the desired regular cylindrical (r,z) grid
        nrc = 500#256             # - number of unique r points
        rmin = self.Rin       # - minimum r [AU]
        rmax = self.Rout      # - maximum r [AU]
        nzc = int(2.5*nrc)#*5           # - number of unique z points
        zmin = .1*Disk.AU      # - minimum z [AU] *****0.1?
        rf = np.logspace(np.log10(rmin),np.log10(rmax),nrc)
        zf = np.logspace(np.log10(zmin),np.log10(self.zmax),nzc)

        #idz = np.zeros(nzc)+1
        #rcf = np.outer(rf,idz)
        idr = np.ones(nrc)
        zcf = np.outer(idr,zf)
        rcf = rf[:,np.newaxis]*np.ones(nzc)

        # rcf[0][:] = radius at all z for first radial bin
        # zcf[0][:] = z in first radial bin

        # Build a turbulence matrix of the correct dimensions that is constant in all of the disk (AD)
        vt = np.outer(np.ones(nrc),np.ones(nzc))*self.vturb/Disk.kms


        # Interpolate dust temperature and density onto cylindrical grid
        tf = 0.5*np.pi-np.arctan(zcf/rcf)  # theta values
        rrf = np.sqrt(rcf**2.+zcf**2)

        # bundle the grid for helper functions
        grid = {'nrc':nrc,'nzc':nzc,'rf':rf,'rmax':rmax,'zcf':zcf}
        self.grid=grid

        #define temperature structure
        # use Dartois (03) type II temperature structure
        delta = 1.                # shape parameter
        zq = self.zq0*Disk.AU*(rcf/(150*Disk.AU))**1.3 #1.3
        tmid = self.tmid0*(rcf/(150*Disk.AU))**self.qq#[0] #******************#
        tatm = self.tatm0*(rcf/(150*Disk.AU))**self.qq#[1] #******************#
        tempg = tatm + (tmid-tatm)*np.cos((np.pi/(2*zq))*zcf)**(2.*delta)
        ii = zcf > zq
        tempg[ii] = tatm[ii]
        if exp_temp:
            #Type I temperature structure
            tempg = tmid*np.exp(np.log(tatm/tmid)*zcf/zq)
            ii = tempg > 500 #cap on temperatures
            tempg[ii] = 500.


        # Calculate vertical density structure
        Sc = self.McoG*np.abs(2.-self.pp)/(2*np.pi*self.Rc*self.Rc)
        siggas = Sc*(rf/self.Rc)**(-1*self.pp)*np.exp(-1*(rf/self.Rc)**(2-self.pp))
        if self.ring is not None:
            w = np.abs(rcf-self.Rring)<self.Wring/2.
            if w.sum() > 0:
                tempg[w] = tempg[w]*(rcf[w]/(150*Disk.AU))**(self.sig_enhance-self.qq)/((rcf[w].max())/(150.*Disk.AU))**(-self.qq+self.sig_enhance)
                #print('Temperature enhancement at ring center: ',(self.Rring/(150*Disk.AU))**(self.sig_enhance-self.qq)/((rcf[w].max())/(150.*Disk.AU))**(-self.qq+self.sig_enhance))

        self.calc_hydrostatic(tempg,siggas,grid)

        #Calculate radial pressure differential
        Pgas = Disk.kB/Disk.m0*self.rho0*tempg
        dPdr = (np.roll(Pgas,-1,axis=0)-Pgas)/(np.roll(rcf,-1,axis=0)-rcf)

        #Calculate self-gravity
        if self.include_selfgrav:
            dphidr = np.zeros((len(rf),len(zf)))
            siggasf = siggas[:,np.newaxis]*np.ones(nzc)
            for i in np.arange(len(rf)):
                r = rf[i]
                k = np.sqrt(4*r*rcf/((r+rcf)**2.+zcf**2.))
                Kk = ellipk(k)
                Ek = ellipe(k)
                integrand = (Kk-.25*(k**2./(1-k**2.))*(rcf/r-r/rcf+zcf**2./(r*rcf))*Ek)*np.sqrt(rcf/r)*k*siggasf
                dphidr[i,:] = (self.G/r)*trapz(integrand,rcf,axis=0)

        #Calculate velocity field
        if self.include_selfgrav:
            Omg = np.sqrt((dPdr/(rcf*self.rho0)+Disk.G*self.Mstar/(rcf**2+zcf**2)**1.5+dphidr/rcf))
        else:
            Omg = np.sqrt((dPdr/(rcf*self.rho0)+Disk.G*self.Mstar/(rcf**2+zcf**2)**1.5))
        Omk = np.sqrt(Disk.G*self.Mstar/rcf**3.)

        # Check for NANs
        ii = np.isnan(Omg)
        Omg[ii] = Omk[ii]
        ii = np.isnan(self.rho0)
        if ii.sum() > 0:
            self.rho0[ii] = 1e-60
            print('Beware: removed NaNs from density (#%s)' % ii.sum())
        ii = np.isnan(tempg)
        if ii.sum() > 0:
            tempg[ii] = 2.73
            print('Beware: removed NaNs from temperature (#%s)' % ii.sum())
        ii = np.isinf(Omg)
        if ii.sum() > 0:
            Omg[ii] = Omk[ii]

        # find photodissociation boundary layer from top
        sig_col = np.zeros((nrc,nzc)) #Cumulative mass surface density along vertical lines starting at z=170AU
        for ir in range(nrc):
            psl = (Disk.Hnuctog/Disk.m0*self.rho0[ir,:])[::-1]
            zsl = self.zmax - (zcf[ir,:])[::-1]
            foo = (zsl-np.roll(zsl,1))*(psl+np.roll(psl,1))/2.
            foo[0] = 0
            nsl = foo.cumsum()
            #cumulative mass column density along vertical columns
            sig_col[ir,:] = nsl[::-1]*Disk.m0/Disk.Hnuctog
        self.sig_col = sig_col #save it for later

        # Set default freeze-out temp
        self.Tco = Disk.Tco

        # Defining a sound speed matrix using the temperature matrix (AD)
        cs = np.sqrt((Disk.kB*tempg)/Disk.m0)
        # Defining a vector that approximates the pressure scale height at different radii using the midplane sound speed
        est = cs[:,0]/Omg[:,0]
        self.estimH = []
        for e in est:
            self.estimH += [nzc*[e]]
        self.estimH = np.array(self.estimH)

        if self.vvtr:
            #In case of radial dependency
            vtr = np.power((rf/(150*Disk.AU)),self.vtalpha)
            vt = vt*vtr[:,np.newaxis]
            self.vtrv = vtr
            print('Radial dependence')
        if self.vvtz:
            # In case of height dependency
            vtz = 1-np.exp(-np.power(zcf,2)/(2*np.power(self.vtsig*self.estimH,2)))
            vt = vt*vtz
            self.vtzv = vtz
            print('height dependence')

        self.rf = rf
        self.nrc = nrc
        self.zf = zf
        self.nzc = nzc
        self.tempg = tempg
        self.Omg0 = Omg
        self.vtm=vt




    def set_rt_grid(self,vcs=True):
        ### Start of Radiative Transfer portion of the code...
        # Define and initialize cylindrical grid
       
        #Smin = 1*Disk.AU                 # offset from zero to log scale
        #if self.thet > np.arctan(self.Rout/self.zmax):
        #    Smax = 2*self.Rout/self.sinthet
        #    print('*Edge*-on disk')
        #else:
        #    Smax = 2.*self.zmax/self.costhet       # los distance through disk
        #print('Smax: ',Smax/self.AU)
        #Smid = Smax/2.                    # halfway along los
        #ytop = Smax*self.sinthet/2.       # y origin offset for observer xy center
        #print('ytop: ',ytop/self.AU)

        R = np.linspace(0,self.Rout,self.nr)
        phi = np.arange(self.nphi)*2*np.pi/(self.nphi-1)
        #foo = np.floor(self.nz/2)

        #S_old = np.arange(2*foo)/(2*foo)*(Smax-Smin)+Smin


        # Basically copy S_old, with length nz,  into each column of a nphi*nr*nz matrix
        #S2 = (S_old[:,np.newaxis,np.newaxis]*np.ones((self.nr,self.nphi))).T
        #print('S2 min, max: ',S2.min()/self.AU,S2.max()/self.AU)

        # arrays in [phi,r,s]
        X = (np.outer(R,np.cos(phi))).transpose()
        Y = (np.outer(R,np.sin(phi))).transpose()

        #Use a rotation matrix to transform between radiative transfer grid and physical structure grid
        if np.abs(self.thet) > np.arctan(self.Rout/self.zmax):
            zsky_max = np.abs(2*self.Rout/self.sinthet)
            #print('i>90,edge zsky, max')
        else:
            zsky_max = 2*(self.zmax/self.costhet)
            #print('i<90,top zsky,max')
        zsky = np.arange(self.nz)/self.nz*(-zsky_max)+zsky_max/2.
        #zsky = np.arange(self.nz)/(self.nz)*(-2*self.Rout)+self.Rout#Smid-S
        
        tdiskZ = (Y.repeat(self.nz).reshape(self.nphi,self.nr,self.nz))*self.sinthet+zsky*self.costhet
        tdiskY = (Y.repeat(self.nz).reshape(self.nphi,self.nr,self.nz))*self.costhet-zsky*self.sinthet
        #theta_crit = np.arctan((self.Rout+tdiskY)/(self.Rout-tdiskZ))
        #S = (self.Rout-tdiskZ)/self.costhet
        #theta_crit = np.arctan((self.Rout+tdiskY)/(self.zmax-tdiskZ))
        if (self.thet<np.pi/2) & (self.thet>0):
            theta_crit = np.arctan((self.Rout+tdiskY)/(self.zmax-tdiskZ))
            S = (self.zmax-tdiskZ)/self.costhet
        #print('S (rotation) min, max: ',S.min()/self.AU,S.max()/self.AU)
            S[(theta_crit<self.thet)] = ((self.Rout+tdiskY[(theta_crit<self.thet)])/self.sinthet)
            #print(theta_crit)
            #print((theta_crit<self.thet).shape)
        elif self.thet>np.pi/2:
            theta_crit = np.arctan((self.Rout+tdiskY)/(self.zmax+tdiskZ))
            #print(theta_crit)
            #print((theta_crit<self.thet).shape)
            S = -(self.zmax+tdiskZ)/self.costhet
            #print(S.shape)
            S[(theta_crit<(np.pi-self.thet))] = ((self.Rout+tdiskY[(theta_crit<(np.pi-self.thet))])/self.sinthet)
        elif (self.thet<0) & (self.thet>-np.pi/2):
            theta_crit = np.arctan((self.Rout-tdiskY)/(self.zmax-tdiskZ))
            S = (self.zmax-tdiskZ)/self.costhet
            S[(theta_crit<np.abs(self.thet))] = -((self.Rout-tdiskY[(theta_crit<np.abs(self.thet))])/self.sinthet)
        #print('S (rotation+correction) min, max: ',S.min()/self.AU,S.max()/self.AU)
        #print('Rout,zmax: ',self.Rout/self.AU,self.zmax/self.AU)

        # transform grid
        #tdiskZ = self.zmax*(np.ones((self.nphi,self.nr,self.nz)))-self.costhet*S2
        #if self.thet > np.arctan(self.Rout/self.zmax):
        #    tdiskZ -=(Y*self.sinthet).repeat(self.nz).reshape(self.nphi,self.nr,self.nz)
        #tdiskY = ytop - self.sinthet*S2 + (Y/self.costhet).repeat(self.nz).reshape(self.nphi,self.nr,self.nz)
        #tr2 = np.sqrt(X.repeat(self.nz).reshape(self.nphi,self.nr,self.nz)**2+tdiskY2**2)
        tr = np.sqrt(X.repeat(self.nz).reshape(self.nphi,self.nr,self.nz)**2+tdiskY**2)
        notdisk = (tr > self.Rout) | (tr < self.Rin) | (np.abs(tdiskZ)>self.zmax) # - individual grid elements not in disk
        isdisk = (tr>self.Rin) & (tr<self.Rout) & (np.abs(tdiskZ)<self.zmax)
        #S=S2
        S -= S[isdisk].min() #Reset the min S to 0
        #print('S (isdisk) min, max: ',S[isdisk].min()/self.AU,S[isdisk].max()/self.AU)
        #xydisk =  tr[:,:,0] <= self.Rout+Smax*self.sinthet  # - tracing outline of disk on observer xy plane

        #print('Zdisk max, min: ',tdiskZ[isdisk].max()/self.AU,tdiskZ[isdisk].min()/self.AU)
        #print('Rdisk max, min: ',tr[isdisk].max()/self.AU,tr[isdisk].min()/self.AU)
        #isdisk2 = (tr2>self.Rin) & (tr2<self.Rout) & (np.abs(tdiskZ2)<self.zmax)
        #print('Zdisk2 max, min: ',tdiskZ2[isdisk2].max()/self.AU,tdiskZ2[isdisk2].min()/self.AU)
        #print('Rdisk2 max, min: ',tr2[isdisk2].max()/self.AU,tr2[isdisk2].min()/self.AU)
        #plt.plot(tdiskZ.flatten()[::100]/self.AU,np.degrees(theta_crit).flatten()[::100],'.k')
        #plt.plot(tr.flatten()[::100]/self.AU,np.degrees(theta_crit).flatten()[::100],'.k')
        #plt.figure()
        #plt.tricontourf(tr[isdisk].flatten()[::20]/self.AU,tdiskZ[isdisk].flatten()[::20]/self.AU,S[isdisk].flatten()[::20]/self.AU,100)
        #plt.title('New grid')
        #plt.xlabel('R (au)')
        #plt.ylabel('Z (au)')
        #plt.colorbar()
        #plt.figure()
        #plt.tricontourf(tr2[isdisk2].flatten()[::20]/self.AU,tdiskZ2[isdisk2].flatten()[::20]/self.AU,S2[isdisk2].flatten()[::20]/self.AU,100)
        #plt.title('Old grid')
        #plt.xlabel('R (au)')
        #plt.ylabel('Z (au)')
        #plt.colorbar()

        #plt.figure()
        #plt.subplot(131)
        #plt.plot(tdiskZ[isdisk].flatten()[::293]/self.AU,S[isdisk].flatten()[::293]/self.AU,'.k')
        #plt.plot(tdiskZ2[isdisk2].flatten()[::293]/self.AU,S2[isdisk2].flatten()[::293]/self.AU,'.r',alpha=.5)
        #plt.legend(('New grid','Old grid'))
        #plt.xlabel('Z (au)')
        #plt.ylabel('S (au)')
        #plt.subplot(132)
        #plt.plot(tdiskY[isdisk].flatten()[::293]/self.AU,S[isdisk].flatten()[::293]/self.AU,'.k')
        #plt.plot(tdiskY2[isdisk2].flatten()[::293]/self.AU,S2[isdisk2].flatten()[::293]/self.AU,'.r',alpha=.5)
        #plt.xlabel('Y (au)')
        #plt.ylabel('S (au)')
        #plt.subplot(133)
        #plt.plot(X.repeat(self.nz).reshape(self.nphi,self.nr,self.nz)[isdisk].flatten()[::293]/self.AU,S[isdisk].flatten()[::293]/self.AU,'.k')
        #plt.plot(X.repeat(self.nz).reshape(self.nphi,self.nr,self.nz)[isdisk2].flatten()[::293]/self.AU,S2[isdisk2].flatten()[::293]/self.AU,'.r',alpha=.5)
        #plt.xlabel('X (au)')
        #plt.ylabel('S (au)')

        # interpolate to calculate disk temperature and densities
        #print('interpolating onto radiative transfer grid')
        #need to interpolate tempg from the 2-d rcf,zcf onto 3-d tr
        xind = np.interp(tr.flatten(),self.rf,range(self.nrc)) #rf,nrc
        yind = np.interp(np.abs(tdiskZ).flatten(),self.zf,range(self.nzc)) #zf,nzc
        tT = ndimage.map_coordinates(self.tempg,[[xind],[yind]],order=1).reshape(self.nphi,self.nr,self.nz) #interpolate onto coordinates xind,yind #tempg
        Omg = ndimage.map_coordinates(self.Omg0,[[xind],[yind]],order=1).reshape(self.nphi,self.nr,self.nz) #Omg
        tsig_col = ndimage.map_coordinates(self.sig_col,[[xind],[yind]],order=1).reshape(self.nphi,self.nr,self.nz)###
        vt = ndimage.map_coordinates(self.vtm,[[xind],[yind]],order=1).reshape(self.nphi,self.nr,self.nz) #AD

        tT[notdisk]=2.73
        self.r = tr
        #print('Negative column densities, before interpolation: ',(self.sig_col<0).sum())
        self.sig_col = tsig_col
        #print('Negative column densities, after interpolation: ',(self.sig_col<0).sum())
        self.sig_col[self.sig_col<0] = 0

        #print('Negative temperatures before interpolation: ',(self.tempg<0).sum())
        #print('Negative temperatures, after interpolation: ',(tT<0).sum())

        #Set molecular abundance
        self.add_mol_ring(self.Rabund[0]/Disk.AU,self.Rabund[1]/Disk.AU,self.sigbound[0]/Disk.sc,self.sigbound[1]/Disk.sc,self.Xco,initialize=True)

        if np.size(self.Xco)>1:
            #gaussian rings
            self.Xmol = self.Xco[0]*np.exp(-(self.Rabund[0]-tr)**2/(2*self.Rabund[3]**2))+self.Xco[1]*np.exp(-(self.Rabund[1]-tr)**2/(2*self.Rabund[4]**2))+self.Xco[2]*np.exp(-(self.Rabund[2]-tr)**2/(2*self.Rabund[5]**2))

        #Freeze-out
        zap = (tT<self.Tco)
        if zap.sum() > 0:
            self.Xmol[zap] = 1/5.*self.Xmol[zap]

        trhoH2 = Disk.H2tog/Disk.m0*ndimage.map_coordinates(self.rho0,[[xind],[yind]],order=1).reshape(self.nphi,self.nr,self.nz)
        trhoH2[notdisk] = 0
        trhoG = trhoH2*self.Xmol
        #trhoG[notdisk] = 0
        self.rhoH2 = trhoH2

        #print('Negative densities before interpolation: ',(self.rho0<0).sum())
        #print('Negative densities, after interpolation: ',(trhoH2<0).sum())

        self.add_dust_ring(self.Rin,self.Rout,0.,0.,initialize=True) #initialize dust density to 0

        ##from scipy.special import erfc
        ##trhoG = .5*erfc((tdiskZ-zpht_up)/(.1*zpht_up))*trhoG

        #temperature and turbulence broadening
        #tdBV = np.sqrt(2.*Disk.kB/(Disk.Da*Disk.mCO)*tT+self.vturb**2)
        if vcs:
            tdBV = np.sqrt((1+vt**2.)*(2.*Disk.kB/(Disk.Da*Disk.mCO)*tT)) #AD
            #tdBV = np.sqrt((1+(self.vturb/Disk.kms)**2.)*(2.*Disk.kB/(Disk.Da*Disk.mCO)*tT)) #vturb proportional to cs
            #self.r = tr
            #self.Z = tdiskZ
            #vt = self.doppler()*np.sqrt(Disk.Da*Disk.mCO/(Disk.Da*Disk.mu))
            #tdBV = np.sqrt((1+vt**2.)*(2.*Disk.kB/(Disk.Da*Disk.mCO)*tT))
        else:
            #tdBV = np.sqrt(2.*Disk.kB/(Disk.Da*Disk.mCO)*tT+self.vturb**2)
            tdBV = np.sqrt(2.*Disk.kB/(Disk.Da*Disk.mCO)*tT+(vt*Disk.kms)**2) #AD


        # store disk
        self.X = X
        self.Y = Y
        self.Z = tdiskZ
        self.S = S
        #self.r = tr
        self.T = tT
        self.dBV = tdBV
        self.rhoG = trhoG
        self.Omg = Omg
        self.i_notdisk = notdisk
        #self.i_xydisk = xydisk
        self.cs = np.sqrt(2*self.kB/(self.Da*2)*self.T)
        self.vt = vt #AD
        #self.sig_col=tsig_col
        #self.rhoH2 = trhoH2

    def set_line(self,line='co',vcs=True):
        self.line = line
        try:
            if line.lower()[:2]=='co':
                self.m_mol = 12.011+15.999
            elif line.lower()[:4]=='c18o':
                self.m_mol = 12.011+17.999
            elif line.lower()[:4]=='13co':
                self.m_mol = 13.003+15.999
            elif line.lower()[:3] == 'hco':
                self.m_mol = 1.01 + 12.01 + 16.0
            elif line.lower()[:3] == 'hcn':
                self.m_mol = 1.01 + 12.01 + 14.01
            elif line.lower()[:2] == 'cs':
                self.m_mol = 12.01 + 32.06
            elif line.lower()[:3] == 'dco':
                self.m_mol = 12.011+15.999+2.014
            else:
                raise ValueError('Choose a known molecule [CO, C18O, 13CO, HCO, HCO+, HCN, CS, DCO+] for the line parameter')
            #assume it is DCO+
        except ValueError as error:
            raise
        if vcs:
            #temperature and turbulence broadening
            #tdBV = np.sqrt(2.*Disk.kB/(Disk.Da*Disk.mHCO)*tT+self.vturb**2)
            #tdBV = np.sqrt((1+(self.vturb/Disk.kms)**2.)*(2.*Disk.kB/(Disk.Da*self.m_mol)*self.T)) #vturb proportional to cs
            tdBV = np.sqrt((1+self.vt**2.)*(2.*Disk.kB/(Disk.Da*self.m_mol)*self.T)) #vturb proportional to cs, AD

        else: #assume line.lower()=='co'
            #temperature and turbulence broadening
            #tdBV = np.sqrt(2.*Disk.kB/(Disk.Da*self.m_mol)*self.T+self.vturb**2)
            tdBV = np.sqrt(2.*Disk.kB/(Disk.Da*self.m_mol)*self.T+(self.vt*Disk.kms)**2) #AD


        self.dBV=tdBV

    def add_dust_ring(self,Rin,Rout,dtg,ppD,initialize=False):
        '''Add a ring of dust with a specified inner radius, outer radius, dust-to-gas ratio (defined at the midpoint) and slope of the dust-to-gas-ratio'''

        if initialize:
            self.dtg = 0*self.r
            self.kap = 2.3

        w = (self.r>(Rin*Disk.AU)) & (self.r<(Rout*Disk.AU))
        Rmid = (Rin+Rout)/2.*Disk.AU
        self.dtg[w] += dtg*(self.r[w]/Rmid)**(-ppD)
        self.rhoD = self.rhoH2*self.dtg*2*Disk.mh

    def add_mol_ring(self,Rin,Rout,Sig0,Sig1,abund,alpha=0,initialize=False,just_frozen=False):
        '''Add a ring of fixed abundance, between Rin and Rout (in the radial direction) and Sig0 and Sig1 (in the vertical direction). The abundance is treated as a power law in the radial direction, with alpha as the power law exponent, and normalized at the inner edge of the ring (abund~abund0*(r/Rin)^(alpha))
        disk.add_mol_ring(10,100,.79,1000,1e-4)
        just_frozen: only apply the abundance adjustment to the areas of the disk where CO is nominally frozen out
        '''
        if initialize:
            self.Xmol = np.zeros(np.shape(self.r))+1e-18
        if just_frozen:
            add_mol = (self.sig_col*Disk.Hnuctog/Disk.m0>Sig0*Disk.sc) & (self.sig_col*Disk.Hnuctog/Disk.m0<Sig1*Disk.sc) & (self.r>Rin*Disk.AU) & (self.r<Rout*Disk.AU) & (self.T < self.Tco)
        else:
            add_mol = (self.sig_col*Disk.Hnuctog/Disk.m0>Sig0*Disk.sc) & (self.sig_col*Disk.Hnuctog/Disk.m0<Sig1*Disk.sc) & (self.r>Rin*Disk.AU) & (self.r<Rout*Disk.AU)
        if add_mol.sum()>0:
            self.Xmol[add_mol]+=abund*(self.r[add_mol]/(Rin*Disk.AU))**(alpha)
        
        ## New method, using sigmoids to soften the boundaries.
        ### This seems like a better idea, since it softens the abundance on either side of the boundary, but it doesn't actually solve the problem I want it to solve. Commenting out for now (KMF, July 2025)
        #xvar_sig=(self.sig_col*Disk.Hnuctog/(Disk.m0*Disk.sc)-np.max([Sig0,1e-30])/10)/(np.max([Sig0,1e-30])/10)
        #fabund_sig = 1*(1+1e8*np.exp(-xvar_sig))**(-1.)
        #xvar_Rout = (self.r-Rout*Disk.AU)/(Rout*Disk.AU)
        #fabund_Rout = (1-np.tanh(xvar_Rout*100.))/2.#1*(1+1e8*np.exp(-xvar_Rout))**(-1.)
        #xvar_Rin = (self.r-Rin*Disk.AU)/(Rin*Disk.AU)
        #fabund_Rin = (1+np.tanh(xvar_Rin*.01))/2.#1*(1+1e8*np.exp(-xvar_Rin))**(-1.)

        #self.Xmol *= fabund_sig*fabund_Rout*fabund_Rin

        #plt.figure()
        #plt.subplot(211)
        #plt.loglog(xvar_sig.flatten()[::100],self.Xmol.flatten()[::100],'.k')
        #plt.axhline(abund)
        #plt.xlabel('Column density')
        #plt.ylabel('Abundance modification')
        #plt.subplot(212)
        #plt.loglog(self.r.flatten()[::100]/Disk.AU,self.Xmol.flatten()[::100],'.k')
        #plt.axhline(abund)
        #plt.xlabel('Radius (au)')
        #plt.ylabel('Abundance modification')
        #plt.subplot(313)
        #plt.plot(self.r.flatten()[::100]/Disk.AU,fabund_Rin.flatten()[::100],'.k')
        #plt.xlabel('Radius (au)')
        #plt.ylabel('Abundance modification')        
        
        #add soft boundaries
        edge1 = (self.sig_col*Disk.Hnuctog/Disk.m0>Sig0*Disk.sc) & (self.sig_col*Disk.Hnuctog/Disk.m0<Sig1*Disk.sc) & (self.r>Rout*Disk.AU) #Outer edge of the disk
        if edge1.sum()>0:
            self.Xmol[edge1] += abund*(self.r[edge1]/(Rin*Disk.AU))**(alpha)*np.exp(-(self.r[edge1]/(Rout*Disk.AU))**16) #16
        edge2 = (self.sig_col*Disk.Hnuctog/Disk.m0>Sig0*Disk.sc) & (self.sig_col*Disk.Hnuctog/Disk.m0<Sig1*Disk.sc) & (self.r<Rin*Disk.AU) #Inner edge of the disk
        if edge2.sum()>0:
            self.Xmol[edge2] += abund*(self.r[edge2]/(Rin*Disk.AU))**(alpha)*(1-np.exp(-(self.r[edge2]/(Rin*Disk.AU))**20.))
        edge3 = (self.sig_col*Disk.Hnuctog/Disk.m0<Sig0*Disk.sc) & (self.r>Rin*Disk.AU) & (self.r<Rout*Disk.AU) # Top of the disk
        if edge3.sum()>0:
            self.Xmol[edge3] += abund*(self.r[edge3]/(Rin*Disk.AU))**(alpha)*(1-np.exp(-((self.sig_col[edge3]*Disk.Hnuctog/Disk.m0)/(Sig0*Disk.sc))**8.))
        
        zap = (self.Xmol<0)
        if zap.sum()>0:
            self.Xmol[zap]=1e-18
        if not initialize:
            self.rhoG = self.rhoH2*self.Xmol


    def calc_hydrostatic(self,tempg,siggas,grid):
        nrc = grid['nrc']
        nzc = grid['nzc']
        zcf = grid['zcf']
        rf = grid['rf']

        #compute rho structure
        rho0 = np.zeros((nrc,nzc))
        sigint = siggas

        #print('Doing hydrostatic equilibrium')
        for ir in range(nrc):
            #compute gravo-thermal constant
            grvc = Disk.G*self.Mstar*Disk.m0/Disk.kB

            #extract the T(z) profile at a given radius
            T = tempg[ir]

            #differential equation for vertical density profile
            z = zcf[ir]
            dz = (z - np.roll(z,1))
            dlnT = (np.log(T)-np.roll(np.log(T),1))/dz
            dlnp = -1*grvc*z/(T*(rf[ir]**2.+z**2.)**1.5)-dlnT
            dlnp[0] = -1*grvc*z[0]/(T[0]*(rf[ir]**2.+z[0]**2.)**1.5)

            #numerical integration to get vertical density profile
            foo = dz*(dlnp+np.roll(dlnp,1))/2.
            foo[0] = 0.
            lnp = foo.cumsum()

            #normalize the density profile (note: this is just half the sigma value!)
            dens = 0.5*sigint[ir]*np.exp(lnp)/np.trapz(np.exp(lnp),z)

            #gaussian profile
            #hr = np.sqrt(2*T[0]*rf[ir]**3./grvc)
            #dens = sigint[ir]/(np.sqrt(np.pi)*hr)*np.exp(-(z/hr)**2.)

            rho0[ir,:] = dens


        self.rho0=rho0


    def density(self):
        'Return the density structure'
        return self.rho0

    def temperature(self):
        'Return the temperature structure'
        return self.tempg

    def grid(self):
        'Return an XYZ grid (but which one??)'
        return self.grid

    def get_params(self):
        params=[]
        params.append(self.qq)
        params.append(self.McoG/Disk.Msun)
        params.append(self.pp)
        params.append(self.Rin/Disk.AU)
        params.append(self.Rout/Disk.AU)
        params.append(self.Rc/Disk.AU)
        params.append(math.degrees(self.thet))
        params.append(self.Mstar/Disk.Msun)
        params.append(self.Xco)
        params.append(self.vturb/Disk.kms)
        params.append(self.zq0)
        params.append(self.tmid0)
        params.append(self.tatm0)
        params.append(self.handed)
        return params

    def get_obs(self):
        obs = []
        obs.append(self.nr)
        obs.append(self.nphi)
        obs.append(self.nz)
        obs.append(self.zmax/Disk.AU)
        return obs

    def plot_structure(self,sound_speed=False,beta=None,dust=False,rmax=500,zmax=150):
        ''' Plot temperature and density structure of the disk'''
        plt.figure()
        plt.rc('axes',lw=2)
        cs2 = plt.contour(self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,np.log10((self.rhoG/self.Xmol)[0,:,:]),np.arange(0,11,0.1))
        #cs2 = plt.contour(self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,np.log10((self.rhoG)[0,:,:]),np.arange(-4,7,0.1))
        #cs3 = plt.contour(self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,np.log10(self.sig_col[0,:,:]),(-2,-1),linestyles=':',linewidths=3,colors='k')
        cs3 = plt.contour(self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,np.log10(self.sig_col[0,:,:]*Disk.Hnuctog/Disk.m0),(np.log10(0.*Disk.sc),np.log10(1e-6*Disk.sc),np.log10(1e-5*Disk.sc),np.log10(1e-4*Disk.sc),np.log10(1e-3*Disk.sc),np.log10(1e-2*Disk.sc),np.log10(.1*Disk.sc)),linestyles=':',linewidths=3,colors='k')
        ax = plt.gca()
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
            tick.label1.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
            tick.label1.set_fontweight('bold')
        if sound_speed:
            cs = self.r*self.Omg#np.sqrt(2*self.kB/(self.Da*self.mCO)*self.T)
            cs3 = plt.contour(self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,cs[0,:,:]/Disk.kms,100,colors='k')#*3.438
            manual_locations=[(350,50),(250,60),(220,65),(190,65),(130,55),(70,40),(25,25)]
            plt.clabel(cs3,fmt='%0.2f',manual=manual_locations)
        elif beta is not None:
            cs = np.sqrt(2*self.kB/(self.Da*self.mu)*self.T)
            rho = (self.rhoG+4)*self.mu*self.Da #mass density
            Bmag = np.sqrt(8*np.pi*rho*cs**2/beta) #magnetic field
            cs3 = plt.contour(self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,np.log10(Bmag[0,:,:]),20)
            plt.clabel(cs3)
        elif dust:
            cs3 = plt.contour(self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,np.log10(self.rhoD[0,:,:]),100,colors='k',linestyles='--')
        else:
            manual_locations=[(150,-100),(200,-77),(250,-100),(350,-120),(350,-80),(380,-40),(380,40),(350,80),(350,120),(250,100),(200,77),(150,100)]
            manual_locations=[(300,30),(250,60),(180,50),(180,70),(110,60),(45,30)]
            cs3 = plt.contour(self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,self.T[0,:,:],(20,40,60,80,100,120),colors='k',linestyles='--')
            plt.clabel(cs3,fmt='%1i',manual=manual_locations)
        plt.colorbar(cs2,label='log n')
        #plt.colorbar(cs2,label='log $\Sigma_{FUV}$')
        plt.xlim(0,rmax)
        plt.xlabel('R (AU)',fontsize=20)
        plt.ylabel('Z (AU)',fontsize=20)
        plt.ylim(0,zmax)
        plt.show()

    def calcH(self,verbose=True,return_pow=False):
        ''' Calculate the equivalent of the pressure scale height within our disks. This is useful for comparison with other models that take this as a free parameter. H is defined as 2^(-.5) times the height where the density drops by 1/e. (The factor of 2^(-.5) is included to be consistent with a vertical density distribution that falls off as exp(-z^2/2H^2))'''

        nrc = self.nrc
        zf = self.zf
        rf = self.rf
        rho0 = self.rho0

        H = np.zeros(nrc)
        for i in range(nrc):
            rho_cen = rho0[i,0]
            diff = abs(rho_cen/np.e-rho0[i,:])
            H[i] = zf[(diff == diff.min())]/np.sqrt(2)

        H100 = np.interp(100*Disk.AU,rf,H)
        psi = (np.polyfit(np.log10(rf),np.log10(H),1))[0]

        if verbose:
            print('H100 (AU): {:.3f}'.format(H100/Disk.AU))
            print('power law: {:.3f}'.format(psi))

        if return_pow:
            return (H100/Disk.AU,psi)
        else:
            return H
