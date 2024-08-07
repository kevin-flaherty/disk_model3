# Define the Disk class. This class of objects can be used to create a disk structure. Given parameters defining the disk, it calculates the desnity structure under hydrostatic equilibrium and defines the grid used for radiative transfer. This object can then be fed into the modelling code which does the radiative transfer given this structure.

# unlike disk.py, this code uses a simple power law for the surface density (rather than a power law plus exponential tail

#two methods for creating an instance of this class

# from disk_pow import *
# x=Disk()

# import disk_pow
# x = disk.Disk()

# For testing purposes use the second method. Then I can use reload(disk) after updating the code
import math
#from mol_dat import mol_dat
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
    AU = const.au.cgs.value          # - astronomical unit (cm)
    Rsun = const.R_sun.cgs.value     # - radius of the sun (cm)
    c = const.c.cgs.value            # - speed of light (cm/s)
    h = const.h.cgs.value            # - Planck's constant (erg/s)
    kB = const.k_B.cgs.value         # - Boltzmann's constant (erg/K)
    pc = const.pc.cgs.value          # - parsec (cm)
    Jy = 1.e23                       # - cgs flux density (Janskys)
    Lsun = const.L_sun.cgs.value     # - luminosity of the sun (ergs)
    Mearth = const.M_earth.cgs.value # - mass of the earth (g)
    mh = const.m_p.cgs.value         # - proton mass (g)
    Da = mh                          # - atmoic mass unit (g)
    Msun = const.M_sun.cgs.value     # - solar mass (g)
    G = const.G.cgs.value            # - gravitational constant (cm^3/g/s^2)
    rad = 206264.806   # - radian to arcsecond conversion
    kms = 1e5          # - convert km/s to cm/s
    GHz = 1e9          # - convert from GHz to Hz
    mCO = 12.011+15.999# - CO molecular weight
    mu = 2.37          # - gas mean molecular weight
    m0 = mu*mh     # - gas mean molecular opacity
    Hnuctog = 0.706*mu   # - H nuclei abundance fraction (H nuclei:gas)
    sc = 1.59e21   # - Av --> H column density (C. Qi 08,11)
    H2tog = 0.8    # - H2 abundance fraction (H2:gas)
    Tco = 19.    # - freeze out
    sigphot = 0.79*sc   # - photo-dissociation column

#    def __init__(self,params=[-0.5,0.09,1.,10.,1000.,150.,51.5,2.3,1e-4,0.01,33.9,19.,69.3,-1],obs=[180,131,300,170],rtg=True,vcs=True,exp_temp=False,line='co',ring=None,include_selfgrav=False):
    def __init__(self,q=-0.5,McoG=0.09,pp=1.,Rin=10.,Rout=1000.,Rc=150.,incl=51.5,
                 Mstar=2.3,Xco=1e-4,vturb=0.01,Zq0=33.9,Tmid0=19.,Tatm0=69.3,
                 handed=-1,nr=180,nphi=131,nz=300,zmax=170,rtg=True,vcs=True,exp_temp=True,
                 line='co',ring=None,include_selfgrav=False):
        
        params=[q,McoG,pp,Rin,Rout,Rc,incl,Mstar,Xco,vturb,Zq0,Tmid0,Tatm0,handed]
        obs=[nr,nphi,nz,zmax]
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
        self.handed = params[13]            #
        self.costhet = np.cos(self.thet)  # - cos(i)
        self.sinthet = np.sin(self.thet)  # - sin(i)
        if self.ring is not None:
            self.Rring = self.ring[0]*Disk.AU #location of the ring
            self.Wring = self.ring[1]*Disk.AU #width of the ring
            self.sig_enhance = self.ring[2] #power law slope of the inner temperature structure

    def set_obs(self,obs):
        'Set the observational parameters. These parameters are the number of r, phi, S grid points in the radiative transer grid, along with the maximum height of the grid.'
        self.nr = obs[0]
        self.nphi = obs[1]
        self.nz = obs[2]
        self.zmax = obs[3]*Disk.AU


    def set_structure(self,exp_temp=False):
        '''Calculate the disk density and temperature structure given the specified parameters'''
        # Define the desired regular cylindrical (r,z) grid
        nrc = 256             # - number of unique r points
        rmin = self.Rin       # - minimum r [AU]
        rmax = self.Rout      # - maximum r [AU]
        nzc = nrc*5           # - number of unique z points
        zmin = .05*Disk.AU      # - minimum z [AU]
        rf = np.logspace(np.log10(rmin),np.log10(rmax),nrc)
        zf = np.logspace(np.log10(zmin),np.log10(self.zmax),nzc)

        #idz = np.zeros(nzc)+1
        #rcf = np.outer(rf,idz)
        idr = np.zeros(nrc)+1
        zcf = np.outer(idr,zf)
        rcf = rf[:,np.newaxis]*np.ones(nzc)

        # rcf[0][:] = radius at all z for first radial bin
        # zcf[0][:] = z in first radial bin

        # Here introduce new z-grid (for now just leave old one in)

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
        #zq = self.zq0*Disk.AU*(rcf/(150*Disk.AU))**1.1
        tmid = self.tmid0*(rcf/(150*Disk.AU))**self.qq
        tatm = self.tatm0*(rcf/(150*Disk.AU))**self.qq
        tempg = tatm + (tmid-tatm)*np.cos((np.pi/(2*zq))*zcf)**(2.*delta)
        ii = zcf > zq
        tempg[ii] = tatm[ii]
        if exp_temp:
            #Type I temperature structure
            tempg = tmid*np.exp(np.log(tatm/tmid)*zcf/zq)
            ii = tempg > 500 #cap on temperatures
            tempg[ii] = 500.


        # Calculate vertical density structure
        Sc_old = self.McoG*(2.-self.pp)/(2*np.pi*self.Rc*self.Rc)
                         #        siggas = Sc*(rf/self.Rc)**(-1*self.pp)*np.exp(-1*(rf/self.Rc)**(2-self.pp))
        #change the previous line to be a power law in radius, with index pp, extending from Rin to Rc. From Rc to Rout the surface density is 0 (will the zeros cause problems for the hydrostatic equilibrium calculation? if it does then I could simply set Rout=Rc)
        #what is normalization on Sig_gas, in terms of Mdisk?
        if self.pp ==2:
            Sc = self.McoG/(2*np.pi*(np.log(self.Rc)-np.log(self.Rin)))
        else:
            Sc = self.McoG*(2.-self.pp)/(2*np.pi*(self.Rc**(2-self.pp)-self.Rin**(2-self.pp)))

        siggas = Sc*rf**(-1*self.pp)
        siggas[rf<self.Rin] = 1e-60
        siggas[rf>self.Rc] = 1e-60

        if self.ring is not None:
            w = np.abs(rcf-self.Rring)<self.Wring/2.
            if w.sum()>0:
                tempg[w] = tempg[w]*(rcf[w]/(150*Disk.AU))**(self.sig_enhance-self.qq)((rcf[w].max())/(150.*Disk.AU))**(-self.qq+self.enhance)


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
        #Omg = Omk

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
        zpht = np.zeros(nrc)
        sig_col = np.zeros((nrc,nzc)) #Cumulative mass surface density along vertical lines starting at z=170 au.
        for ir in range(nrc):
            psl = (Disk.Hnuctog/Disk.m0*self.rho0[ir,:])[::-1]
            zsl = self.zmax - (zcf[ir,:])[::-1]
            foo = (zsl-np.roll(zsl,1))*(psl+np.roll(psl,1))/2.
            foo[0] = 0
            nsl = foo.cumsum()
            sig_col[ir,:] = nsl[::-1]*Disk.m0/Disk.Hnuctog
            pht = (np.abs(nsl) >= Disk.sigphot)
            if pht.sum() == 0:
                zpht[ir] = np.min(self.zmax-zsl)
            else:
                zpht[ir] = np.max(self.zmax-zsl[pht])
        szpht = zpht
        self.sig_col = sig_col
        #zpht = scipy.signal.medfilt(zpht,kernel_size=7) #smooth it

        # find zpht with an error function
        #zpht2 = np.zeros(nrc)
        #for ir in range(nrc):
        #    cs = np.sqrt(Disk.kB*tempg[ir,0]/Disk.m0)
        #    omega = np.sqrt(Disk.G*self.Mstar/rf[ir]**3.)
        #    hr = np.sqrt(2.)*cs/omega
        #    ntot = 2.37*.706*siggas[ir]/2.*scipy.special.erfc(zcf[ir,:]/hr)/Disk.m0
      #      zpht2[ir] = zcf[ir][(ntot<.79*1.59e21)].min()

        #plt.plot(rf/Disk.AU,zpht/Disk.AU,color='k')
        #plt.plot(rf/Disk.AU,zpht2/Disk.AU,color='r')

        # find height where CO freezes out
        zice = np.zeros(nrc)
        for ir in range(nrc):
            foo = (tempg[ir,:] < Disk.Tco)
            if foo.sum() > 0:
                zice[ir] = np.max(zcf[ir,foo])
            else:
                zice[ir] = zmin


        self.rf = rf
        self.nrc = nrc
        self.zf = zf
        self.nzc = nzc
        self.tempg = tempg
        self.Omg0 = Omg
        self.zpht = zpht




        if 0:
            plt.figure()
            cs = plt.contour(rcf/Disk.AU,zcf/Disk.AU,np.log10(self.rho0/(Disk.mu*Disk.mh)),np.arange(0,10,1))
            #cs2 = plt.contour(tr[0,:,:]/Disk.AU,tdiskZ[0,:,:]/Disk.AU,np.log10(self.rhoG[0,:,:])+4,np.arange(0,11,1))
            cs2 = plt.contour(rcf/Disk.AU,zcf/Disk.AU,tempg,(30,50,70,100,120,150),colors='k')
            #cs3 = plt.contour(tr[0,:,:]/Disk.AU,tdiskZ[0,:,:]/Disk.AU,tT[0,:,:],(20,40,60,80,100,120),colors='k',ls='--')
            #plt.plot(tr[0,:,:]/Disk.AU,zpht[0,:,:]/Disk.AU,color='k',lw=8,ls='--')
            plt.plot(rf/Disk.AU,zice/Disk.AU,color='b',lw=6,ls='--')
            plt.plot(rf/Disk.AU,szpht/Disk.AU,color='k',lw=6,ls='--')
            plt.colorbar(cs,label='log n')
            plt.clabel(cs2,fmt='%1i')
            #plt.clabel(cs3,fmt='%3i')
            plt.xlim(0,500)
            plt.xlabel('R (AU)',fontsize=20)
            plt.ylabel('Z (AU)',fontsize=20)
            plt.show()

    def set_rt_grid(self,vcs=True):
        ### Start of Radiative Transfer portion of the code...
        # Define and initialize cylindrical grid
        #Smin = 1*Disk.AU                 # offset from zero to log scale
        #if self.thet > np.arctan(self.Rout/self.zmax):
        #    Smax = 2*self.Rout/self.sinthet
        #else:
        #    Smax = 2.*self.zmax/self.costhet       # los distance through disk
        #Smid = Smax/2.                    # halfway along los
        #ytop = Smax*self.sinthet/2.       # y origin offset for observer xy center
        R = np.linspace(0,self.Rout,self.nr)#np.logspace(np.log10(self.Rin),np.log10(self.Rout),self.nr)
        phi = np.arange(self.nphi)*2*np.pi/(self.nphi-1)
        #foo = np.floor(self.nz/2)

        #S_old = np.arange(2*foo)/(2*foo)*(Smax-Smin)+Smin#np.concatenate([Smid+Smin-10**(np.log10(Smid)+np.log10(Smin/Smid)*np.arange(foo)/(foo)),Smid-Smin+10**(np.log10(Smin)+np.log10(Smid/Smin)*np.arange(foo)/(foo))])


        # Basically copy S_old, with length nz,  into each column of a nphi*nr*nz matrix
        #S = (S_old[:,np.newaxis,np.newaxis]*np.ones((self.nr,self.nphi))).T

        # arrays in [phi,r,s]
        X = (np.outer(R,np.cos(phi))).transpose()
        Y = (np.outer(R,np.sin(phi))).transpose()

         #Use a rotation matrix to transform between radiative transfer grid and physical structure grid
        if np.abs(self.thet) > np.arctan(self.Rout/self.zmax):
            zsky_max = np.abs(2*self.Rout/self.sinthet)
        else:
            zsky_max = 2*(self.zmax/self.costhet)
        zsky = np.arange(self.nz)/self.nz*(-zsky_max)+zsky_max/2.
        
        tdiskZ = (Y.repeat(self.nz).reshape(self.nphi,self.nr,self.nz))*self.sinthet+zsky*self.costhet
        tdiskY = (Y.repeat(self.nz).reshape(self.nphi,self.nr,self.nz))*self.costhet-zsky*self.sinthet
        if (self.thet<np.pi/2) & (self.thet>0):
            theta_crit = np.arctan((self.Rout+tdiskY)/(self.zmax-tdiskZ))
            S = (self.zmax-tdiskZ)/self.costhet
            S[(theta_crit<self.thet)] = ((self.Rout+tdiskY[(theta_crit<self.thet)])/self.sinthet)
        elif self.thet>np.pi/2:
            theta_crit = np.arctan((self.Rout+tdiskY)/(self.zmax+tdiskZ))
            S = -(self.zmax+tdiskZ)/self.costhet
            S[(theta_crit<(np.pi-self.thet))] = ((self.Rout+tdiskY[(theta_crit<(np.pi-self.thet))])/self.sinthet)
        elif (self.thet<0) & (self.thet>-np.pi/2):
            theta_crit = np.arctan((self.Rout-tdiskY)/(self.zmax-tdiskZ))
            S = (self.zmax-tdiskZ)/self.costhet
            S[(theta_crit<np.abs(self.thet))] = -((self.Rout-tdiskY[(theta_crit<np.abs(self.thet))])/self.sinthet)

        # transform grid
        #tdiskZ = self.zmax*(np.ones((self.nphi,self.nr,self.nz)))-self.costhet*S
        #if self.thet > np.arctan(self.Rout/self.zmax):
        #    tdiskZ -=(Y*self.sinthet).repeat(self.nz).reshape(self.nphi,self.nr,self.nz)
        #tdiskY = ytop - self.sinthet*S + (Y/self.costhet).repeat(self.nz).reshape(self.nphi,self.nr,self.nz)
        tr = np.sqrt(X.repeat(self.nz).reshape(self.nphi,self.nr,self.nz)**2+tdiskY**2)
        notdisk = (tr > self.Rout) | (tr < self.Rin) | (np.abs(tdiskZ)>self.zmax)  # - individual grid elements not in disk
        isdisk = (tr>self.Rin) & (tr<self.Rout) & (np.abs(tdiskZ)<self.zmax)
        S -= S[isdisk].min() #Reset the min S to 0
        #xydisk =  tr[:,:,0] <= self.Rout+Smax*self.sinthet  # - tracing outline of disk on observer xy plane

        # interpolate to calculate disk temperature and densities
        #print('interpolating onto radiative transfer grid')
        #need to interpolate tempg from the 2-d rcf,zcf onto 3-d tr
        xind = np.interp(tr.flatten(),self.rf,range(self.nrc)) #rf,nrc
        yind = np.interp(np.abs(tdiskZ).flatten(),self.zf,range(self.nzc)) #zf,nzc
        tT = ndimage.map_coordinates(self.tempg,[[xind],[yind]],order=1).reshape(self.nphi,self.nr,self.nz) #interpolate onto coordinates xind,yind #tempg
        Omg = ndimage.map_coordinates(self.Omg0,[[xind],[yind]],order=1).reshape(self.nphi,self.nr,self.nz) #Omg
        trhoG = Disk.H2tog*self.Xco/Disk.m0*ndimage.map_coordinates(self.rho0,[[xind],[yind]],order=1).reshape(self.nphi,self.nr,self.nz)
        trhoH2 = trhoG/self.Xco
        tsig_col = ndimage.map_coordinates(self.sig_col,[[xind],[yind]],order=2).reshape(self.nphi,self.nr,self.nz)
        zpht = np.interp(tr.flatten(),self.rf,self.zpht).reshape(self.nphi,self.nr,self.nz) #tr,rf,zpht
        tT[notdisk] = 0
        trhoG[notdisk] = 0
        trhoH2[notdisk] = 0


        # photo-dissociation
        zap = (np.abs(tdiskZ) >= zpht)
        if zap.sum() > 0:
            trhoG[zap] = 1e-18*trhoG[zap]

        #outer disk (for power law disks)
        zap = (tr > self.Rc)
        if zap.sum() > 0:
            trhoG[zap] = 1e-18*trhoG[zap]


        # freeze out
        zap = (tT <= Disk.Tco)
        if zap.sum() >0:
            trhoG[zap] = trhoG[zap]#1e-18*trhoG[zap] #**************#

        #temperature and turbulence broadening
        #tdBV = np.sqrt(2.*Disk.kB/(Disk.Da*Disk.mCO)*tT+self.vturb**2)
        if vcs:
            tdBV = np.sqrt((1+(self.vturb/Disk.kms)**2.)*(2.*Disk.kB/(Disk.Da*Disk.mCO)*tT)) #vturb proportional to cs
            #self.r = tr
            #self.Z = tdiskZ
            #vt = self.doppler()*np.sqrt(Disk.Da*Disk.mCO/(Disk.Da*Disk.mu))
            #tdBV = np.sqrt((1+vt**2.)*(2.*Disk.kB/(Disk.Da*Disk.mCO)*tT))
        else:
            tdBV = np.sqrt(2.*Disk.kB/(Disk.Da*Disk.mCO)*tT+self.vturb**2)


        # store disk
        self.X = X
        self.Y = Y
        self.Z = tdiskZ
        self.S = S
        self.r = tr
        self.T = tT
        self.dBV = tdBV
        self.rhoG = trhoG
        self.Omg = Omg
        self.i_notdisk = notdisk
        #self.i_xydisk = xydisk
        self.cs = np.sqrt(2*self.kB/(self.Da*2)*self.T)
        self.sig_col = tsig_col
        self.Xmol=self.Xco
        self.rhoH2 = trhoH2
        #self.tempg = tempg
        #self.zpht = zpht

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
            tdBV = np.sqrt((1+(self.vturb/Disk.kms)**2.)*(2.*Disk.kB/(Disk.Da*self.m_mol)*self.T)) #vturb proportional to cs

        else: #assume line.lower()=='co'
            #temperature and turbulence broadening
            tdBV = np.sqrt(2.*Disk.kB/(Disk.Da*self.m_mol)*tT+self.vturb**2)


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
            rho0[ir,:] = dens

            cs = np.sqrt(Disk.kB*T[0]/Disk.m0)
            omega=np.sqrt(Disk.G*self.Mstar/rf[ir]**3.)
            hr = np.sqrt(2.)*cs/omega
            rho0[ir,:] = sigint[ir]/(np.sqrt(np.pi)*hr)*np.exp(-1.0*pow(z/hr,2.))

            #if ir == 0:
            #    cs = np.sqrt(Disk.kB*T[0]/Disk.m0)
            #    omega = np.sqrt(Disk.G*self.Mstar/rf[ir]**3)
            #    hr = np.sqrt(2)*cs/omega
            #    print(rf[ir]/self.AU)
            #    rho2 = sigint[ir]/(np.sqrt(np.pi)*hr)*np.exp(-1*(z/hr)**2)
            #    plt.plot(z,dens,color='k',lw=6)
            #    plt.plot(z,rho2,color='r',ls='--')

        self.rho0=rho0
        #print(Disk.G,self.Mstar,Disk.m0,Disk.kB)

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

    def plot_structure(self,sound_speed=False,beta=None,dust=False,rmax=250,zmax=100):
        ''' Plot temperature and density structure of the disk'''
        plt.figure()
        plt.rc('axes',lw=2)
        cs2 = plt.contour(self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,np.log10(self.rhoG[0,:,:])+4,np.arange(0,11,0.1))
        #cs_vrot = plt.contour(self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,self.Omg[0,:,:]*self.r[0,:,:]/1e5,np.arange(3,7,.5),linestyles='--',colors='k')
        #plt.clabel(cs_vrot)
        cs3 = plt.contour(self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,np.log10(self.sig_col[0,:,:]),(-2,-1),linestyles=':',linewidths=3,colors='k')
        ax = plt.gca()
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
            tick.label1.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
            tick.label1.set_fontweight('bold')
        if sound_speed:
            cs = np.sqrt(2*self.kB/(self.Da*self.mCO)*self.T)
            cs3 = plt.contour(self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,cs[0,:,:]/Disk.kms*3.438,10,colors='k')
            manual_locations=[(450,10),(350,50),(300,60),(250,70),(220,75),(190,75),(130,75),(45,40)]
            plt.clabel(cs3,fmt='%0.2f')#manual=manual_locations
        elif beta is not None:
            cs = np.sqrt(2*self.kB/(self.Da*self.mu)*self.T)
            rho = (self.rhoG+4)*self.mu*self.Da #mass density
            Bmag = np.sqrt(8*np.pi*rho*cs**2/beta) #magnetic field
            cs3 = plt.contour(self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,np.log10(Bmag[0,:,:]),20)
            plt.clabel(cs3)
        elif dust:
            cs3 = plt.contour(self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,np.log10(self.rhoD[0,:,:]),100,colors='k',linestyles='--')
        else:
            manual_locations=[(240,30),(150,22),(100,18),(80,12),(60,8),(30,4)]
            cs3 = plt.contour(self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,self.T[0,:,:],(20,25,30,35,40,50),colors='k',linestyles='--')
            plt.clabel(cs3,fmt='%1i',manual=manual_locations)
        plt.colorbar(cs2,label='log n')
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

    def outer_co(self):
        '''Return the outermost radius, in AU, at which there is CO'''
        w = self.rhoG>1
        if w.sum()>0:
            return np.max(self.r[w])/self.AU

        return 0.
