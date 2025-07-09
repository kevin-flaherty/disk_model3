# Define the Disk class. This class of objects can be used to create a disk structure. Given parameters defining the disk, it calculates the desnity structure under hydrostatic equilibrium and defines the grid used for radiative transfer. This object can then be fed into the modelling code which does the radiative transfer given this structure.

#two methods for creating an instance of this class

# from disk import *
# x=Disk()

# import disk
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
import giggle_my_version as giggle

from scipy.interpolate import LinearNDInterpolator as interpnd

#testing time
import time

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

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
    Da = mh                          # - atomic mass unit (g)
    Msun = const.M_sun.cgs.value     # - solar mass (g)
    G = const.G.cgs.value            # - gravitational constant (cm^3/g/s^2)
    rad = 206264.806   # - radian to arcsecond conversion
    kms = 1e5          # - convert km/s to cm/s
    GHz = 1e9          # - convert from GHz to Hz
    mCO = 12.011+15.999# - CO molecular weight
    mHCO = mCO+1.008-0.0005 # - HCO molecular weight
    mu = 2.37          # - gas mean molecular weight
    m0 = mu*mh     # - gas mean molecular opacity
    Hnuctog = 0.706*mu   # - H nuclei abundance fraction (H nuclei:gas)
    sc = 1.59e21   # - Av --> H column density (C. Qi 08,11)
    H2tog = 0.8    # - H2 abundance fraction (H2:gas)
    Tco = 19.    # - freeze out
    sigphot = 0.79*sc   # - photo-dissociation column

#    def __init__(self,params=[-0.5,0.09,1.,10.,1000.,150.,51.5,2.3,1e-4,0.01,33.9,19.,69.3,-1,0,0,[.76,1000],[10,800]],obs=[180,131,300,170],rtg=True,vcs=True,line='co',ring=None):
    def __init__(self,q=-0.5,McoG=0.09,pp=1.,Ain=10.,Aout=1000.,Rc=150.,incl=51.5,
                 Mstar=2.3,Xco=1e-4,vturb=0.01,Zq0=33.9,Tmid0=19.,Tatm0=69.3,
                 handed=-1,ecc=0.,aop=0.,sigbound=[.79,1000],Rabund=[10,800],
                 nr=180,nphi=131,nz=300,zmax=170,rtg=True,vcs=True,line='co',ring=None):
        
        params=[q,McoG,pp,Ain,Aout,Rc,incl,Mstar,Xco,vturb,Zq0,Tmid0,Tatm0,handed,ecc,aop,sigbound,Rabund]
        obs=[nr,nphi,nz,zmax]
        #tb = time.clock()
        self.ring=ring
        self.set_obs(obs)   # set the observational parameters
        self.set_params(params) # set the structure parameters

        self.set_structure()  # use obs and params to create disk structure
        if rtg:
            self.set_rt_grid()
            self.set_line(line=line,vcs=vcs)
        #tf = time.clock()
        #print("disk init took {t} seconds".format(t=(tf-tb)))

    def set_params(self,params):
        'Set the disk structure parameters'

        '''I will add these as real parameters at some point but just adding default values to test here'''
        ms = 1 #star mass
        md = 0.35 #disc mass
        p = -.5 #surface density
        ap = 60*np.pi/180 #pitch angle
        m = 2 #azimuthal wavenumber
        beta = 5 #cool
        incl = np.pi/2.1 #inclination of the disc towards the line of sight
        pos = 90 # rotation of spiral (degrees), starting north, cw


        self.qq = params[0]                 # - temperature index
        self.McoG = params[1]*Disk.Msun     # - gas mass
        self.pp = params[2]                 # - surface density index
        self.Ain = params[3]*Disk.AU        # - inner edge in cm
        self.Aout = params[4]*Disk.AU       # - outer edge in cm
        self.Rc = params[5]*Disk.AU         # - critical radius in cm
        self.thet = math.radians(params[6]) # - convert inclination to radians
        self.Mstar = params[7]*Disk.Msun    # - convert mass of star to g
        self.Xco = params[8]                # - CO gas fraction
        self.vturb = params[9]*Disk.kms     # - turbulence velocity
        self.zq0 = params[10]               # - Zq, in AU, at 150 AU
        self.tmid0 = params[11]             # - Tmid at 150 AU
        self.tatm0 = params[12]             # - Tatm at 150 AU
        self.handed = params[13]            #
        self.ecc = params[14]               # - eccentricity of disk
        self.aop = math.radians(params[15]) # - angle between los and perapsis convert to radians
        self.sigbound = [params[16][0]*Disk.sc,params[16][1]*Disk.sc] #-upper and lower column density boundaries
        if len(params[17])==2:
            # - inner and outer abundance boundaries
            self.Rabund = [params[17][0]*Disk.AU,params[17][1]*Disk.AU]
        else:
            self.Rabund=[params[17][0]*Disk.AU,params[17][1]*Disk.AU,params[17][2]*Disk.AU,params[17][3]*Disk.AU,params[17][4]*Disk.AU,params[17][5]*Disk.AU]
        self.costhet = np.cos(self.thet)  # - cos(i)
        self.sinthet = np.sin(self.thet)  # - sin(i)
        self.cosaop = np.cos(self.aop)
        self.sinaop = np.sin(self.aop)
        if self.ring is not None:
            self.Rring = self.ring[0]*Disk.AU # location of ring
            self.Wring = self.ring[1]*Disk.AU # width of ring
            self.sig_enhance = self.ring[2] # surface density enhancement (a multiplicative factor) above the background

    def set_obs(self,obs):
        'Set the observational parameters. These parameters are the number of r, phi, S grid points in the radiative transer grid, along with the maximum height of the grid.'
        self.nr = obs[0]
        self.nphi = obs[1]
        self.nz = obs[2]
        self.zmax = obs[3]*Disk.AU


    def set_structure(self):
        #tst=time.clock()
        '''Calculate the disk density and temperature structure given the specified parameters'''
        # Define the desired regular cylindrical (r,z) grid
        nac = 500#256             # - number of unique a rings
        #nrc = 256             # - numver of unique r points
        amin = self.Ain       # - minimum a [AU]
        amax = self.Aout      # - maximum a [AU]
        e = self.ecc          # - eccentricity
        nzc = int(2.5*nac)#nac*5           # - number of unique z points
        '''defining z-array: .1 AU to specified max value in AU. logarithmic. number specified by
        # of annuli'''
        zmin = .1*Disk.AU      # - minimum z [AU]
        nfc = self.nphi       # - number of unique f points

        '''it seems like maybe the log spacing is not playing nice with the spiral model...? 
        swiching to linear for now'''
        af = np.linspace(amin,amax,nac)
        zf = np.linspace(zmin,self.zmax,nzc)

        #adding this to triple check z-dimension is doing what I think it is
        print("1d z-array " + str(zf))

        pf = np.linspace(0,2*np.pi,self.nphi) #f is with refrence to semi major axis
        ff = (pf - self.aop) % (2*np.pi) # phi values are offset by aop- refrence to sky
        rf = np.zeros((nac,nfc))
        for i in range(nac):
            for j in range(nfc):
                rf[i,j] = (af[i]*(1.-e*e))/(1.+e*np.cos(ff[j]))
        
        print("rf (1d) " +str(rf))

        '''1d array of z-values as ones'''
        idz = np.ones(nzc)
        idf = np.ones(self.nphi)

        ida = np.ones(nac)

        #order of dimensions: a, f, z
        '''meshgrid of z values above midplane'''
        pcf,acf,zcf = np.meshgrid(pf,af,zf)

        fcf = (pcf - self.aop) % (2*np.pi)

        
        '''should be 0 grid in shape of radius, phi, z above midplane'''
        rcf=rf[:,:,np.newaxis]*idz


        # Here introduce new z-grid (for now just leave old one in)

        # Interpolate dust temperature and density onto cylindrical grid

        # bundle the grid for helper functions
        '''nac, nfc, nzc are resolution (int) in each dimension. rcf is NOT 0s grid in 3d. 
        amax is max a (AU), zcf is z meshgrid'''
        grid = {'nac':nac,'nfc':nfc,'nzc':nzc,'rcf':rcf,'amax':amax,'zcf':zcf}#'ff':ff,'af':af,
        self.grid=grid

        #define temperature structure
        # use Dartois (03) type II temperature structure
        ###### expanding to 3D should not affect this ######
        delta = 1.                # shape parameter
        rcf150=rcf/(150.*Disk.AU)

        #qq is "temperature index....???"
        rcf150q=rcf150**self.qq
        
        '''# zq0 = Zq, in AU, at 150 AU (????)'''
        '''zq should be 3d and scalled by 150 AU...???'''
        '''z has default shape of 500, 131, 1250 (so yes, 3d grid, but no negative values)'''
        zq = self.zq0*Disk.AU*rcf150**1.3
        
        #zq = self.zq0*Disk.AU*(rcf/(150*Disk.AU))**1.1
        tmid = self.tmid0*rcf150q
        tatm = self.tatm0*rcf150q
        tempg = tatm + (tmid-tatm)*np.cos((np.pi/(2*zq))*zcf)**(2.*delta)

        '''ii is 3d boolean grid of z values above some critical value'''
        ii = zcf > zq
        tempg[ii] = tatm[ii]
        #Type I structure
#        tempg = tmid*np.exp(np.log(tatm/tmid)*zcf/zq)
        ###### this step is slow!!! ######
        #print("temp struct {t}".format(t=time.clock()-tst)

        # Calculate vertical density structure

        ## Circular:
        #Sc = self.McoG*(2.-self.pp)/(2*np.pi*self.Rc*self.Rc)
        #siggas = Sc*(rf/self.Rc)**(-1*self.pp)*np.exp(-1*(rf/self.Rc)**(2-self.pp))
        ## Elliptical:
        #asum = (np.power(af,-1*self.pp)).sum()
        rp1 = np.roll(rf,-1,axis=0)
        rm1 = np.roll(rf,1,axis=0)
        #*** Approximations used here ***#
        Sc = self.McoG*(2.-self.pp)/(self.Rc*self.Rc)
        siggas_r = Sc*(acf[:,:,0]/self.Rc)**(-1*self.pp)*np.exp(-1*(acf[:,:,0]/self.Rc)**(2-self.pp))

        dsdth = (acf[:,:,0]*(1-e*e)*np.sqrt(1+2*e*np.cos(fcf[:,:,0])+e*e))/(1+e*np.cos(fcf[:,:,0]))**2
        '''commenting this out to use my surface density profile for now'''
        #siggas = ((siggas_r*np.sqrt(1.-e*e))/(2*np.pi*acf[:,:,0]*np.sqrt(1+2*e*np.cos(fcf[:,:,0])+e*e)))*dsdth

        '''adding spiral feature
        feature is a perturbation of dsigma/sigma, so I think needs to be multiplied by and added to
        original surface denncity profile
        
        The feaure should be very subtle...'''

        #understanding structure of perturbed_sigma and how to use it
        #Parameters
        ms = 1 #star mass
        md = 0.35 #disc mass
        p = -.5 #surface density
        ap = 60*np.pi/180 #pitch angle
        m = 2 #azimuthal wavenumber
        beta = 5 #cool
        incl = np.pi/2.1 #inclination of the disc towards the line of sight
        pos = 90 # rotation of spiral (degrees), starting north, cw

        #1d r array
        #r = np.linspace(1,100,500)
        #1d phi array
        #phi = np.linspace(-np.pi,np.pi,360)
        #meshgrid but... why not use r array? And what is purpose of j? Indexing?
        #gr, gphi = np.mgrid[1:100:500j, -np.pi:np.pi:360j] #rin:rout:resolution
        #print("gr shape " + str(gr.shape))
        #print("gphi shape " + str(gphi.shape))
        #print("gr" + str(gr))
        #print("gphi" + str(gphi))


        #x_grid, y_grid = pol2cart(acf[:,:,0], pcf[:,:,0])
        #r_cart = 


        gx, gy = np.mgrid[-1000:1000:100j,-1000:1000:100j]
        g_r, g_phi = cart2pol(gx, gy)
        car = np.linspace(-100,100,400)
        grid_angle = 0*gx
        #g_r = (gx**2+gy**2)**(0.5)
        #print(str(gr.shape)+"gr shape")

        #print(str(grid_angle.shape)+"grid_angle shape")
        #print(str(gr.shape)+"gr shape")
        spir0 = giggle.perturbed_sigma(g_r, g_phi, p, self.Ain, self.Aout, md, beta, m, ap,0)

        plt.imshow(spir0)
        plt.colorbar()
        plt.savefig("cart_spir_surf.png")
        plt.show()
        
        
        plt.scatter(g_r, g_phi, c=spir0)
        plt.colorbar()
        plt.savefig("spiral_b4_interp_surf.png")
        plt.show()
        


        interp_test = interpnd((np.ravel(g_r), np.ravel(g_phi)), np.ravel(spir0))
        print("g_r " + str(g_r))
        print("g_phi " + str(g_phi))
        plt.scatter(np.ravel(g_r), np.ravel(g_phi), c=spir0)
        plt.colorbar()
        plt.savefig("density_plotted1darray.png")
        plt.show()
        print("acf [:,:,0] " + str(acf[:,:,0]))
        print("pcf[:,:,0]-np.pi " + str(pcf[:,:,0]-np.pi))

        siggas = interp_test(acf[:,:,0]/Disk.AU, pcf[:,:,0]-np.pi) + 5

        print("siggas " + str(siggas))

        plt.imshow(siggas)
        plt.colorbar()
        plt.savefig("after_interp_surf.png")
        plt.show()


        '''redifining siggas as my own var
        Right now just adding one until I figure out how to actually use surface density profile'''
        #siggas = spir0+1

        #print(spir0.shape)
        #spir1 = giggle.perturbed_sigma(g_r, grid_angle, p, 1, 100, md, beta, m, ap,30)
        #spir2 = giggle.perturbed_sigma(g_r, grid_angle, p, 1, 100, md, beta, m, ap,90)

        ## Add an extra ring
        if self.ring is not None:
            w = np.abs(rcf-self.Rring)<self.Wring/2.
            if w.sum()>0:
                tempg[w] = tempg[w]*(rcdf[w]/(150*Disk.AU))**(self.sig_enhance-self.qq)/((rcf[w].max())/(150.*Disk.AU))**(-self.qq+self.sig_enhance)

        
        
        

        self.calc_hydrostatic(tempg,siggas,grid)


        #https://pdfs.semanticscholar.org/75d1/c8533025d0a7c42d64a7fef87b0d96aba47e.pdf
        #Lovis & Fischer 2010, Exoplanets edited by S. Seager (eq 11 assuming m2>>m1)
        #self.vel = np.sqrt(Disk.G*self.Mstar/(acf*(1-self.ecc**2.)))*(np.cos(self.aop+fcf)+self.ecc*self.cosaop)
        #print("self.vel shape " + str(self.vel.shape))

        


        '''for spiral model, we want to make copies of the spiral extending in the z direction. Right now should
        just be above the midplane I think. '''

        ms = 1 #star mass
        md = 0.35 #disc mass
        p = -.5 #surface density
        ap = 13*np.pi/180 #pitch angle
        m = 2 #azimuthal wavenumber
        beta = 5 #cool
        incl = np.pi/2.1 #inclination of the disc towards the line of sight
        pos = 90 # rotation of spiral (degrees), starting north, cw

        #self.vel = giggle.momentone(rcf, pcf, ms, md, p, m, 1, beta, amin, amax, ap, 0, 0)

        
        #self.vel_phi = giggle.uph(rcf, pcf, ms, md, p, m, 1, beta, amin, amax, ap, 0)[:,:,np.newaxis]*idz
        phi_vel = giggle.uphC(gx, gy, ms, md, p, m, 1, beta, amin, amax, ap, 0)
        #print("self.vel_phi shape " + str(self.vel_phi.shape))
        rad_vel = giggle.urC(gx, gy, ms, md, p, 2, 1, beta, amin, amax, ap, 0) 
        #print("self.vel_rad shape " + str(self.vel_rad.shape))

        plt.scatter(g_r, g_phi, c=phi_vel)
        plt.colorbar()
        plt.savefig("vel_phi_polarscatter.png")
        plt.show()

        interp_test_phi = interpnd((np.ravel(g_r), np.ravel(g_phi)), np.ravel(phi_vel))
        interp_test_rad = interpnd((np.ravel(g_r), np.ravel(g_phi)), np.ravel(rad_vel))

        self.vel_phi = interp_test_phi(acf[:,:,0]/Disk.AU, pcf[:,:,0]-np.pi)[:,:,np.newaxis]*idz
        self.vel_rad = interp_test_rad(acf[:,:,0]/Disk.AU, pcf[:,:,0]-np.pi)[:,:,np.newaxis]*idz

        plt.imshow(self.vel_phi[:,:,0])
        plt.colorbar()
        plt.savefig("phi_vel_afterinterp.png")
        plt.show()

        plt.imshow(self.vel_rad[:,:,0])
        plt.colorbar()
        plt.savefig("rad_vel_afterinterp.png")
        plt.show()


        #self.vel = np.sqrt(self.vel_phi**2 + self.vel_rad **2)
        self.vel = self.vel_phi

        #fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        #scatter = ax.scatter(pcf[:,:,0], acf[:,:,0], c=self.vel_phi, label="phi velocity, face on")
        #plt.colorbar(scatter, ax=ax)

        #plt.imshow(self.vel[:,:,0])

        
        #plt.savefig("phi_vel.png", dpi = 300)

        ii = np.isnan(self.rho0)
        if ii.sum() > 0:
            self.rho0[ii] = 1e-60
            print('Beware: removed NaNs from density (#%s)' % ii.sum())
        ii = np.isnan(tempg)
        if ii.sum() > 0:
            tempg[ii] = 2.73
            print('Beware: removed NaNs from temperature (#%s)' % ii.sum())

        # find photodissociation boundary layer from top
        zpht_up = np.zeros((nac,nfc))
        zpht_low = np.zeros((nac,nfc))
        sig_col = np.zeros((nac,nfc,nzc))
        #zice = np.zeros((nac,nfc))
        for ia in range(nac):
            for jf in range (nfc):
                psl = (Disk.Hnuctog/Disk.m0*self.rho0[ia,jf,:])[::-1]
                zsl = self.zmax - (zcf[ia,jf,:])[::-1]
                foo = (zsl-np.roll(zsl,1))*(psl+np.roll(psl,1))/2.
                foo[0] = 0
                nsl = foo.cumsum()
                sig_col[ia,jf,:] = nsl[::-1]*Disk.m0/Disk.Hnuctog
                pht = (np.abs(nsl) >= self.sigbound[0])
                if pht.sum() == 0:
                    zpht_up[ia,jf] = np.min(self.zmax-zsl)
                else:
                    zpht_up[ia,jf] = np.max(self.zmax-zsl[pht])
                #Height of lower column density boundary
                pht = (np.abs(nsl) >= self.sigbound[1])
                if pht.sum() == 0:
                    zpht_low[ia,jf] = np.min(self.zmax-zsl)
                else:
                    zpht_low[ia,jf] = np.max(self.zmax-zsl[pht])
                #used to be a seperate loop
                ###### only used for plotting
                #foo = (tempg[ia,jf,:] < Disk.Tco)
                #if foo.sum() > 0:
                #    zice[ia,jf] = np.max(zcf[ia,jf,foo])
                #else:
                #    zice[ia,jf] = zmin
        self.sig_col = sig_col
        #szpht = zpht
        #print("Zpht {t} seconds".format(t=(time.clock()-tst)))

        self.af = af
        #self.ff = ff
        #self.rf = rf
        self.pf = pf
        self.nac = nac
        self.zf = zf
        self.nzc = nzc
        self.tempg = tempg
        #self.Omg0 = Omg#velrot
        self.zpht_up = zpht_up
        self.zpht_low = zpht_low
        self.pcf = pcf  #only used for plotting can remove after testing
        self.rcf = rcf  #only used for plotting can remove after testing


    def set_rt_grid(self):
        #tst=time.clock()
        ### Start of Radiative Transfer portion of the code...
        # Define and initialize cylindrical grid
        #Smin = 1*Disk.AU                 # offset from zero to log scale
        #if self.thet > np.arctan(self.Aout/self.zmax):
        #    Smax = 2*self.Aout/self.sinthet
        #else:
        #    Smax = 2.*self.zmax/self.costhet       # los distance through disk
        #Smid = Smax/2.                    # halfway along los
        #ytop = Smax*self.sinthet/2.       # y origin offset for observer xy center
        #sky coordinates
        #R = np.logspace(np.log10(self.Ain*(1-self.ecc)),np.log10(self.Aout*(1+self.ecc)),self.nr)
        R = np.linspace(0,self.Aout*(1+self.ecc),self.nr) #******* not on cluster*** #
        phi = np.arange(self.nphi)*2*np.pi/(self.nphi-1)
        #foo = np.floor(self.nz/2)

        #S_old = np.concatenate([Smid+Smin-10**(np.log10(Smid)+np.log10(Smin/Smid)*np.arange(foo)/(foo)),Smid-Smin+10**(np.log10(Smin)+np.log10(Smid/Smin)*np.arange(foo)/(foo))])
        #S_old = np.arange(2*foo)/(2*foo)*(Smax-Smin)+Smin #*** not on cluster**

        #print("grid {t}".format(t=time.clock()-tst))
        # Basically copy S_old, with length nz,  into each column of a nphi*nr*nz matrix
        #S = (S_old[:,np.newaxis,np.newaxis]*np.ones((self.nr,self.nphi))).T

        '''my guess is that z reflection happen here'''

        # arrays in [phi,r,s] on sky coordinates
        X = (np.outer(R,np.cos(phi))).transpose()
        Y = (np.outer(R,np.sin(phi))).transpose()

        #Use a rotation matrix to transform between radiative transfer grid and physical structure grid
        if np.abs(self.thet) > np.arctan(self.Aout*(1+self.ecc)/self.zmax):
            zsky_max = np.abs(2*self.Aout*(1+self.ecc)/self.sinthet)
        else:
            zsky_max = 2*(self.zmax/self.costhet)

        '''this looks like the reflection...??'''
        '''I think nz is number of z points but not sure how it's different from nzc
        theta is inclination'''
        zsky = np.arange(self.nz)/self.nz*(-zsky_max)+zsky_max/2.
        
        '''are these x & y projected onto sky plane..?'''
        tdiskZ = (Y.repeat(self.nz).reshape(self.nphi,self.nr,self.nz))*self.sinthet+zsky*self.costhet
        tdiskY = (Y.repeat(self.nz).reshape(self.nphi,self.nr,self.nz))*self.costhet-zsky*self.sinthet
        if (self.thet<np.pi/2) & (self.thet>0):

            '''what is this theta_crit value...?'''
            theta_crit = np.arctan((self.Aout*(1+self.ecc)+tdiskY)/(self.zmax-tdiskZ))
            S = (self.zmax-tdiskZ)/self.costhet
            S[(theta_crit<self.thet)] = ((self.Aout*(1+self.ecc)+tdiskY[(theta_crit<self.thet)])/self.sinthet)
        elif self.thet>np.pi/2:
            theta_crit = np.arctan((self.Aout*(1+self.ecc)+tdiskY)/(self.zmax+tdiskZ))
            S = -(self.zmax+tdiskZ)/self.costhet
            S[(theta_crit<(np.pi-self.thet))] = ((self.Aout*(1+self.ecc)+tdiskY[(theta_crit<(np.pi-self.thet))])/self.sinthet)
        elif (self.thet<0) & (self.thet>-np.pi/2):
            theta_crit = np.arctan((self.Aout*(1+self.ecc)-tdiskY)/(self.zmax-tdiskZ))
            S = (self.zmax-tdiskZ)/self.costhet
            S[(theta_crit<np.abs(self.thet))] = -((self.Aout*(1+self.ecc)-tdiskY[(theta_crit<np.abs(self.thet))])/self.sinthet)

        # transform grid to disk coordinates
        #tdiskZ = self.zmax*(np.ones((self.nphi,self.nr,self.nz)))-self.costhet*S
        #if self.thet > np.arctan(self.Aout/self.zmax):
        #    tdiskZ -=(Y*self.sinthet).repeat(self.nz).reshape(self.nphi,self.nr,self.nz)
        #tdiskY = ytop - self.sinthet*S + (Y/self.costhet).repeat(self.nz).reshape(self.nphi,self.nr,self.nz)
        tr = np.sqrt(X.repeat(self.nz).reshape(self.nphi,self.nr,self.nz)**2+tdiskY**2)
        tphi = np.arctan2(tdiskY,X.repeat(self.nz).reshape(self.nphi,self.nr,self.nz))%(2*np.pi)
        ###### should be real outline? requiring a loop over f or just Aout(1+ecc)######
        notdisk = (tr > self.Aout*(1.+self.ecc)) | (tr < self.Ain*(1-self.ecc))  # - individual grid elements not in disk
        isdisk = (tr>self.Ain*(1-self.ecc)) & (tr<self.Aout*(1+self.ecc)) & (np.abs(tdiskZ)<self.zmax)
        S -= S[isdisk].min() #Reset the min S to 0
        #xydisk =  tr[:,:,0] <= self.Aout*(1.+self.ecc)+Smax*self.sinthet  # - tracing outline of disk on observer xy plane
        self.r = tr




        #print("new grid {t}".format(t=time.clock()-tst))
        # Here include section that redefines S along the line of sight
        # (for now just use the old grid)

        # interpolate to calculate disk temperature and densities
        #print('interpolating onto radiative transfer grid')
        #need to interpolate tempg from the 2-d rcf,zcf onto 3-d tr
        # x is xy plane, y is z axis
        ###### rf is 2d, zf is still 1d ######
        #xind = np.interp(tr.flatten(),self.rf,range(self.nrc)) #rf,nrc
        #yind = np.interp(np.abs(tdiskZ).flatten(),self.zf,range(self.nzc)) #zf,nzc
        #indices in structure arrays of coordinates in transform grid`
        zind = np.interp(np.abs(tdiskZ).flatten(),self.zf,range(self.nzc)) #zf,nzc
        phiind = np.interp(tphi.flatten(),self.pf,range(self.nphi))
        aind = np.interp((tr.flatten()*(1+self.ecc*np.cos(tphi.flatten()-self.aop)))/(1.-self.ecc**2),self.af,range(self.nac),right=self.nac)


        #print("index interp {t}".format(t=time.clock()-tst))
        ###### fixed T,Omg,rhoG still need to work on zpht ######
        tT = ndimage.map_coordinates(self.tempg,[[aind],[phiind],[zind]],order=1,cval=1e-18).reshape(self.nphi,self.nr,self.nz) #interpolate onto coordinates xind,yind #tempg
        #Omgx = ndimage.map_coordinates(self.Omg0[0],[[aind],[phiind],[zind]],order=1,cval=1e-18).reshape(self.nphi,self.nr,self.nz) #Omgs
        #Omg = ndimage.map_coordinates(self.Omg0,[[aind],[phiind],[zind]],order=1,cval=1e-18).reshape(self.nphi,self.nr,self.nz) #Omgy
        
        '''this is where I need to make sure velocity map is being used correctly'''

        plt.imshow(self.vel_phi[:,:,0])
        plt.savefig('vel_phi_beforecoord.png', dpi = 300)
        plt.show()

        #modified to use phi and r velocities
        tvelphi = ndimage.map_coordinates(self.vel_phi,[[aind],[phiind],[zind]],order=1).reshape(self.nphi,self.nr,self.nz)*Disk.kms
        tvelr = ndimage.map_coordinates(self.vel_rad,[[aind],[phiind],[zind]],order=1).reshape(self.nphi,self.nr,self.nz)*Disk.kms
        tvel = ndimage.map_coordinates(self.vel,[[aind],[phiind],[zind]],order=1).reshape(self.nphi,self.nr,self.nz)

        plt.imshow(tvelphi[:,:,0])
        plt.colorbar()
        plt.savefig("tvelphi.png")
        plt.show()
        plt.imshow(tvelr[:,:,0])
        plt.colorbar()
        plt.savefig("tvelr.png")
        plt.show()

        self.p_grid = ndimage.map_coordinates(self.pcf,[[aind],[phiind],[zind]],order=1).reshape(self.nphi,self.nr,self.nz)
        plt.imshow(self.p_grid[:,:,0])
        plt.colorbar()
        plt.savefig("p_grid.png")
        plt.show()
        
        #Omgz = np.zeros(np.shape(Omgy))
        #trhoG = Disk.H2tog*self.Xmol/Disk.m0*ndimage.map_coordinates(self.rho0,[[aind],[phiind],[zind]],order=1,cval=1e-18).reshape(self.nphi,self.nr,self.nz)
        #trhoH2 = trhoG/self.Xmol #** not on cluster**
        #zpht = np.interp(tr.flatten(),self.rf,self.zpht).reshape(self.nphi,self.nr,self.nz) #tr,rf,zpht
        tsig_col = ndimage.map_coordinates(self.sig_col,[[aind],[phiind],[zind]],order=1,cval=1e-18).reshape(self.nphi,self.nr,self.nz)
        zpht_up = ndimage.map_coordinates(self.zpht_up,[[aind],[phiind]],order=1).reshape(self.nphi,self.nr,self.nz) #tr,rf,zpht
        zpht_low = ndimage.map_coordinates(self.zpht_low,[[aind],[phiind]],order=1).reshape(self.nphi,self.nr,self.nz) #tr,rf,zpht
        tT[notdisk] = 0
        self.sig_col = tsig_col

        self.add_mol_ring(self.Rabund[0]/Disk.AU,self.Rabund[1]/Disk.AU,self.sigbound[0]/Disk.sc,self.sigbound[1]/Disk.sc,self.Xco,initialize=True)

        if np.size(self.Xco)>1:
            Xmol = self.Xco[0]*np.exp(-(self.Rabund[0]-tr)**2/(2*self.Rabund[3]**2))+self.Xco[1]*np.exp(-(self.Rabund[1]-tr)**2/(2*self.Rabund[4]**2))+self.Xco[2]*np.exp(-(self.Rabund[2]-tr)**2/(2*self.Rabund[5]**2))
        #else:
        #    Xmol = self.Xco



        #print("image interp {t}".format(t=time.clock()-tst))

        # photo-dissociation
        #zap = (np.abs(tdiskZ) > zpht_up)
        #if zap.sum() > 0:
        #    trhoG[zap] = 1e-18*trhoG[zap]
        #zap = (np.abs(tdiskZ) < zpht_low)
        #if zap.sum()>0:
        #    trhoG[zap] = 1e-18*trhoG[zap]

        #if np.size(self.Xco)<2:
        #    #Inner and outer abundance boundaries
        #    zap = (tr<=self.Rabund[0]) | (tr>=self.Rabund[1])
        #    if zap.sum()>0:
        #        trhoG[zap] = 1e-18*trhoG[zap]

        # freeze out
        zap = (tT <= Disk.Tco)
        if zap.sum() >0:
            self.Xmol[zap] = 1e-8*self.Xmol[zap]
            #trhoG[zap] = 1e-8*trhoG[zap]

        trhoH2 = Disk.H2tog/Disk.m0*ndimage.map_coordinates(self.rho0,[[aind],[phiind],[zind]],order=1,cval=1e-18).reshape(self.nphi,self.nr,self.nz)
        trhoG = trhoH2*self.Xmol
        trhoH2[notdisk] = 0
        trhoG[notdisk] = 0
        self.rhoH2 = trhoH2

        #print("zap {t}".format(t=time.clock()-tst))
        #temperature and turbulence broadening
        #moved this to the set_line method
        #tdBV = np.sqrt(2.*Disk.kB/(Disk.Da*Disk.mCO)*tT+self.vturb**2)
        #tdBV = np.sqrt((1+(self.vturb/Disk.kms)**2.)*(2.*Disk.kB/(Disk.Da*Disk.mCO)*tT)) #vturb proportional to cs



        # store disk
        self.X = X
        self.Y = Y
        self.Z = tdiskZ
        self.S = S
        #self.r = tr
        self.T = tT
        #self.dBV = tdBV
        self.rhoG = trhoG
        #self.Omg = Omg#Omgy #need to combine omgx,y,z

        '''changed these to include r and phi velocity values'''
        self.vel_rad = tvelr
        self.vel_phi = tvelphi
        self.vel = tvel
        print("vel_rad shape " + str(self.vel_rad.shape))
        print("vel_phi shape " + str(self.vel_rad.shape))

        plt.imshow(self.vel_rad[:,:,0])
        plt.savefig('pvel_rad_z=0.png', dpi = 300)
        plt.show()
        
        plt.imshow(self.vel_rad[:,:,10])
        plt.savefig('pvel_rad_z=10.png', dpi = 300)
        plt.show()

        plt.imshow(self.vel_phi[:,:,0])
        plt.savefig('pvel_phi_z=0.png', dpi = 300)
        plt.show()
         
        plt.imshow(self.vel_phi[:,:,10])
        plt.savefig('pvel_phi_z=10.png', dpi = 300)
        plt.show()
        

        self.i_notdisk = notdisk
        #self.i_xydisk = xydisk
        #self.rhoH2 = trhoH2 #*** not on cluster ***
        #self.sig_col=tsig_col
        #self.Xmol = Xmol
        self.cs = np.sqrt(2*self.kB/(self.Da*2)*self.T)
        #self.tempg = tempg
        #self.zpht = zpht
        #self.phi = tphi

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
            #tdBV = np.sqrt((1+(self.vturb/Disk.kms)**2.)*(2.*Disk.kB/(Disk.Da*Disk.mCO)*self.T)) #vturb proportional to cs

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
        ''' Add a ring of fixed abundance, between Rin and Rout (in the radial direction) and Sig0 and Sig1 (in the vertical direction). The abundance is treated as a power law in the radial direction, with alpha as the power law exponent, and normalized at the inner edge of the ring (abund~abund0*(r/Rin)^(alpha))
        disk.add_mol_ring(10,100,.79,1000,1e-4)
        just_frozen: only apply the abundance adjustment to the areas of the disk where CO is nominally frozen out.'''
        if initialize:
            self.Xmol = np.zeros(np.shape(self.r))+1e-18
        if just_frozen:
            add_mol = (self.sig_col*Disk.Hnuctog/Disk.m0>Sig0*Disk.sc) & (self.sig_col*Disk.Hnuctog/Disk.m0<Sig1*Disk.sc) & (self.r>Rin*Disk.AU) & (self.r<Rout*Disk.AU) & (self.T<self.Tco)
        else:
            add_mol = (self.sig_col*Disk.Hnuctog/Disk.m0>Sig0*Disk.sc) & (self.sig_col*Disk.Hnuctog/Disk.m0<Sig1*Disk.sc) & (self.r>Rin*Disk.AU) & (self.r<Rout*Disk.AU)
        if add_mol.sum()>0:
            self.Xmol[add_mol]+=abund*(self.r[add_mol]/(Rin*Disk.AU))**(alpha)
        #add soft boundaries
        edge1 = (self.sig_col*Disk.Hnuctog/Disk.m0>Sig0*Disk.sc) & (self.sig_col*Disk.Hnuctog/Disk.m0<Sig1*Disk.sc) & (self.r>Rout*Disk.AU)
        if edge1.sum()>0:
            self.Xmol[edge1] += abund*(self.r[edge1]/(Rin*Disk.AU))**(alpha)*np.exp(-(self.r[edge1]/(Rout*Disk.AU))**16)
        edge2 = (self.sig_col*Disk.Hnuctog/Disk.m0>Sig0*Disk.sc) & (self.sig_col*Disk.Hnuctog/Disk.m0<Sig1*Disk.sc) & (self.r<Rin*Disk.AU)
        if edge2.sum()>0:
            self.Xmol[edge2] += abund*(self.r[edge2]/(Rin*Disk.AU))**(alpha)*(1-np.exp(-(self.r[edge2]/(Rin*Disk.AU))**20.))
        edge3 = (self.sig_col*Disk.Hnuctog/Disk.m0<Sig0*Disk.sc) & (self.r>Rin*Disk.AU) & (self.r<Rout*Disk.AU)
        if edge3.sum()>0:
            self.Xmol[edge3] += abund*(self.r[edge3]/(Rin*Disk.AU))**(alpha)*(1-np.exp(-((self.sig_col[edge3]*Disk.Hnuctog/Disk.m0)/(Sig0*Disk.sc))**8.))
        zap = (self.Xmol<0)
        if zap.sum()>0:
            self.Xmol[zap]=1e-18
        if not initialize:
            self.rhoG = self.rhoH2*self.Xmol



    def calc_hydrostatic(self,tempg,siggas,grid):
        nac = grid['nac']
        nfc = grid['nfc']
        nzc = grid['nzc']
        rcf = grid['rcf']
        zcf = grid['zcf']
        dz = (zcf - np.roll(zcf,1))#,axis=2))

        #compute rho structure
        rho0 = np.zeros((nac,nfc,nzc))
        sigint = siggas

        #compute gravo-thermal constant
        grvc = Disk.G*self.Mstar*Disk.m0/Disk.kB

        #t1 = time.clock()
        #differential equation for vertical density profile
        dlnT = (np.log(tempg)-np.roll(np.log(tempg),1,axis=2))/dz
        dlnp = -1.*grvc*zcf/(tempg*(rcf**2+zcf**2)**1.5)-dlnT
        dlnp[:,:,0] = -1.*grvc*zcf[:,:,0]/(tempg[:,:,0]*(rcf[:,:,0]**2.+zcf[:,:,0]**2.)**1.5)

        #numerical integration to get vertical density profile
        foo = dz*(dlnp+np.roll(dlnp,1,axis=2))/2.
        foo[:,:,0] = np.zeros((nac,nfc))
        lnp = foo.cumsum(axis=2)

        #normalize the density profile (note: this is just half the sigma value!)
        rho0 = 0.5*((sigint/np.trapz(np.exp(lnp),zcf,axis=2))[:,:,np.newaxis]*np.ones(nzc))*np.exp(lnp)
        #t2=time.clock()
        #print("hydrostatic loop took {t} seconds".format(t=(t2-t1)))

        #print('Doing hydrostatic equilibrium')
        #t1 = time.clock()
        #for ia in range(nac):
        #    for jf in range(nfc):
        #
        #        #extract the T(z) profile at a given radius
        #        T = tempg[ia,jf]
        #
        #        z=zcf[ia,jf]
        #        #differential equation for vertical density profile
        #        dlnT = (np.log(T)-np.roll(np.log(T),1))/dz[ia,jf]
        #        dlnp = -1*grvc*z/(T*(rcf[ia,jf]**2.+z**2.)**1.5)-dlnT
        #        dlnp[0] = -1*grvc*z[0]/(T[0]*(rcf[ia,jf,0]**2.+z[0]**2.)**1.5)
        #
        #        #numerical integration to get vertical density profile
        #        foo = dz[ia,jf]*(dlnp+np.roll(dlnp,1))/2.
        #        foo[0] = 0.
        #        lnp = foo.cumsum()
        #
        #        #normalize the density profile (note: this is just half the sigma value!)
        #        #print(lnp.shape,grvc.shape,z.shape,T.shape,rcf[ia,jf].shape,dlnT.shape)
        #        dens = 0.5*sigint[ia,jf]*np.exp(lnp)/np.trapz(np.exp(lnp),z)
        #        rho0[ia,jf,:] = dens
        #        #if ir == 200:
        #        #    plt.plot(z/Disk.AU,dlnT)
        #        #    plt.plot(z/Disk.AU,dlnp)
        #t2=time.clock()
        #print("hydrostatic loop took {t} seconds".format(t=(t2-t1)))

        self.rho0=rho0
        #print(Disk.G,self.Mstar,Disk.m0,Disk.kB)
        if 0:
            print('plotting')
            plt.pcolor(rf[:,0,np.newaxis]*np.ones(nzc),zcf[:,0,:],np.log10(rho0[:,0,:]))
            plt.colorbar()
            plt.show()


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
        params.append(self.Ain/Disk.AU)
        params.append(self.Aout/Disk.AU)
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

    def plot_structure(self,sound_speed=False,beta=None,dust=False,rmax=500,zmax=170):
        ''' Plot temperature and density structure of the disk'''
        plt.figure()
        plt.rc('axes',lw=2)
        print("self.Z shape "+str(self.Z.shape))
        print("self.Z[0,:,:] shape "+str(self.Z[0,:,:,].shape))
        print("z[0,:,:,] contents "+str(self.Z[0,:,:]))
        cs2 = plt.contour(self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,np.log10(self.rhoG[0,:,:])+4,np.arange(0,11,0.1))
        cs2 = plt.contour(-self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,np.log10(self.rhoG[int(self.nphi/2),:,:])+4,np.arange(0,11,0.1))
        ax = plt.gca()
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
            tick.label1.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
            tick.label1.set_fontweight('bold')
        if sound_speed:
            cs = self.r*self.Omg#np.sqrt(2*self.kB/(self.Da*self.mCO)*self.T)
            cs3 = plt.contour(self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,cs[0,:,:]/Disk.kms,100,colors='k')
            cs3 = plt.contour(-self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,cs[int(self.nphi/2.),:,:]/Disk.kms,100,colors='k')
            plt.clabel(cs3)
        elif beta is not None:
            cs = np.sqrt(2*self.kB/(self.Da*self.mu)*self.T)
            rho = (self.rhoG+4)*self.mu*self.Da #mass density
            Bmag = np.sqrt(8*np.pi*rho*cs**2/beta) #magnetic field
            cs3 = plt.contour(self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,np.log10(Bmag[0,:,:]),20)
            cs3 = plt.contour(-self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,np.log10(Bmag[int(self.nphi/2.),:,:]),20)
        elif dust:
            cs3 = plt.contour(self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,np.log10(self.rhoD[0,:,:]),100,colors='k',linestyles='--')
            cs3 = plt.contour(-self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,np.log10(self.rhoD[int(self.nphi/2.),:,:]),100,colors='k',linestyles='--')
        else:
            cs3 = plt.contour(self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,self.T[0,:,:],(20,40,60,80,100,120),colors='k',ls='--')
            cs3 = plt.contour(-self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,self.T[int(self.nphi/2.),:,:],(20,40,60,80,100,120),colors='k',ls='--')
            plt.clabel(cs3,fmt='%1i')
        plt.colorbar(cs2,label='log n')
        plt.xlim(-1*rmax,rmax)
        plt.ylim(0,zmax)
        plt.xlabel('R (AU)',fontsize=20)
        plt.ylabel('Z (AU)',fontsize=20)
        plt.show()

    def calcH(self,verbose=True):
        ''' Calculate the equivalent of the pressure scale height within our disks. This is useful for comparison with other models that take this as a free parameter. H is defined as 2^(-.5) times the height where the density drops by 1/e. (The factor of 2^(-.5) is included to be consistent with a vertical density distribution that falls off as exp(-z^2/2H^2))'''
        ###### this method does not work with the elliptical disk (must expand to 3d) ######
        nrc = self.nrc
        zf = self.zf
        rf = self.rf
        rho0 = self.rho0

        H = np.zeros(nrc)
        for i in range(nrc):
            rho_cen = rho0[i,0]
            diff = abs(rho_cen/np.e-rho0[i,:])
            H[i] = zf[(diff == diff.min())]/np.sqrt(2.)

        if verbose:
            H100 = np.interp(100*Disk.AU,rf,H)
            psi = (np.polyfit(np.log10(rf),np.log10(H),1))[0]
            #print(H100/Disk.AU)
            #print(psi)
            print('H100 (AU): {:.3f}'.format(H100/Disk.AU))
            print('power law: {:.3f}'.format(psi))

        return H
