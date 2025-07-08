#
# Cristiano Longarini
#
# Units:
# - distances in au
# - mass in msun
# - velocities in km/s
#

import matplotlib.pyplot as plt
import scipy
import numpy as np
import math
import matplotlib.image as mpimg
from scipy.interpolate import griddata
from scipy import special
import scipy.integrate as integrate
from scipy.integrate import quad
from scipy.integrate import simps
from astropy import constants as const
from scipy.interpolate import griddata
from scipy import ndimage as ndimage

G = 4.30091e-3 * 206265 

def omega(ms,r):
    
    '''Keplerian frequency in [Hz]
    ms = mass of the central object [msun]
    r = radius [au]'''
   
    return np.sqrt(G*ms/r**3)


'''what is zeta?'''
def zeta(r1,r,z):
    
    return np.sqrt((4*r1*r )/ ((r+r1)**2 +z**2))



def sigmain(p, rin, rout, md):
    
    x = rout/rin
    return ((2+p)*md)/(2*np.pi*rin**2) * (x**(2+p) -1)**(-1)



def sigma(p, rin, rout ,md, r):
    
    '''Surface density of the disc in [Msun/au^2]
    p = power law index of the density profile. \Sigma \propto r^(p)
    rin = inner radius of the disc [au]
    rout = outer radius of the disc [au]
    md = mass of the disc [msun]
    r = radius [au]'''
    
    return sigmain(p,rin,rout,md) * (r/rin)**(+p)



def integrand(r1, r, z, md, p, rin, rout):
    
    zet = zeta(r1,r,z)
    kappa = scipy.special.ellipk(zet)
    ellip = scipy.special.ellipe(zet)
    
    return (kappa - 1/4 * (zet**2 /(1-zet**2)) * (r1/r -r/r1+ 
           (z**2)/(r*r1))*ellip)*np.sqrt(r1/r)*zet*sigma(p,rin,rout,md,r1)



def veldisc(r, z, md, p, rin, rout):
    
    def expint(r,z,md,p,rin,rout):
        return quad(integrand, 1/2 * rin, 2*rout, args=(r,z,md,p,rin,rout))[0]
    vec_expint = np.vectorize(expint)
    
    return G  * vec_expint(r,z,md,p,rin,rout)



def basicspeed(r, z, md, p, rin, rout, ms):
    
    '''Rotation curve of a self gravitating disc [km/s]
    r = radius [au]
    z = height [au] (midplane z=1e-3)
    md = mass of the disc [msun]
    p = power law index of the density profile. \Sigma \propto r^(p)
    rin = inner radius of the disc [au]
    rout = outer radius of the disc [au]
    ms = mass of the central object [msun]'''
    
    return np.sqrt( (G*ms/r) + veldisc(r,z,md,p,rin,rout))



def q(ms, md, p, rin, rout, r):
    
    '''Disc to star mass ratio for a Q=1 disc
    ms = mass of the central object [msun]
    md = mass of the disc [msun]
    p = power law index of the density profile. \Sigma \propto r^(p)
    rin = inner radius of the disc [au]
    rout = outer radius of the disc [au]
    r = radius [au]'''
    
    q_ext = md / ms
    
    return q_ext * (rout/rin)**(-2-p) * (r/rin)**(2+p)



def ura(ms, md, p, m, chi, beta, rin, rout, r):
    
    '''Module of the radial velocity perturbation [km/s]
    ms = mass of the central object [msun]
    md = mass of the disc [msun]
    p = power law index of the density profile. \Sigma \propto r^(p)
    m = number of the spiral arms
    chi = heating factors (=1)
    beta = cooling factor
    rin = inner radius of the disc [au]
    rout = outer radius of the disc [au]
    r = radius [au]'''

    #plt.imshow((2 * m * chi * beta**(-1/2) * q(ms, md, p, rin, rout, r)**2 * omega(ms,r) * r)[:,:,0])
    #plt.savefig("ura_output.png")
    
    return 2 * m * chi * beta**(-1/2) * q(ms, md, p, rin, rout, r)**2 * omega(ms,r) * r



def upha(ms, md, p, m, chi, beta, rin, rout, r):
    
    '''Module of the azimuthal velocity perturbation [km/s]
    ms = mass of the central object [msun]
    md = mass of the disc [msun]
    p = power law index of the density profile. \Sigma \propto r^(p)
    m = number of the spiral arms
    chi = heating factors (=1)
    beta = cooling factor
    rin = inner radius of the disc [au]
    rout = outer radius of the disc [au]
    r = radius [au]'''

    #plt.imshow((- (m * chi * beta**(-1/2)) / 2  * q(ms, md, p, rin, rout, r) * omega(ms,r) * r)[:,:,0])
    #plt.savefig("upha_output.png")
    
    return - (m * chi * beta**(-1/2)) / 2  * q(ms, md, p, rin, rout, r) * omega(ms,r) * r



def ur(grid_radius, grid_angle, ms, md, p, m, chi, beta, rin, rout, alpha, off):
    
    '''2D radial velocity perturbation [km/s] in polar coordinates
    grid_radius = radial grid [au]
    grid_angle = azimuthal grid [-np.pi,np.pi] [rad]
    ms = mass of the central object [msun]
    md = mass of the disc [msun]
    p = power law index of the density profile. \Sigma \propto r^(p)
    m = number of the spiral arms
    chi = heating factors (=1)
    beta = cooling factor
    rin = inner radius of the disc [au]
    rout = outer radius of the disc [au]
    alpha = pitch angle of the spiral [rad]'''

    plt.imshow((- ura(ms, md, p, m, chi, beta, rin, rout, grid_radius)  * np.sin(
        m * grid_angle + m/np.tan(alpha) * np.log(grid_radius) + off))[:,:,0])
    plt.savefig("ur_output.png")
    
    return - ura(ms, md, p, m, chi, beta, rin, rout, grid_radius)  * np.sin(
        m * grid_angle + m/np.tan(alpha) * np.log(grid_radius) + off)




def uph(grid_radius, grid_angle, ms, md, p, m, chi, beta, rin, rout, alpha, off):
    
    '''2D azimuthal velocity perturbation [km/s] in polar coordinates
    grid_radius = radial grid [au]
    grid_angle = azimuthal grid [-np.pi,np.pi] [rad]
    ms = mass of the central object [msun]
    md = mass of the disc [msun]
    p = power law index of the density profile. \Sigma \propto r^(p)
    m = number of the spiral arms
    chi = heating factors (=1)
    beta = cooling factor
    rin = inner radius of the disc [au]
    rout = outer radius of the disc [au]
    alpha = pitch angle of the spiral [rad]'''
    
    x = np.linspace(rin,rout,grid_radius.shape[0])
    phase = m / np.tan(alpha)  * np.log(x)
    '''kevin's grid defines phi 0 to 2pi, so trying to match that here'''
    an = np.linspace(0,2*np.pi,grid_radius.shape[1])
    bs = basicspeed(x, 0.001, md, p, rin, rout, ms)
    vec = np.zeros([grid_radius.shape[0],grid_radius.shape[1]])
    vp1 = upha(ms, md, p, m, chi, beta, rin, rout, rin)
    for i in range(grid_radius.shape[1]):
        vec[:,i] = bs[:] - vp1* x**(3/2 + p) /rin * np.sin(m*an[i] + phase[:] + off)

    plt.imshow(vec)
    plt.savefig("uph_output.png")
        
    return vec



def momentone(grid_radius, grid_angle, ms, md, p, m, chi, beta, rin, rout, alpha, incl, off):
    
    '''Moment one map / projected velocity field towards the line of sight [km/s]
    grid_radius = radial grid [au]
    grid_angle = azimuthal grid [-np.pi,np.pi] [rad]
    ms = mass of the central object [msun]
    md = mass of the disc [msun]
    p = power law index of the density profile. \Sigma \propto r^(p)
    m = number of the spiral arms
    chi = heating factors (=1)
    beta = cooling factor
    rin = inner radius of the disc [au]
    rout = outer radius of the disc [au]
    alpha = pitch angle of the spiral [rad]
    incl = inclination angle [rad]
    NB: The moment one map is given in polar coordinates, the disk is face on and the observer is rotated by an angle incl'''
    
    return uph(grid_radius, grid_angle, ms, md, p, m, chi, beta, rin, rout, alpha, off) * np.cos(
        grid_angle) * np.sin(incl) + ur(grid_radius, grid_angle, ms, md, p, m, chi, beta,
        rin, rout, alpha, off) * np.sin(grid_angle) * np.sin(incl)



def momentone_keplerian(grid_radius, grid_angle, ms, incl):
    
    '''Keplerian moment one map / projected velocity field towards the line of sight [km/s]
    grid_radius = radial grid [au]
    grid_angle = azimuthal grid [-np.pi,np.pi] [rad]
    ms = mass of the central object [msun]
    incl = inclination angle [rad]'''
    
    vk = (omega(ms, grid_radius) * grid_radius)
    
    return vk * np.cos(grid_angle) * np.sin(incl)



def perturbed_sigma(grid_radius, grid_angle, p, rin, rout ,md, beta, m, alpha, pos):
    
    '''Spiral-perturbed surface density [msun / au^2]
    grid_radius = radial grid [au]
    grid_angle = azimuthal grid [-np.pi,np.pi] [rad]
    p = power law index of the density profile. \Sigma \propto r^(p)
    rin = inner radius of the disc [au]
    rout = outer radius of the disc [au]
    md = mass of the disc [msun]
    m = number of the spiral arms
    alpha = pitch angle of the spiral [rad]
    pos = angle of the spiral within the disc [rad]'''
    
    return   beta**(-1/2) * -np.cos(-(m * 
    grid_angle + m/np.tan(alpha) * -np.log(grid_radius) + pos))



def get_masses(statement, rot_curve, radii, ms, z, starmin, starmax, discmin, discmax, p, n):
    
    '''Very simple algorithm that optimises the mass of the disc+star / disc / star from the rotation curve
    The output is the 1D / 2D array containing the std deviation.
    statement = -1 if you want mass of the disc and mass of the star,
                 0 if you want only the mass of the disc
                 1 if you want only the mass of the star
    rot_curve = vector of the rotation curve [km/s]
    radii = vector of the radii [au]
    ms = mass of the central object [msun], if you want to find it simply put 0
    z = height [au] (for midplane put z=1e-3)
    starmin = minimum value of the mass of the star [msun], if you want to find only md put 0 
    starmax = maximum value of the mass of the star [msun], if you want to find only md put 0 
    discmin = minimum value of the mass of the disc [msun], if you want to find only ms put 0 
    discmax = minimum value of the mass of the disc [msun], if you want to find only ms put 0 
    p = power law index of the density profile. \Sigma \propto r^(p)
    n = number of the point within the interval of research'''
    
    if (statement == -1): #compute both star and disc mass
        a = np.linspace(starmin, starmax, n)
        b = np.linspace(discmin, discmax, n)
        vec = np.zeros([n,n])
        for i in range (n):
            for j in range (n):
                vec[i,j] = np.sum( (rot_curve - basicspeed(radii, z, b[j], p, radii[0], 
                            radii[len(radii)-1], a[i]))**2 )

        
    if (statement == 0): #compute only disc mass
        b = np.linspace(discmin, discmax, n)
        vec = np.zeros(n)
        for i in range(n):
            vec[i] = np.sum( (rot_curve - basicspeed(radii, z, b[j], p, radii[0], radii[len(radii)-1], ms))**2 )
        
        
    if (statement == 1): #compute only star mass
        a = np.linspace(starmin, starmax, n)
        vec = np.zeros(n)
        for i in range(n):
            vec[i] = np.sum( (rot_curve - omega(a[i], radii) * radii )**2)
    

    else :
        print('Error in statement, you must choose between -1, 0 and 1.')
    
    return vec



def amplitude_central_channel(grid_radius, grid_angle, ms, md, p, m, chi, beta, rin, rout, alpha, incl):
    
    '''Amplitude of the central channel of the moment one map (v_obs=v_syst) [rad]
    grid_radius = radial grid [au]
    grid_angle = azimuthal grid [-np.pi,np.pi] [rad]
    ms = mass of the central object [msun]
    md = mass of the disc [msun]
    p = power law index of the density profile. \Sigma \propto r^(p)
    m = number of the spiral arms
    chi = heating factors (=1)
    beta = cooling factor
    rin = inner radius of the disc [au]
    rout = outer radius of the disc [au]
    alpha = pitch angle of the spiral [rad]
    incl = inclination angle [rad]
    '''
    
    m1 = momentone(grid_radius, grid_angle, ms, md, p, m, chi, beta, rin, rout, alpha, incl)
    wigg = np.zeros(grid_radius.shape[0])
    
    for i in range(grid_radius.shape[0]):
        for j in range(grid_angle.shape[0]-1):
            if (m1[i,j] < 0 and m1[i,j+1] > 0):
                wigg[i] = grid_angle[0,:][j]
    
    return np.std(wigg)




def urC(gx, gy, ms, md, p, m, chi, beta, rin, rout, alpha, off):
    
    '''2D radial velocity perturbation [km/s] in polar coordinates
    gx = x grid [au]
    gy = y grid [au] 
    ms = mass of the central object [msun]
    md = mass of the disc [msun]
    p = power law index of the density profile. \Sigma \propto r^(p)
    m = number of the spiral arms
    chi = heating factors (=1)
    beta = cooling factor
    rin = inner radius of the disc [au]
    rout = outer radius of the disc [au]
    alpha = pitch angle of the spiral [rad]'''
    
    
    grid_radius = np.sqrt(gx**2 + gy**2)
    car = np.linspace(-rout,rout,gx.shape[0])
    grid_angle = np.zeros([len(car),len(car)])
    for i in range(len(car)):
        for j in range(len(car)):
            grid_angle[i,j] = math.atan2(car[i], car[j])
    return - ura(ms, md, p, m, chi, beta, rin, rout, grid_radius)  * np.sin(
        m * grid_angle + m/np.tan(alpha) * -np.log(grid_radius) + off)



def uphC(gx, gy, ms, md, p, m, chi, beta, rin, rout, alpha, off):
    
    '''2D azimuthal velocity perturbation [km/s] in polar coordinates
    gx = x grid [au]
    gy = y grid [au] 
    ms = mass of the central object [msun]
    md = mass of the disc [msun]
    p = power law index of the density profile. \Sigma \propto r^(p)
    m = number of the spiral arms
    chi = heating factors (=1)
    beta = cooling factor
    rin = inner radius of the disc [au]
    rout = outer radius of the disc [au]
    alpha = pitch angle of the spiral [rad]'''
    
    # x = np.linspace(np.min(grid_radius),np.max(grid_radius),grid_radius.shape[0]/2)
    #phase = m / np.tan(alpha)  * np.log(x)
    #an = np.linspace(-np.pi,np.pi,grid_radius.shape[1])
    #bs = basicspeed(x, 0.001, md, p, rin, rout, ms)
    #vec = np.zeros([grid_radius.shape[0],grid_radius.shape[1]])
    #vp1 = upha(ms, md, p, m, chi, beta, rin, rout, rin)
    #for i in range(grid_radius.shape[1]):
    #   vec[:,i] = bs[:] - vp1* x**(3/2 + p) /rin * np.sin(m*an[i] + phase[:] + off)
    
    grid_radius = np.sqrt(gx**2 + gy**2)
    phase = m / np.tan(alpha) * -np.log(grid_radius)
    car = np.linspace(-rout,rout,gx.shape[0])
    grid_angle = np.zeros([len(car),len(car)])
    for i in range(len(car)):
        for j in range(len(car)):
            grid_angle[i,j] = math.atan2(car[i], car[j])
    
    radii = np.linspace(np.min(grid_radius), np.max(grid_radius), 1000)
    rc = basicspeed(radii, 1e-3, md, p, rin, rout, ms)
    #vec = - upha(ms, md, p, m, chi, beta, rin, rout, grid_radius) * np.sin(m*grid_angle + phase + off)  + rc[
     #   ((grid_radius - np.min(grid_radius))/(radii[1] - radii[0])).astype(int)] 
    vec = np.sqrt( (G*ms/grid_radius)) * beta**(-0.5) / 2 * (md/ms) * np.sin(m*grid_angle + phase + off) + rc[
        ((grid_radius - np.min(grid_radius))/(radii[1] - radii[0])).astype(int)] 
    return vec



def momentoneC(gx, gy, ms, md, p, m, chi, beta, rin, rout, alpha, incl, off):
    
    '''Moment one map / projected velocity field towards the line of sight [km/s]
    gx = x grid [au]
    gy = y grid [au] 
    ms = mass of the central object [msun]
    md = mass of the disc [msun]
    p = power law index of the density profile. \Sigma \propto r^(p)
    m = number of the spiral arms
    chi = heating factors (=1)
    beta = cooling factor
    rin = inner radius of the disc [au]
    rout = outer radius of the disc [au]
    alpha = pitch angle of the spiral [rad]
    incl = inclination angle [rad]
    NB: The moment one map is given in polar coordinates, the disk is face on and the observer is rotated by an angle incl'''
    
    grid_radius = np.sqrt(gx**2 + gy**2)
    car = np.linspace(-rout,rout,gx.shape[0])
    grid_angle = np.zeros([len(car),len(car)])
    for i in range(len(car)):
        for j in range(len(car)):
            grid_angle[i,j] = math.atan2(car[i], car[j])
    
    M1 = uphC(gx, gy, ms, md, p, m, chi, beta, rin, rout, alpha, off) * np.cos(
        grid_angle) * np.sin(incl) + urC(gx, gy, ms, md, p, m, chi, beta,
        rin, rout, alpha, off) * np.sin(grid_angle) * np.sin(incl)
    for i in range(M1.shape[0]):
        for j in range(M1.shape[0]):
            if(grid_radius[i,j] > rout):
                M1[i,j] = -np.inf
    return M1

def uraB(ms, md, p, m, chi, beta, rin, rout, r, n):
    
    '''Module of the radial velocity perturbation [km/s]
    ms = mass of the central object [msun]
    md = mass of the disc [msun]
    p = power law index of the density profile. \Sigma \propto r^(p)
    m = number of the spiral arms
    chi = heating factors (=1)
    beta = cooling factor
    rin = inner radius of the disc [au]
    rout = outer radius of the disc [au]
    r = radius [au]'''
    
    return 8 * m * chi * beta**(-1/2) * q(ms, md, p, rin, rout, r)**2 * omega(ms,r) * r * (r/rin)**(-n/2)



def uphaB(ms, md, p, m, chi, beta, rin, rout, r, n):
    
    '''Module of the azimuthal velocity perturbation [km/s]
    ms = mass of the central object [msun]
    md = mass of the disc [msun]
    p = power law index of the density profile. \Sigma \propto r^(p)
    m = number of the spiral arms
    chi = heating factors (=1)
    beta = cooling factor
    rin = inner radius of the disc [au]
    rout = outer radius of the disc [au]
    r = radius [au]'''
    
    return - (m * chi * beta**(-1/2))  * q(ms, md, p, rin, rout, r) * omega(ms,r) * r * (r/rin)**(-n/2)

def urCeta(gx, gy, ms, md, p, m, chi, beta, rin, rout, alpha, off, n):
    
    '''2D radial velocity perturbation [km/s] in polar coordinates
    grid_radius = radial grid [au]
    grid_angle = azimuthal grid [-np.pi,np.pi] [rad]
    ms = mass of the central object [msun]
    md = mass of the disc [msun]
    p = power law index of the density profile. \Sigma \propto r^(p)
    m = number of the spiral arms
    chi = heating factors (=1)
    beta = cooling factor
    rin = inner radius of the disc [au]
    rout = outer radius of the disc [au]
    alpha = pitch angle of the spiral [rad]'''
    
    
    grid_radius = np.sqrt(gx**2 + gy**2)
    car = np.linspace(-rout,rout,gx.shape[0])
    grid_angle = np.zeros([len(car),len(car)])
    for i in range(len(car)):
        for j in range(len(car)):
            grid_angle[i,j] = math.atan2(car[i], car[j])
    return - uraB(ms, md, p, m, chi, beta, rin, rout, grid_radius, n)  * np.sin(
        m * grid_angle + m/np.tan(alpha) * np.log(grid_radius) + off)



def uphCeta(gx, gy, ms, md, p, m, chi, beta, rin, rout, alpha, off, n):
    
    '''2D azimuthal velocity perturbation [km/s] in polar coordinates
    grid_radius = radial grid [au]
    grid_angle = azimuthal grid [-np.pi,np.pi] [rad]
    ms = mass of the central object [msun]
    md = mass of the disc [msun]
    p = power law index of the density profile. \Sigma \propto r^(p)
    m = number of the spiral arms
    chi = heating factors (=1)
    beta = cooling factor
    rin = inner radius of the disc [au]
    rout = outer radius of the disc [au]
    alpha = pitch angle of the spiral [rad]'''
    
    # x = np.linspace(np.min(grid_radius),np.max(grid_radius),grid_radius.shape[0]/2)
    #phase = m / np.tan(alpha)  * np.log(x)
    #an = np.linspace(-np.pi,np.pi,grid_radius.shape[1])
    #bs = basicspeed(x, 0.001, md, p, rin, rout, ms)
    #vec = np.zeros([grid_radius.shape[0],grid_radius.shape[1]])
    #vp1 = upha(ms, md, p, m, chi, beta, rin, rout, rin)
    #for i in range(grid_radius.shape[1]):
    #   vec[:,i] = bs[:] - vp1* x**(3/2 + p) /rin * np.sin(m*an[i] + phase[:] + off)
    
    grid_radius = np.sqrt(gx**2 + gy**2)
    car = np.linspace(-rout,rout,gx.shape[0])
    grid_angle = np.zeros([len(car),len(car)])
    for i in range(len(car)):
        for j in range(len(car)):
            grid_angle[i,j] = math.atan2(car[i], car[j])
    
    radii = np.linspace(np.min(grid_radius), np.max(grid_radius), 1000)
    rc = basicspeed(radii, 1e-3, md, p, rin, rout, ms)
    vec = - uphaB(ms, md, p, m, chi, beta, rin, rout, grid_radius, n) + rc[
        ((grid_radius - np.min(grid_radius))/(radii[1] - radii[0])).astype(int)] 
    return vec



def momentoneCeta(gx, gy, ms, md, p, m, chi, beta, rin, rout, alpha, incl, off, n):
    
    '''Moment one map / projected velocity field towards the line of sight [km/s]
    gx = x grid [au]
    gy = y grid [au] 
    ms = mass of the central object [msun]
    md = mass of the disc [msun]
    p = power law index of the density profile. \Sigma \propto r^(p)
    m = number of the spiral arms
    chi = heating factors (=1)
    beta = cooling factor
    rin = inner radius of the disc [au]
    rout = outer radius of the disc [au]
    alpha = pitch angle of the spiral [rad]
    incl = inclination angle [rad]
    NB: The moment one map is given in polar coordinates, the disk is face on and the observer is rotated by an angle incl'''
    
    grid_radius = np.sqrt(gx**2 + gy**2)
    car = np.linspace(-rout,rout,gx.shape[0])
    grid_angle = np.zeros([len(car),len(car)])
    for i in range(len(car)):
        for j in range(len(car)):
            grid_angle[i,j] = math.atan2(car[i], car[j])
    
    M1 = uphCeta(gx, gy, ms, md, p, m, chi, beta, rin, rout, alpha, off, n) * np.cos(
        grid_angle) * np.sin(incl) + urCeta(gx, gy, ms, md, p, m, chi, beta,
        rin, rout, alpha, off, n) * np.sin(grid_angle) * np.sin(incl)
    for i in range(M1.shape[0]):
        for j in range(M1.shape[0]):
            if(grid_radius[i,j] > rout):
                M1[i,j] = -np.inf
    return M1


'''


#Parameters
ms = 1 #star mass
md = 0.35 #disc mass
p = -1.5 #surface density
ap = 13*np.pi/180 #pitch angle
m = 2 #azimuthal wavenumber
beta = 5 #cool
incl = np.pi/2.1 #inclination of the disc towards the line of sight


r = np.linspace(1,100,500)
phi = np.linspace(-np.pi,np.pi,360)
gr, gphi = np.mgrid[1:100:500j, -np.pi:np.pi:360j] #rin:rout:resolution
gx, gy = np.mgrid[-100:100:400j,-100:100:400j]
car = np.linspace(-100,100,400)
grid_angle = 0*gx
g_r = (gx**2+gy**2)**(0.5)
    
m1c = momentoneC(gx, gy, ms, md , p, m, 1, beta, 1, 100, ap, incl, 0)
m1k = momentoneC(gx, gy, ms, 0 , p, m, 1, beta, 1, 100, ap, incl, 0)
for i in range(len(car)):
    for j in range(len(car)):
        grid_angle[i,j] = math.atan2(car[i], car[j])
spir = perturbed_sigma(g_r, grid_angle, p, 1, 100, md, beta, m, ap,0)
#Spiral surface density
for i in range(len(car)):
    for j in range(len(car)):
        if(g_r[i,j] > 100):
            spir[i,j] = -np.inf


#Rotation of the disc
matrix_y_deproject = [[np.cos((incl)), 0],[0, 1]]
matrix_y_deproject = np.asarray(matrix_y_deproject)
matrix_y_deproject = np.linalg.inv(matrix_y_deproject)

m1crot = ndimage.affine_transform(m1c, matrix_y_deproject, 
                    offset=(-30,-0),order=1)
m1krot = ndimage.affine_transform(m1k, matrix_y_deproject, 
                    offset=(-30,-0),order=1)
spir_rot = ndimage.affine_transform(spir, matrix_y_deproject, 
                    offset=(-30,-0),order=1)

'''

'''
#plot1
fig, (ax) = plt.subplots(1,1,figsize=(8,8))
plt.imshow(m1crot, cmap='seismic', vmin = -3, vmax = 3, origin='lower')
plt.text(150+70,222,'20', size=17)
plt.text(150+95,250,'40', size=17)
plt.text(150+120,275,'60', size=17)
plt.text(150+150,302,'80', size=17)
plt.text(150+180,330,'100', size=17)
plt.text(405,195, '$0$', size=17)
plt.text(187.5,380, '$\pi/2$', size=17)
plt.text(-20,195, r'$\pi$', size=17)
plt.text(187.5,-0, r'$3\pi/2$', size=17)
plt.axis('off')
plt.colorbar(label=r'$v_{obs}$ [km/s]')
plt.savefig('p1.png', dpi=300)


#plot2
fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(20,5))
ax1.contour(m1crot, [0], colors='mediumblue')
ax1.plot(200*np.cos(phi) + 200, 172*np.sin(phi)+ 198, c='black', lw=1)
ax1.plot(200/5*np.cos(phi) + 200, 172/5*np.sin(phi)+ 198, c='black', lw=0.25)
ax1.plot(200/5 * 2 *np.cos(phi) + 200, 172/5 * 2*np.sin(phi)+ 198, c='black', lw=0.25)
ax1.plot(200/5 * 2*np.cos(phi) + 200, 172/5 * 2*np.sin(phi)+ 198, c='black', lw=0.25)
ax1.plot(200/5 * 3*np.cos(phi) + 200, 172/5 * 3*np.sin(phi)+ 198, c='black', lw=0.25)
ax1.plot(200/5 * 4*np.cos(phi) + 200, 172/5 * 4*np.sin(phi)+ 198, c='black', lw=0.25)
ax1.text(150,222,'20', size=15)
ax1.text(150-28,250,'40', size=15)
ax1.text(150-28-25,275,'60', size=15)
ax1.text(150-28-25-27,302,'80', size=15)
ax1.text(150-28-25-27-28,330,'100', size=15)
ax1.text(420,195, '$0$', size=17)
ax1.text(175,390, '$\pi/2$', size=17)
ax1.text(-30,195, r'$\pi$', size=17)
ax1.text(175,-10, r'$3\pi/2$', size=17)
ax1.set_xlim(-5,403)
ax1.plot(np.linspace(0,1,10) * 0 + 200, np.linspace(0,1,10)*344 + 26, lw=0.5, c='black')
ax1.plot(np.linspace(0,1,10)*400 + 1, np.linspace(0,1,10) * 0 + 200, lw=0.5, c='black')
ax1.text(100,-60,r'$v_{obs} = 0 $ km/s', size=22)
ax1.axis('off')
ax1.axis('equal')

ax2.contour(m1crot, [1], colors='mediumblue')
ax2.plot(200*np.cos(phi) + 200, 172*np.sin(phi)+ 198, c='black', lw=1)
ax2.plot(200/5*np.cos(phi) + 200, 172/5*np.sin(phi)+ 198, c='black', lw=0.25)
ax2.plot(200/5 * 2 *np.cos(phi) + 200, 172/5 * 2*np.sin(phi)+ 198, c='black', lw=0.25)
ax2.plot(200/5 * 2*np.cos(phi) + 200, 172/5 * 2*np.sin(phi)+ 198, c='black', lw=0.25)
ax2.plot(200/5 * 3*np.cos(phi) + 200, 172/5 * 3*np.sin(phi)+ 198, c='black', lw=0.25)
ax2.plot(200/5 * 4*np.cos(phi) + 200, 172/5 * 4*np.sin(phi)+ 198, c='black', lw=0.25)
ax2.text(150,222,'20', size=15)
ax2.text(150-28,250,'40', size=15)
ax2.text(150-28-25,275,'60', size=15)
ax2.text(150-28-25-27,302,'80', size=15)
ax2.text(150-28-25-27-28,330,'100', size=15)
ax2.text(420,195, '$0$', size=17)
ax2.text(175,390, '$\pi/2$', size=17)
ax2.text(-30,195, r'$\pi$', size=17)
ax2.text(175,-10, r'$3\pi/2$', size=17)
ax2.set_xlim(-5,403)
ax2.plot(np.linspace(0,1,10) * 0 + 200, np.linspace(0,1,10)*344 + 26, lw=0.5, c='black')
ax2.plot(np.linspace(0,1,10)*400 + 1, np.linspace(0,1,10) * 0 + 200, lw=0.5, c='black')
ax2.text(100,-60,r'$v_{obs} = 1 $ km/s', size=22)
ax2.axis('off')
ax2.axis('equal')

ax3.contour(m1crot, [2], colors='mediumblue')
ax3.plot(200*np.cos(phi) + 200, 172*np.sin(phi)+ 198, c='black', lw=1)
ax3.plot(200/5*np.cos(phi) + 200, 172/5*np.sin(phi)+ 198, c='black', lw=0.25)
ax3.plot(200/5 * 2 *np.cos(phi) + 200, 172/5 * 2*np.sin(phi)+ 198, c='black', lw=0.25)
ax3.plot(200/5 * 2*np.cos(phi) + 200, 172/5 * 2*np.sin(phi)+ 198, c='black', lw=0.25)
ax3.plot(200/5 * 3*np.cos(phi) + 200, 172/5 * 3*np.sin(phi)+ 198, c='black', lw=0.25)
ax3.plot(200/5 * 4*np.cos(phi) + 200, 172/5 * 4*np.sin(phi)+ 198, c='black', lw=0.25)
ax3.text(150,222,'20', size=15)
ax3.text(150-28,250,'40', size=15)
ax3.text(150-28-25,275,'60', size=15)
ax3.text(150-28-25-27,302,'80', size=15)
ax3.text(150-28-25-27-28,330,'100', size=15)
ax3.text(420,195, '$0$', size=17)
ax3.text(175,390, '$\pi/2$', size=17)
ax3.text(-30,195, r'$\pi$', size=17)
ax3.text(175,-10, r'$3\pi/2$', size=17)
ax3.set_xlim(-5,403)
ax3.plot(np.linspace(0,1,10) * 0 + 200, np.linspace(0,1,10)*344 + 26, lw=0.5, c='black')
ax3.plot(np.linspace(0,1,10)*400 + 1, np.linspace(0,1,10) * 0 + 200, lw=0.5, c='black')
ax3.text(100,-60,r'$v_{obs} = 2 $ km/s', size=22)
ax3.axis('off')
ax3.axis('equal')

ax4.contour(m1crot, [3], colors='mediumblue')
ax4.plot(200*np.cos(phi) + 200, 172*np.sin(phi)+ 198, c='black', lw=1)
ax4.plot(200/5*np.cos(phi) + 200, 172/5*np.sin(phi)+ 198, c='black', lw=0.25)
ax4.plot(200/5 * 2 *np.cos(phi) + 200, 172/5 * 2*np.sin(phi)+ 198, c='black', lw=0.25)
ax4.plot(200/5 * 2*np.cos(phi) + 200, 172/5 * 2*np.sin(phi)+ 198, c='black', lw=0.25)
ax4.plot(200/5 * 3*np.cos(phi) + 200, 172/5 * 3*np.sin(phi)+ 198, c='black', lw=0.25)
ax4.plot(200/5 * 4*np.cos(phi) + 200, 172/5 * 4*np.sin(phi)+ 198, c='black', lw=0.25)
ax4.text(150,222,'20', size=15)
ax4.text(150-28,250,'40', size=15)
ax4.text(150-28-25,275,'60', size=15)
ax4.text(150-28-25-27,302,'80', size=15)
ax4.text(150-28-25-27-28,330,'100', size=15)
ax4.text(420,195, '$0$', size=17)
ax4.text(175,390, '$\pi/2$', size=17)
ax4.text(-30,195, r'$\pi$', size=17)
ax4.text(175,-10, r'$3\pi/2$', size=17)
ax4.set_xlim(-5,403)
ax4.plot(np.linspace(0,1,10) * 0 + 200, np.linspace(0,1,10)*344 + 26, lw=0.5, c='black')
ax4.plot(np.linspace(0,1,10)*400 + 1, np.linspace(0,1,10) * 0 + 200, lw=0.5, c='black')
ax4.text(100,-60,r'$v_{obs} = 3 $ km/s', size=22)
ax4.axis('off')
ax4.axis('equal')
plt.savefig('p2.png', dpi=300)


#plot3
fig, (ax) = plt.subplots(1,1,figsize=(8,8))

plt.imshow(np.log(spir_rot), cmap='jet', vmin = -12.5, vmax = -8., origin='lower')
plt.text(150+70,222,'20', size=17)
plt.text(150+95,250,'40', size=17)
plt.text(150+120,275,'60', size=17)
plt.text(150+150,302,'80', size=17)
plt.text(150+180,330,'100', size=17)
plt.text(405,195, '$0$', size=17)
plt.text(187.5,380, '$\pi/2$', size=17)
plt.text(-20,195, r'$\pi$', size=17)
plt.text(187.5,-0, r'$3\pi/2$', size=17)
plt.axis('off')
plt.colorbar(label=r'$v_{obs}$ [km/s]')
plt.savefig('p3.png', dpi=300)



#plot4
fig, (ax) = plt.subplots(1,1,figsize=(8,8))

plt.imshow(-m1crot +m1krot, cmap='seismic', vmin = -0.8, vmax = 0.8, origin='lower')
plt.colorbar(label=r'$v_{obs}$ [km/s]')

ax.text(150+70,222,'20', size=17)
ax.text(150+95,250,'40', size=17)
ax.text(150+120,275,'60', size=17)
ax.text(150+150,302,'80', size=17)
ax.text(150+180,330,'100', size=17)

ax.text(405,195, '$0$', size=17)
ax.text(187.5,390, '$\pi/2$', size=17)
ax.text(-20,195, r'$\pi$', size=17)
ax.text(187.5,-10, r'$3\pi/2$', size=17)
ax.set_xlim(-5,403)
ax.axis('off')
plt.savefig('p4.png', dpi = 300)
'''

'''my additions start'''
# I just want to get a sense of what the grid looks like and how I can use it

'''
test_ur = ur(grid_radius, grid_angle, ms, md, p, m, chi, beta,
        rin, rout, alpha, off)
        '''


'''my additions end'''