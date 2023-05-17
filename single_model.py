
from astropy.io import fits
import os
from disk import *
from raytrace import *
from scipy.optimize import curve_fit
import scipy.interpolate
from scipy.integrate import cumtrapz,trapz
from galario import double as gdouble
import time
import uuid


##############################################################################
def make_model_vis(datfile='data/HD163296.CO32.regridded.cen15',modfile='testpy_alma',isgas=True,freq0=345.79599):
    '''Create model visibilities from a model image, using Miriad. Deprecated in favor of Galario.'''
    if isgas:
        cmd = ' ./sample_alma.csh '+modfile+' '+datfile+' '+str(freq0)
    else:
        cmd = ' ./sample_cont.csh '+modfile+' '+datfile+' '+str(freq0)
    os.system(cmd)

def compare_vis(datfile='data/HD163296.CO32.regridded.cen15',modfile='model/testpy_alma',new_weight=[1,],systematic=False,isgas=True,plot_resid=False):
    '''Calculate the raw chi-squared based on the difference between the model and data visibilities.
    Deprecated in favor of compare_vis_galario, which uses galario to calculate the model visibilities, instead of Miriad, which is used here.

    :param datfile: (default = 'data/HD163296.CO32.regridded.cen15')
     The base name for the data file. The code reads in the visibilities from datfile+'.vis.fits'

     :param modfile" (default='model/testpy_alma')
     The base name for the model file. The code reads in the visibilities from modfile+'.model.vis.fits'

     :param new_weight:
     An array containing the weights to be used in the chi-squared calculation. This should have the same dimensions as the real and imaginary part of the visibilities (ie Nbas x Nchan)

     :param systematic:
     The systematic weight to be applied. The value sent with this keyword is used to scale the absolute flux level of the model. It is defined such that a value >1 decreases the model and a value <1 increases the model (the model visibilities are divided by the systematic parameter). It is meant to mimic a true flux in the data which is larger or smaller by the fraction systematic (e.g. specifying systematic=1.2 is equivalent to saying that the true flux of the data is 20% brighter than what has been observed, with this scaling applied to the model instead of changing the data)

     :param isgas:
     If the data is line emission then the data has an extra dimension covering the >1 channels. Set this keyword to ensure that the data is read in properly. Conversely, if you are comparing continuum data then set this keyword to False.

'''

    # - Read in object visibilities
    obj = fits.open(datfile+'.vis.fits')
    freq0 = obj[0].header['crval4']
    klam = freq0/1e3
    u_obj,v_obj = obj[0].data['UU']*klam,obj[0].data['VV']*klam
    # for now assume it is alma data
    vis_obj = (obj[0].data['data']).squeeze()
    if isgas:
        if obj[0].header['telescop'] == 'ALMA':
            if obj[0].header['naxis3'] == 2:
                real_obj = (vis_obj[:,:,0,0]+vis_obj[:,:,1,0])/2. #format=Nbase*Nchan*3*2
                imag_obj = (vis_obj[:,:,0,1]+vis_obj[:,:,1,1])/2.
                weight_real = vis_obj[:,:,0,2]
                weight_imag = vis_obj[:,:,1,2]
                #weight_alma = (vis_obj[:,:,0,2]+vis_obj[:,:,1,2])/2.

        # - Read in model visibilities
                model = fits.open(modfile+'.model.vis.fits')
                vis_mod = model[0].data['data']
                real_model = vis_mod[::2,0,0,:,0,0]
                imag_model = vis_mod[::2,0,0,:,0,1]
            else:
                real_obj = (vis_obj[::2,:,0]) #format=Nbase*Nchan*3*2
                imag_obj = (vis_obj[::2,:,1])

        # - Read in model visibilities
                model = fits.open(modfile+'.model.vis.fits')
                vis_mod = model[0].data['data']
                real_model = vis_mod[::2,0,0,:,0,0]
                imag_model = vis_mod[::2,0,0,:,0,1]
    else:

        if obj[0].header['telescop'] == 'ALMA':
            if obj[0].header['naxis3'] == 2:
                real_obj = (vis_obj[:,0,0]+vis_obj[:,1,0])/2.
                imag_obj = (vis_obj[:,0,1]+vis_obj[:,1,1])/2.
                weight_real = vis_obj[:,0,2]
                weight_imag = vis_obj[:,1,2]
                #weight_alma = (vis_obj[:,0,2]+vis_obj[:,1,2])/2.

                model = fits.open(modfile+'.model.vis.fits')
                vis_mod = model[0].data['data'].squeeze()
                real_model = vis_mod[::2,0]
                imag_model = vis_mod[::2,1]
    obj.close()
    model.close()


    # if model has nans then just return chi=inf
    if (np.isnan(real_model)).sum() >0:
        return np.inf

    # - Add systematic uncertainty
    if systematic:
        real_model = real_model/systematic
        imag_model = imag_model/systematic

    if len(new_weight) > 1:
        #weight_alma = new_weight
        weight_real = new_weight
        weight_imag = new_weight

    #wremove = (real_obj == 0.) & (imag_obj == 0.) #| (weight_alma<.05)
    #weight_alma[wremove] = 0.
    weight_real[real_obj==0] = 0.
    weight_imag[imag_obj==0] = 0.
    print('Removed data %i' % ((weight_real ==0).sum()+(weight_imag==0).sum()))
#    print('Retained data %i' % ((weight_alma !=0).sum()))

    if plot_resid:
        #Code to plot, and fit, residuals
        #If errors are Gaussian, then residuals should have gaussian shape
        #If error size is correct, residuals will have std=1
        uv = np.sqrt(u_obj**2+v_obj**2)
        use = (weight_real > .05) & (weight_imag>.05)
        diff = np.concatenate((((real_model[use]-real_obj[use])*np.sqrt(weight_real[use])),((imag_model[use]-imag_obj[use])*np.sqrt(weight_imag[use]))))
        diff = diff.flatten()
        n,bins,patches = plt.hist(diff,10000,normed=1,histtype='step',color='k',label='Data',lw=3)
        popt,pcov = curve_fit(gaussian,bins[1:],n)
        y=gaussian(bins,popt[0],popt[1],popt[2])
        print('Gaussian fit parameters (amp,width,center): ',popt)
        print('If errors are properly scaled, then width should be close to 1')
        plt.plot(bins,y,'r--',lw=6,label='gaussuian')
        #slight deviations from gaussian, but gaussian is still the best...
        plt.xlabel('(Model-Data)/$\sigma$',fontweight='bold',fontsize=20)
        ax=plt.gca()
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(20)
            tick.label1.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(20)
            tick.label1.set_fontweight('bold')

        plt.show()

    # - Calculate Chi-squared
    uv = np.sqrt(u_obj**2+v_obj**2)
    chi = ((real_model-real_obj)**2*weight_real).sum() + ((imag_model-imag_obj)**2*weight_imag).sum()
    return chi

def gaussian(x,amp,width,center):
    return amp/(width*np.sqrt(2*np.pi))*np.exp(-.5*((x-center)**2)/width**2)

#def lorentzian(x,amp,width,center):
#    return 1/np.pi*(width/((x-center)**2.+width**2))*amp

#def laplace(x,amp,width,center):
#    return amp/(2*width)*np.exp(-np.abs(x-center)/width)

def compare_vis_galario(datfile='data/HD163296.CO32.regridded.cen15',modfile='model/testpy_alma',new_weight=[1,],systematic=False,isgas=True,plot_resid=False):
    '''Calculate the raw chi-squared based on the difference between the model and data visibilities.

    :param datfile: (default = 'data/HD163296.CO32.regridded.cen15')
     The base name for the data file. The code reads in the visibilities from datfile+'.vis.fits'

     :param modfile: (default='model/testpy_alma')
     The base name for the model file. The code reads in the visibilities from modfile+'.model.vis.fits'

     :param new_weight:
     An array containing the weights to be used in the chi-squared calculation. This should have the same dimensions as the real and imaginary part of the visibilities (ie Nbas x Nchan)

     :param systematic:
     The systematic weight to be applied. The value sent with this keyword is used to scale the absolute flux level of the model. It is defined such that a value >1 decreases the model and a value <1 increases the model (the model visibilities are divided by the systematic parameter). It is meant to mimic a true flux in the data which is larger or smaller by the fraction systematic (e.g. specifying systematic=1.2 is equivalent to saying that the true flux of the data is 20% brighter than what has been observed, with this scaling applied to the model instead of changing the data)

     :param isgas:
     If the data is line emission then the data has an extra dimension covering the >1 channels. Set this keyword to ensure that the data is read in properly. Conversely, if you are comparing continuum data then set this keyword to False.

'''
    #Limit the multi-threading of Galario (necessary on the computing cluster)
    gdouble.threads(1)

    # - Read in object visibilities
    obj = fits.open(datfile+'.vis.fits')
    freq0 = obj[0].header['crval4']
    u_obj,v_obj = (obj[0].data['UU']*freq0).astype(np.float64),(obj[0].data['VV']*freq0).astype(np.float64)
    vis_obj = (obj[0].data['data']).squeeze()
    if isgas:
        if obj[0].header['telescop'] == 'ALMA':
            if obj[0].header['naxis3'] == 2:
                real_obj = (vis_obj[:,:,0,0]+vis_obj[:,:,1,0])/2.
                imag_obj = (vis_obj[:,:,0,1]+vis_obj[:,:,1,1])/2.
                weight_real = vis_obj[:,:,0,2]
                weight_imag = vis_obj[:,:,1,2]
            else:
                real_obj = vis_obj[::2,:,0]
                imag_obj = vis_obj[::2,:,1]
    else:
        if obj[0].header['telescop'] == 'ALMA':
            if obj[0].header['naxis3'] == 2:
                real_obj = (vis_obj[:,0,0]+vis_obj[:,1,0])/2.
                imag_obj = (vis_obj[:,0,1]+vis_obj[:,1,1])/2.
                weight_real = vis_obj[:,0,2]
                weight_imag = vis_obj[:,1,2]

    obj.close()

    #Generate model visibilities
    model_fits = fits.open(modfile+'.fits')
    model = model_fits[0].data.squeeze()
    nxy,dxy = model_fits[0].header['naxis1'],np.radians(np.abs(model_fits[0].header['cdelt1']))
    model_fits.close()
    if isgas:
        real_model = np.zeros(real_obj.shape)
        imag_model = np.zeros(imag_obj.shape)
        for i in range(real_obj.shape[1]):
            vis = gdouble.sampleImage(np.flipud(model[i,:,:]).byteswap().newbyteorder(),dxy,u_obj,v_obj)
            real_model[:,i] = vis.real
            imag_model[:,i] = vis.imag
    else:
        vis = gdouble.sampleImage(model.byteswap().newbyteorder(),dxy,u_obj,v_obj)
        real_model = vis.real
        imag_model = vis.imag

    if systematic:
        real_model = real_model/systematic
        imag_model = imag_model/systematic

    if len(new_weight) > 1:
        weight_real = new_weight
        weight_imag = new_weight

    weight_real[real_obj==0] = 0.
    weight_imag[imag_obj==0] = 0.
    print('Removed data %i' % ((weight_real ==0).sum()+(weight_imag==0).sum()))

    if plot_resid:
        #Code to plot, and fit, residuals
        #If errors are Gaussian, then residuals should have gaussian shape
        #If error size is correct, residuals will have std=1
        uv = np.sqrt(u_obj**2+v_obj**2)
        use = (weight_real > .05) & (weight_imag>.05)
        diff = np.concatenate((((real_model[use]-real_obj[use])*np.sqrt(weight_real[use])),((imag_model[use]-imag_obj[use])*np.sqrt(weight_imag[use]))))
        diff = diff.flatten()
        n,bins,patches = plt.hist(diff,10000,normed=1,histtype='step',color='k',label='Data',lw=3)
        popt,pcov = curve_fit(gaussian,bins[1:],n)
        y=gaussian(bins,popt[0],popt[1],popt[2])
        print('Gaussian fit parameters (amp,width,center): ',popt)
        print('If errors are properly scaled, then width should be close to 1')
        plt.plot(bins,y,'r--',lw=6,label='gaussuian')
        #slight deviations from gaussian, but gaussian is still the best...
        plt.xlabel('(Model-Data)/$\sigma$',fontweight='bold',fontsize=20)
        ax=plt.gca()
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(20)
            tick.label1.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(20)
            tick.label1.set_fontweight('bold')

        plt.show()



    chi = ((real_model-real_obj)**2*weight_real).sum() + ((imag_model-imag_obj)**2*weight_imag).sum()
    return chi

def lnlike(p,highres=False,massprior=False,cleanup=False,systematic=False,line='co21',vcs=True,exp_temp=False,add_ring=False,use_galario=True):
    '''Calculate the log-likelihood (=-0.5*chi-squared) for a given model.

    p (list): This is the list of variable parameters for the model. While there are many parameters for the model, this list includes the parameters that you want to adjust from one model to the next. For example, the disk gas mass is needed to generate the models, but if you don't want to adjust it, then it won't be included in p. The exact parameters included in p can be changed. By default p includes [q, log(Rc), vturb, Tatm0, Tmid0, inclination, Rabund_in, vsys, x offset, y offset, position angle]. See the all_params dictionary for a full list of model parameters, and to see which are included in p.

    highres (default = False): Use a high-resolution version of the data. This flag is useful if you want to easily access different versions of the data. The code can be modified so that different versions of the data are used when this flag is on or off.

    massprior (default = False): In general, uniform priors are assumed, but a non-uniform prior can be used on any number of variables. The code includes an example of applying a Gaussian prior on the disk mass, but this can be adjusted as needed.

    cleanup (default = False):  If set to True then the model image will be given a unique name, and the model image will be deleted after the log-likelihood has been calculated. This is useful if you are running many log-likelihood calculations (e.g., with an MCMC) and you don't want to save each model image, and you don't want multiple processors that are running simultaneously to create model images with the same name.

    systematic (default = False): If set to True, the code will account for systematic uncertainty in the flux. It will do so by multiplying the model image by a constant scale factor, specified by the last value in the list of parameters, p.

    line (string, default = 'co21'): The line being studied. This can be used to use different data sets (in the same way as highres). It is also used in setting up the disk structure, by setting the line, and associated line broadening, that is used.

    vcs (default = True): By default, the numerical value of the turbulence is assumed to refer to the turbulence relative to the thermal line broadening. Setting this parameter to False will instead interpret the numerical value of the turbulence as the velocity in units of km/sec.

    exp_temp (default = False): By default, the disk modeling code uses the Dartois et al. Type II profile for the shape of the vertical temperature profile. Setting this parameter to True will use an exponential profile instead.

    add_ring (default = False): DON'T USE. Has not been updated to work properly.

    use_galario (default = True): Use Galario (Tazzari et al.) to calculate the visibilities from the model image, rather than using Miriad. 

    '''

    start=time.time()
    all_params = {'q':p[0], #q
    'Mdisk':0.09, #solar masses
    'p':1, #gamma
    'Rin':1., #Model inner domain - generally no need to change
    'Rout':1000., #Model outer domain - generally no need to change
    'Rc':10**(p[1]), #Rc
    'incl':p[5], #inclination, degrees
    'Mstar':2.3, #solar masses
    'Xco':10**(-4.), #Abundance
    'vturb':p[2], #turbulence, as a fraction of the thermal broadening for this line
    'Zq0':70., #Zq0
    'Tmid0':p[4], #K
    'Tatm0':p[3],#K
    'Zabund':[.79,1000], #upper and lower boundaries in column density
    'Rabund':[p[6],800.], #inner and outer boundaries for abundance
    'handed':-1, #handed
    'vsys':p[7], #systemic velocity, km/s
    'offs':[p[8],p[9]], #position offset, arcseconds
    'PA':p[10], #position angle, degrees
    'distance':101.} #distance
    if line.lower() =='co21' or line.lower()=='co32' or line.lower()=='svco21':
        params = [all_params['q'],all_params['Mdisk'],all_params['p'],all_params['Rin'],all_params['Rout'],all_params['Rc'],all_params['incl'],all_params['Mstar'],all_params['Xco'],all_params['vturb'],all_params['Zq0'],all_params['Tmid0'],all_params['Tatm0'],all_params['Zabund'],all_params['Rabund'],all_params['handed']]
    if all_params['Mdisk'] <0 or all_params['Mdisk']>all_params['Mstar'] or all_params['Rin']<0 or all_params['Rin']>all_params['Rout'] or all_params['Rout']<0 or all_params['Rc']<0 or all_params['Mstar']<0 or all_params['vturb']<0 or all_params['Zq0']<0 or all_params['Tmid0']<0 or all_params['Tmid0']>all_params['Tatm0'] or all_params['Tatm0']<0 or all_params['Zabund'][0]<0 or all_params['Zabund'][1]<0 or all_params['Zabund'][1]<all_params['Zabund'][0] or all_params['Rabund'][0]<0 or all_params['Rabund'][0]<all_params['Rin'] or all_params['Rabund'][0]>all_params['Rabund'][1] or all_params['Rabund'][1]<0 or all_params['Rabund'][1]>all_params['Rout']:
        chi = np.inf
        nu = 1
    else:
        if add_ring:
            if systematic:
                if p[-3]<0:
                    print('Bad ring parameters ',p[-3],p[-2])
                    return -np.inf
                else:
                    disk_structure=Disk(params,rtg=False,exp_temp=exp_temp,ring=[(params[3]+p[-3])/2.,p[-3]-params[3],p[-2]])
            else:
                if p[-2]<0:
                    print('Bad Ring parameters',p[-2:])
                    return -np.inf
                else:
                #disk_structure = Disk(params,rtg=False,exp_temp=exp_temp,ring=[p[-3],p[-2],p[-1]])
                    disk_structure=Disk(params,rtg=False,exp_temp=exp_temp,ring=[(params[3]+p[-2])/2.,p[-2]-params[3],p[-1]])
        else:
            disk_structure=Disk(params,rtg=False,exp_temp=exp_temp)
        if cleanup:
            tf = tempfile.NamedTemporaryFile()
            modfile = tf.name[-9:]
            tf.close()
        else:
            modfile = 'alma'

        # The next series of lines are very specific to the CO3-2 ALMA data for HD 163296, and would need to be modified to use another data set. Basically you need to set the keyword datfile, read in the weights, set the degrees of freedom, set the chanmin,nchans,chanstep keywords (specific to this spectra you are trying to simulate) as well as the image offset.
        if line.lower() == 'co21':
            datfile = 'CO_highres_cen'
            hdr=fits.getheader(datfile+'.vis.fits')
            nu = 2*hdr['naxis4']*hdr['gcount']-len(p)-39320 #227478
            vsys=all_params['vsys']#5.76 from grid_search
            obsv,chanstep,nchans,chanmin = calc_chans(hdr,vsys) #use if setting flipme=True
            offs = all_params['offs']
            resolution = 0.05
            obs = [150,101,300,170] #150,101,280,170 rt grid nr,nphi,nz,zmax

            disk_structure.set_obs(obs)
            disk_structure.set_rt_grid(vcs=vcs)
            disk_structure.set_line(line)
            disk_structure.add_mol_ring(all_params['Rabund'][0],all_params['Rabund'][1],.79,3.,all_params['Xco'],just_frozen=True)
            total_model(disk=disk_structure,chanmin=chanmin,nchans=nchans,chanstep=chanstep,offs=offs,modfile=modfile,imres=resolution,obsv=obsv,vsys=vsys,freq0=230.538,Jnum=1,distance=all_params['distance'],hanning=True,PA=all_params['PA'],bin=1)
            if not use_galario:
                make_model_vis(datfile=datfile,modfile=modfile,isgas=True,freq0=230.538)
        if line.lower() == 'co32':
            datfile = 'HD163296.CO32.new'
            hdr=fits.getheader(datfile+'.vis.fits')
            nu = 2*hdr['naxis4']*hdr['gcount']-len(p)-39320 #227478
            vsys=all_params['vsys']#5.76 from grid_search
            obsv,chanstep,nchans,chanmin = calc_chans(hdr,vsys)
            offs = all_params['offs']
            resolution = 0.05
            obs = [150,101,300,170] #150,101,280,170 rt grid nr,nphi,nz,zmax

            disk_structure.set_obs(obs)
            disk_structure.set_rt_grid(vcs = vcs)
            disk_structure.set_line(line)
            disk_structure.add_mol_ring(all_params['Rabund'][0],all_params['Rabund'][1],.79,3.,all_params['Xco'],just_frozen=True)
            total_model(disk=disk_structure,chanmin=chanmin,nchans=nchans,chanstep=chanstep,offs=offs,modfile=modfile,freq0=345.79599,Jnum=2,imres=resolution,vsys=vsys,obsv=obsv,distance=all_params['distance'],hanning=True,PA=all_params['PA'],bin=1)
            if not use_galario:
                make_model_vis(datfile=datfile,modfile=modfile,isgas=True,freq0=345.79599)


        if systematic:
            sys = p[-1]#1+0.2*np.random.randn()
        else:
            sys = None

        if use_galario:
            chi = compare_vis_galario(datfile=datfile,modfile=modfile,systematic=sys)
        else:
            chi = compare_vis(datfile=datfile,modfile=modfile,systematic=sys)

        if cleanup:
            # Clean up files
            files = [modfile+'p.fits',modfile+'p.han.im',modfile+'p.im',modfile+'p.model.vis',modfile+'p.model.vis.fits']
            for file in files:
                os.system('rm -r '+file)

    if np.isnan(chi):
        chi = 100*nu

    print(p)
    print(chi/nu)


    if massprior:
        # Include a prior on mass with the likelihood estimate
        # Assume a gaussian prior with given mean and
        mean_mdisk = 0.09 #mean Mdisk
        sig_mdisk = 0.01 # standard deviation on prior
        lnp = -np.log(sig_mdisk*np.sqrt(2*np.pi))-(p[1]-mean_mdisk)**2/(2*sig_mdisk**2)
    else:
        lnp = 0.0

    #if systematic:
        #if np.abs(p[-1]-1)>.2:
        #    lnp = -np.inf
        #else:
        #    lnp=0.
        #lnp -= (p[-1]-1.)**2/(2*.2**2) #prior on the gain, centered at 1 with dispersion of 0.2

    print('%r minutes' % ((time.time()-start)/60.))
    return -0.5*chi+lnp
    #return chi/nu

def calc_chans(hdr,vsys):
    '''Given the header for a set of data and a systemic velocity, calculate the number of channels, the minimum channel, and the channel spacing, that are needed when using flipme=True. Returns the velocities within the data, chanstep, nchans, chanmin.

    obsv,chanstep,nchans,chanmin = calc_chans(hdr,5.79)
    total_model(disk=disk_structure,chanmin=chanmin,nchans=nchans,chanstep=chanstep,obsv=obsv)

    Setting flipme=True only works when the central channel corresponds to the systemic velocity. Since this is not always true in the data, the code will calculate the spectrum on a grid centered at the systemic velocity, and then interpolate onto the velocity grid of the data. This requires setting nchans and chanmin so that the modeled spectrum covers the full range of the data. This function calculates the values of nchans and chanmin that are needed to make this happen.'''

    freq = (np.arange(hdr['naxis4'])+1-hdr['crpix4'])*hdr['cdelt4']+hdr['crval4']
    obsv = (hdr['restfreq']-freq)/hdr['restfreq']*2.99e5
    chanstep = np.abs(obsv[1]-obsv[0])
    nchans = 2*np.ceil(np.abs(obsv-vsys).max()/chanstep)+1
    chanmin = -(nchans/2.-.5)*chanstep
    return obsv,chanstep,int(nchans),chanmin


def lnlike_dust(p,systematic=False,cleanup=False,add_ring=False,band=6,add_point=False):
    '''Calculate the log-likelihood (=-0.5*chi-squared) for a given model using the dust continuum data'''

    import tempfile
    import time
    start = time.time()#-.216
    params=[-.216,.09,1.,1.,700.,194.,48.3,2.3,10**(-4.),.04*3.438,70.,17.5,93.8,[.79,1000],[1,1000],-1]
    dust_params = [p[0], # base level dust-to-gas ratio
                   p[1], # Inner radius of disk
                   p[2], # Outer radius of disk
                   p[3], # dust-to-gas ratio of 1st ring
                   p[4], # location of first ring
                   p[5], # dust-to-gas ratio of second ring
                   p[6], # location of second ring
                   p[7], # Outer edge of inner ring
                   p[8]] # power law slope of inner ring
    if dust_params[0]<0 or dust_params[1]<0 or dust_params[2]<0 or dust_params[1]>dust_params[2] or dust_params[3]<0 or dust_params[4]<0  or dust_params[5]<0 or dust_params[6]<0 or dust_params[6]<dust_params[4] or dust_params[6]>dust_params[2] or dust_params[7]<0:
        chi = np.inf
        nu=1
    else:
        if add_point:
            if p[-1]<0:
                return -np.inf
            else:
                point_params=[p[-3],p[-2],p[-1]]
        else:
            point_params=[]
        if add_ring:
            if  p[7]<0:
                print('Bad ring parameters')
                return -np.inf
            else:
                disk_structure = Disk(params,ring=[(p[1]+p[7])/2.,p[7]-p[1],p[8]],obs=[500,131,400,120])#,obs=[500,101,500,150])
        else:
            disk_structure = Disk(params,obs=[250,101,350,150])
        disk_structure.add_dust_ring(dust_params[1],dust_params[2],dust_params[0],0.,initialize=True)
        disk_structure.add_dust_ring(dust_params[4]-10,dust_params[4]+10,dust_params[3],0.)
        disk_structure.add_dust_ring(dust_params[6]-10,dust_params[6]+10,dust_params[5],0.)
        #disk_structure.add_dust_ring(dust_params[1],dust_params[8],dust_params[7],dust_params[9])
        if cleanup:
            tf = tempfile.NamedTemporaryFile()
            modfile = tf.name[-9:]
        else:
            modfile='alma'


        nchans = 1
        resolution = 0.05

        #for datfile in datfiles:
        if band == 6:
            datfile='continuum'
            offs = [-.03,.02]
            npix=512
            #disk_structure.kap=2.3
            disk_structure.kap = 2.3*(disk_structure.r/(150.*disk_structure.AU))**(p[10])
        elif band==7:
            datfile = 'continuum_band7_bin'
            offs = [.075,.075] # from grid search
            disk_structure.kap = p[9]*(disk_structure.r/(150.*disk_structure.AU))**p[10]#4.32,.1605 #3.778,.056
            npix=1024
        hdr = fits.getheader(datfile+'.vis.fits')
        freq = hdr['crval4']/1e9
        nu = 2*hdr['gcount']-len(p)-0
        new_weight = (fits.open(datfile+'_weights.fits'))[0].data
        total_model(disk=disk_structure,nchans=nchans,offs=offs,datfile=datfile,modfile=modfile,imres=resolution,freq0=freq,isgas=False,xnpix=npix,add_point=point_params)



        if systematic:
            sys = p[-1]
        else:
            sys = None

        chi = compare_vis(datfile=datfile,modfile=modfile,systematic=sys,isgas=False,new_weight=new_weight)

        if cleanup:
            files = [modfile+'p.fits',modfile+'p.han.im',modfile+'p.im',modfile+'p.model.vis',modfile+'p.model.vis.fits']
            for file in files:
                os.system('rm -r '+file)


    if np.isnan(chi):
        chi = 100*nu


    print(p)
    print(chi/nu)

    if systematic:
        lnp = -(p[-1]-1.)**2/(2*.2**2) #prior on gain, centered at 1 with dispersion of 0.2
    else:
        lnp=0

    print('{:0.3f} minutes'.format((time.time()-start)/60.))
    return -0.5*chi+lnp


def grid_search():
    'Perform a grid search for the minimum of the chi-squared'
    import time
    start = time.clock()
    chi = []

#    nsteps=20
#    xoff_tot=[]
#    yoff_tot=[]
#    for ix in range(nsteps):
#        xoff = -.15+(.15+.15)/nsteps*ix
#        for iy in range(nsteps):
#            yoff = -.15+(.15+.15)/nsteps*iy
#            #p=[.04*3.438,1.1,47,[xoff,yoff]]
#            #p=[-.27,224.,0.31*3.438,79.,17.5,46.1,[xoff,yoff]]
#            p=[.0128,.0328,14.0,197.,103.,1.09,30.,xoff,yoff]
#            chi.append(-2*lnlike_dust(p,cleanup=True,band=7))
#            xoff_tot.append(xoff)
#            yoff_tot.append(yoff)

    nsteps=5
    xoff_tot=[]
    yoff_tot=[]
    for ix in range(nsteps):
        kappa = 3.75+(3.82-3.75)/nsteps*ix
        for iy in range(nsteps):
            gamma = 0.05+(.065-.05)/nsteps*iy
            p=[.009,10.,218.2,.018,80.3,.031,119.6,76.2,-.68,kappa,gamma]
            chi.append(-2*lnlike_dust(p,add_ring=True,cleanup=True,band=7))
            xoff_tot.append(kappa)
            yoff_tot.append(gamma)



    #nsteps_i = 20
    #vsys_tot = []
    #for i in range(nsteps_i):
    #    vsys = 5.6+(5.8-5.6)/nsteps_i*i
    #    #p=[-.299,2.325,.179,75.7,19.1,47.3,10.5,vsys]
    #    p=[-.177,.145,1.087,48.1,22.3,-4.53,vsys]
    #    #p=[.066,48.8,69.3,149.8,257.2,-11.11,-10.93,-11.08,vsys]
    #    chi.append(-2*lnlike(p,line='c18o21',cleanup=True,highres=False))
    #    vsys_tot.append(vsys)

    nu = 21675452#2*15*131078-6-384735.
    chi = np.array(chi)
    xoff_tot = np.array(xoff_tot)
    yoff_tot = np.array(yoff_tot)
#    vsys_tot = np.array(vsys_tot)
    print('xoff: ',xoff_tot[chi==chi.min()])
    print('yoff: ',yoff_tot[chi==chi.min()])
#    print 'vsys: ',vsys_tot[chi==chi.min()]
    print('chi: ',chi.min()/nu)
#    plt.figure()
#    plt.plot(vsys_tot,chi/nu,'.')
#    plt.axvline(vsys_tot[chi==chi.min()])
#    plt.xlabel('vsys')
    plt.subplot(211)
    plt.plot(xoff_tot,chi/nu,'.')
    plt.axvline(xoff_tot[chi==chi.min()])
    plt.xlabel('xoff')
    plt.subplot(212)
    plt.plot(yoff_tot,chi/nu,'.')
    plt.axvline(yoff_tot[chi==chi.min()])
    plt.xlabel('yoff')
    print('%r minutes' % ((time.clock()-start)/60.))
    #print(chi)
