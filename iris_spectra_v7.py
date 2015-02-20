# Krishna Mooroogen
# Mullard Space Science Laboratory
# Iris spectral analysisi tool
# Supervisor: Dr David Williams
# Feburary 2014
# Transform data cube into 1d spectral data by pixel 
# Add offset and calculate errors 
# Fitting
# Calculate wave length
# Calculate line properties 
# Calculate line ratio
# Optical depth and path length
# Pixel by pixel analysis

import sys
import os
import pylab
import pyfits
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.gridspec as gridspec
import math
import scipy
from scipy import integrate
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.cm as cm
import scipy.ndimage
import matplotlib.mlab as mlab
import scipy.interpolate as interpolate
import scipy.optimize as optimize
from shapely.geometry import LineString
from pandas import DataFrame
from scipy.signal import argrelextrema
import itertools
from operator import itemgetter
import sunpy
from scipy.misc import imresize


def collapse(x1,x2,y1,y2,rasterfile='/Users/krishnamooroogen/Documents/PHYSICS/MSSL_projectfiles/instrument_data/iris/data_for_python/iris_l2_20140130_202110_si14.fits',line=1394,plot=True) :
    # collapse 3D data cube to 1d image data by (x,y) and cut spectral profile to len 75 pixel
    # Plot 2D image of spectral data against y for the zeroth pixel to show where y lims need to be placed for mapping
    # remember to chnage ocmstants and subpot shape between files
    '''
    if line==1394:
        path = ('/Users/krishnamooroogen/Documents/PHYSICS/MSSL_projectfiles/images/iris/v7/2014_01_30_20/homogenous/ratio/%s%s_%s%s'%(x1,x2,y1,y2))
        if not os.path.exists(path):
            os.makedirs(path)
    if line==1403:
        path = ('/Users/krishnamooroogen/Documents/PHYSICS/MSSL_projectfiles/images/iris/v7/2014_01_30_20/homogenous/ratio/%s%s_%s%s'%(x1,x2,(y1+5),(y2+5)))
        if not os.path.exists(path):
            os.makedirs(path)
    '''    
    spec_data = pyfits.open(rasterfile)[0].data
    #print 'image shape',spec_data.shape
    spec1d = (np.sum(np.sum(spec_data[x1:x2,y1:y2], axis=1), axis=0))
    x=np.arange(0,len(spec1d),1)
    mu = x[spec1d.argmax()]
    spec1d=spec1d[mu-37.5:mu+37.5]
    xl = np.arange(0,len(spec1d),1)
    #80
    #37.5

    
    if plot:
        
        for i in range(x1,(x1+4),1):
            plt.figure()
            plt.title('2D Spectra %s %s_%s_%s_%s'%(line,x1,x2,y1,y2))
            spec2d = spec_data[i]
            plt.imshow(np.log10(spec2d+1),cmap = cm.Greys)
            plt.figure()
            plt.plot(spec2d.sum(0))
            '''
            filename=os.path.join(path,'2014_01_30_20_si%s_2D_%s%s_%s%s.png'%(line,x1,x2,y1,y2))
            pylab.savefig(filename)
            plt.close()
            '''    
    return spec1d,xl

def file_info(file='/Users/krishnamooroogen/Documents/PHYSICS/MSSL_projectfiles/instrument_data/iris/rasterfiles/iris_l2_20140110_200012_3820011488_raster_t000_r00000.fits') :
    # get exposure time from main raterfile

    f=pyfits.open(file)[0].header
    EXPTIME = f['EXPTIME']
    FOVX = f['FOVX']
    FOVY = f['FOVY']
    XCEN = f['XCEN']
    YCEN = f['YCEN']
    print 'OBSID',f['OBSID']
    return [EXPTIME,FOVX,FOVY,XCEN,YCEN]

def Iris_prep(spec1d,EXPTIME,line=1394,plot=False) :
    # Fixing offset for errors
    # Calculating erros
    offset = 0-spec1d.min()
    
    if plot:
        plt.figure()
        plt.title('Iris Si IV %s $\AA$.'%line)
        plt.xlabel('Spectral index.')
        plt.ylabel('ADU.')
        plt.plot(spec1d)
        plt.axhline(0,color='r',ls='--',label='Offset difference of %s'%(int(offset)))
        plt.axhline(spec1d.min(),color='r',ls='--')
        plt.annotate ('', (20, spec1d.min()), (20, 0), arrowprops={'arrowstyle':'<->'})
        plt.legend(loc='best',prop={'size':8})
    
    spec1d            = spec1d+offset
    
    Gain              = 6.0#electrons per dn
    conversion_factor = 1.5#electrons per photon
    spec1d            = ((spec1d*Gain)*(1.0/conversion_factor))/EXPTIME#photons per second
    Stat_error        = np.sqrt(spec1d)
    read_noise        = (20.0*(1.0/conversion_factor))/EXPTIME
    
    Error = np.sqrt(Stat_error**2 + read_noise**2)
    
    return spec1d, Error

def setting_x_to_wavelength(spec1d,fitted_parameters,Lambda):
    # Adjust axis to wavlengh from pixels
    
    Lambdax = fitted_parameters[2]
    dLambda = 0.012730992043
    startlam = (Lambda-(Lambdax*dLambda))
    dist = dLambda*len(spec1d)
    end = (startlam+dist)-0.01
    step = (end-startlam)/len(spec1d)
    x_wavelength = np.arange(startlam,end,step)
    print len(x_wavelength)
    print len(spec1d)

    return x_wavelength

def label(peaks) :
    # Annotate lines
    for i in peaks:
        plt.text(i[1],(i[0]*0.7),'%1.2f $\AA$'%i[1],fontsize=10)
        plt.axvline(i[1],color='y',ls='-')

def gauss(x,a,sigma,mu,bg) :

    return ((a/(sigma*np.sqrt(2*math.pi)))*np.exp((-(x-mu)**2)/(2*sigma**2))+bg)

def Voigt(x,a, sigma, gamma,mu,bg) :

    z  = ((x-mu)+1j*gamma)/(sigma*np.sqrt(2))
    ff = scipy.special.wofz(z)
    vf = pylab.real(ff)
    
    CC = np.array(vf)
    CCNormalised = CC/CC.max()
    CC = CCNormalised*a+bg
    
    return np.array(CC)

def lorentz(x,a,gamma,mu,bg) :

    return (((a/math.pi)*((0.5*gamma)/((x-mu)**2+(0.5*gamma)**2)))+bg)

def fit(spec1d, x,gf, peaks, num_peaks, gamma_est = 0, sigma_est = 1, bg_est = 0,voigtian=False, voigtian2=False,lorentzian=False, gaussian=False, plot=False) :
    # Fiitng proceedure for Voigt, Gauss and Lorentz
    # smooths data through filter
    # Establishes paramter estimates
    # Fits data
    # Calculate line properties
    # Lorentzian and Gaussina not update as out of use
    # Might need to adjust for absorption lines
    
    k_b = 1.38E-23
    m = (28.0833)*(1.66E-27)
    c = 3E8
    
    spec_smdata = scipy.ndimage.filters.gaussian_filter(spec1d,gf)
    
    
    if voigtian:
        
        if num_peaks==1:
            
            est_voigt = [(peaks[0][0]-200),sigma_est,gamma_est,peaks[0][1],bg_est]
            #est_voigt = [15593,sigma_est,gamma_est,1393.95,bg_est]
            coeff,cov = scipy.optimize.curve_fit(Voigt,x,spec_smdata,est_voigt)
        
            amp = coeff[0]
            sigma = coeff[1]
            gamma = coeff[2]
            mu = coeff[3]
            bg = coeff[4]
       
            fit = Voigt(x,coeff[0],coeff[1],coeff[2],coeff[3],coeff[4])-bg
            
            cov=np.sqrt(cov)
            amp_er = cov[0][0]
            sigma_er = cov[1][1]
            gamma_er = cov[2][2]
            mu_er = cov[3][3]
            bg_er = cov[4][4]
            
            fit_area = amp*sigma*np.sqrt(2*math.pi)#change later to vcdf
            area_er = fit_area*np.sqrt((amp_er/amp)**2+(sigma_er/sigma)**2)
        
            T = (((sigma/mu)**2)*(c**2)*m)/(2*k_b)
            T_er = np.sqrt((((((2*sigma/mu**2))*(c**2)*m)/(2*k_b))*sigma_er**2)+((((-2*(sigma**2)/mu))*(c**2)*m)/(2*k_b)*mu_er**2))#dodgy
            FWHM = (0.5346*2*gamma) + np.sqrt((2*sigma*np.sqrt(2*np.log(2)))**2+(0.2166*(2*gamma)**2))
            chi2 = (((spec1d-fit)**2).sum())/(len(spec1d)-5)
        
            fitted_parameters = [amp,sigma,gamma,mu,bg,T,fit_area,chi2,FWHM]
            fitted_parameters_errs =[amp_er,sigma_er,gamma_er,mu_er,bg_er,T_er,area_er]

        elif num_peaks==2:
            
            #for now gamma and sigma are the same estimate
            vf = lambda x,a1,sigma1,gamma,mu1,bg,a2,mu2,sigma2 : Voigt(x,a1,sigma1,gamma,mu1,bg)+Voigt(x,a2,sigma2,gamma,mu2,bg)
            est_voigt = [peaks[0][0],sigma_est,gamma_est,peaks[0][1],bg_est,peaks[1][0],peaks[1][1],sigma_est]
            coeff,cov = scipy.optimize.curve_fit(vf,x,spec_smdata,est_voigt)
        
            cov = np.sqrt(cov)
            amp1 = coeff[0]
            sigma1 = coeff[1]
            gamma = coeff[2]
            mu1 = coeff[3]
            bg = coeff[4]
            amp2 = coeff[5]
            sigma2 = coeff[7]
            mu2 = coeff[6]
        
            amp1_er = cov[0][0]
            sigma1_er = cov[1][1]
            gamma_er = cov[2][2]
            mu1_er = cov[3][3]
            bg_er = cov[4][4]
            amp2_er = cov[5][5]
            sigma2_er = cov[7][7]
            mu2_er = cov[6][6]
            
            
            fit = Voigt(x,coeff[0],coeff[1],coeff[2],coeff[3],coeff[4])+Voigt(x,coeff[5],coeff[7],coeff[2],coeff[6],coeff[4])-bg
            '''
            amps = [amp1,amp2]
            amp0 = max(amps)
            
            if amps.index(max(amps))==0:
                sigma0 = sigma1
                amp0_er = amp1_er
                sigma0_er = sigma1_er
            elif amps.index(max(amps))==1:
                sigma0 = sigma2
                amp0_er = amp2_er
                sigma0_er = sigma2_er
            '''    
            fit_area = amp1*sigma1*np.sqrt(2*math.pi)
            
            area_er = fit_area*np.sqrt((amp1_er/amp1)**2+(sigma1_er/sigma1)**2)
        
            T1 = (((sigma1/mu1)**2)*(c**2)*m)/(2*k_b)# maybe new temp calc 
            T2 = (((sigma2/mu1)**2)*(c**2)*m)/(2*k_b)
            T_er = np.sqrt((((((2*sigma1/mu1**2))*(c**2)*m)/(2*k_b))*sigma1_er**2)+((((-2*(sigma1**2)/mu1))*(c**2)*m)/(2*k_b)*mu1_er**2))#dodgy
            FWHM1 = (0.5346*2*gamma) + np.sqrt((2*sigma1*np.sqrt(2*np.log(2)))**2+(0.2166*(2*gamma)**2))
            FWHM2 = (0.5346*2*gamma) + np.sqrt((2*sigma2*np.sqrt(2*np.log(2)))**2+(0.2166*(2*gamma)**2))
            
            chi2 = (((spec1d-fit)**2).sum())/(len(spec1d)-len(est_voigt))
        
            fitted_parameters = [amp1,sigma1,gamma,mu1,bg,T1,fit_area,chi2,FWHM1,amp2,sigma2,mu2,T2,FWHM2]
            fitted_parameters_errs =[amp1_er,sigma1_er,gamma_er,mu1_er,bg_er,T_er,area_er,amp2_er,sigma2_er,mu2_er]

        elif num_peaks==3:
            
            
            vf = lambda x,a1,sigma1,gamma,mu1,bg,a2,mu2,sigma2,a3,mu3,sigma3 : Voigt(x,a1,sigma1,gamma,mu1,bg)+Voigt(x,a2,sigma2,gamma,mu2,bg)+Voigt(x,a3,sigma3,gamma,mu3,bg)
            #est_voigt = [70000,sigma_est,gamma_est,38,bg_est,736,peaks[1][1],sigma_est,724,peaks[2][1],sigma_est]
            est_voigt = [peaks[0][0],sigma_est,gamma_est,peaks[0][1],bg_est,peaks[1][0],peaks[1][1],sigma_est,peaks[2][0],peaks[2][1],sigma_est]
            coeff,cov = scipy.optimize.curve_fit(vf,x,spec_smdata,est_voigt)
    
            cov = np.sqrt(cov)
            amp1 = coeff[0]
            sigma1 = coeff[1]
            gamma = coeff[2]
            mu1 = coeff[3]
            bg = coeff[4]
            amp2 = coeff[5]
            sigma2 = coeff[7]
            mu2 = coeff[6]
            amp3 = coeff[8]
            sigma3 = coeff[10]
            mu3 = coeff[9]
            amp1_er = cov[0][0]
            sigma1_er = cov[1][1]
            gamma_er = cov[2][2]
            mu1_er = cov[3][3]
            bg_er = cov[4][4]
            amp2_er = cov[5][5]
            sigma2_er = cov[7][7]
            mu2_er = cov[6][6]
            amp3_er = cov[8][8]
            sigma3_er = cov[10][10]
            mu3_er = cov[9][9]

    
            fit = Voigt(x,coeff[0],coeff[1],coeff[2],coeff[3],coeff[4])+Voigt(x,coeff[5],coeff[7],coeff[2],coeff[6],coeff[4])+Voigt(x,coeff[8],coeff[10],coeff[2],coeff[9],coeff[4])-bg
            '''
            amps = [amp1,amp2,amp3]
            amp0 = max(amps)
            
            if amps.index(max(amps))==0:
                sigma0 = sigma1
                amp0_er = amp1_er
                sigma0_er = sigma1_er
            elif amps.index(max(amps))==1:
                sigma0 = sigma2
                amp0_er = amp2_er
                sigma0_er = sigma2_er
            elif amps.index(max(amps))==2:
                sigma0 = sigma3
                amp0_er = amp3_er
                sigma0_er = sigma3_er
            '''    
            fit_area = amp1*sigma1*np.sqrt(2*math.pi)
        
            area_er = fit_area*np.sqrt((amp1_er/amp1)**2+(sigma1_er/sigma1)**2)
    
            T1 = (((sigma1/mu1)**2)*(c**2)*m)/(2*k_b)#check 
            T2 = (((sigma2/mu2)**2)*(c**2)*m)/(2*k_b)
            T3 = (((sigma3/mu3)**2)*(c**2)*m)/(2*k_b)
            
            T_er = np.sqrt((((((2*sigma1/mu1**2))*(c**2)*m)/(2*k_b))*sigma1_er**2)+((((-2*(sigma1**2)/mu1))*(c**2)*m)/(2*k_b)*mu1_er**2))#dodgy

            FWHM1 = (0.5346*2*gamma) + np.sqrt((2*sigma1*np.sqrt(2*np.log(2)))**2+(0.2166*(2*gamma)**2))
            FWHM2 = (0.5346*2*gamma) + np.sqrt((2*sigma2*np.sqrt(2*np.log(2)))**2+(0.2166*(2*gamma)**2))
            FWHM3 = (0.5346*2*gamma) + np.sqrt((2*sigma3*np.sqrt(2*np.log(2)))**2+(0.2166*(2*gamma)**2))
    
            chi2 = (((spec1d-fit)**2).sum())/(len(spec1d)-len(est_voigt))
    
            fitted_parameters = [amp1,sigma1,gamma,mu1,bg,T1,fit_area,chi2,FWHM1,amp2,sigma2,mu2,T2,FWHM2,amp3,sigma3,mu3,T3,FWHM3]
            fitted_parameters_errs =[amp1_er,sigma1_er,gamma_er,mu1_er,bg_er,T_er,area_er,amp2_er,sigma2_er,mu2_er,amp3_er,sigma3_er,mu3_er]
                
        elif num_peaks==4:
            
            vf = lambda x,a1,sigma1,gamma,mu1,bg,a2,mu2,sigma2,a3,mu3,sigma3,a4,mu4,sigma4 : Voigt(x,a1,sigma1,gamma,mu1,bg)+Voigt(x,a2,sigma2,gamma,mu2,bg)+Voigt(x,a3,sigma3,gamma,mu3,bg)+Voigt(x,a4,sigma4,gamma,mu4,bg)
           
            est_voigt = [peaks[0][0],sigma_est,gamma_est,peaks[0][1],bg_est,peaks[1][0],peaks[1][1],sigma_est,peaks[2][0],peaks[2][1],sigma_est,peaks[3][0],peaks[3][1],sigma_est]
                
            coeff,cov = scipy.optimize.curve_fit(vf,x,spec_smdata,est_voigt)
        
            cov = np.sqrt(cov)
            amp1 = coeff[0]
            sigma1 = coeff[1]
            gamma = coeff[2]
            mu1 = coeff[3]
            bg = coeff[4]
            amp2 = coeff[5]
            sigma2 = coeff[7]
            mu2 = coeff[6]
            amp3 = coeff[8]
            mu3 = coeff[9]
            sigma3 = coeff[10]
            amp4 = coeff[11]
            mu4 = coeff[12]
            sigma4 = coeff[13]
                
        
            amp1_er = cov[0][0]
            sigma1_er = cov[1][1]
            gamma_er = cov[2][2]
            mu1_er = cov[3][3]
            bg_er = cov[4][4]
            amp2_er = cov[5][5]
            sigma2_er = cov[7][7]
            mu2_er = cov[6][6]
            amp3_er = cov[8][8]
            mu3_er = cov[9][9]
            sigma3_er = cov[10][10]
            amp4_er = cov[11][11]
            mu4_er = cov[12][12]
            sigma4_er = cov[13][13]
           
        
        
            fit = Voigt(x,coeff[0],coeff[1],coeff[2],coeff[3],coeff[4])+Voigt(x,coeff[5],coeff[7],coeff[2],coeff[6],coeff[4])+Voigt(x,coeff[8],coeff[10],coeff[2],coeff[9],coeff[4])+Voigt(x,coeff[11],coeff[13],coeff[2],coeff[12],coeff[4])-bg
            
            '''
            amps = [amp1,amp2,amp3,amp4]
            amp0 = max(amps)
            
            if amps.index(max(amps))==0:
                sigma0 = sigma1
                amp0_er = amp1_er
                sigma0_er = sigma1_er
            
            elif amps.index(max(amps))==1:
                sigma0 = sigma2
                amp0_er = amp2_er
                sigma0_er = sigma2_er
            
            elif amps.index(max(amps))==2:
                sigma0 = sigma3
                amp0_er = amp3_er
                sigma0_er = sigma3_er
            
            elif amps.index(max(amps))==3:
                sigma0 = sigma4
                amp0_er = amp4_er
                sigma0_er = sigma4_er
            '''
         
            fit_area = amp1*sigma1*np.sqrt(2*math.pi)
        
            area_er = fit_area*np.sqrt((amp1_er/amp1)**2+(sigma1_er/sigma1)**2)

            T1 = (((sigma1/mu1)**2)*(c**2)*m)/(2*k_b)#need real calc
            T2 = (((sigma2/mu2)**2)*(c**2)*m)/(2*k_b)
            T3 = (((sigma3/mu3)**2)*(c**2)*m)/(2*k_b)
            T4 = (((sigma4/mu4)**2)*(c**2)*m)/(2*k_b)
        
            T_er = np.sqrt((((((2*sigma1/mu1**2))*(c**2)*m)/(2*k_b))*sigma1_er**2)+((((-2*(sigma1**2)/mu1))*(c**2)*m)/(2*k_b)*mu1_er**2))#dodgy
            FWHM1 = (0.5346*2*gamma) + np.sqrt((2*sigma1*np.sqrt(2*np.log(2)))**2+(0.2166*(2*gamma)**2))
            FWHM2 = (0.5346*2*gamma) + np.sqrt((2*sigma2*np.sqrt(2*np.log(2)))**2+(0.2166*(2*gamma)**2))
            FWHM3 = (0.5346*2*gamma) + np.sqrt((2*sigma3*np.sqrt(2*np.log(2)))**2+(0.2166*(2*gamma)**2))
            FWHM4 = (0.5346*2*gamma) + np.sqrt((2*sigma4*np.sqrt(2*np.log(2)))**2+(0.2166*(2*gamma)**2))
        
            chi2 = (((spec1d-fit)**2).sum())/(len(spec1d)-len(est_voigt))
        
            fitted_parameters = [amp1,sigma1,gamma,mu1,bg,T1,fit_area,chi2,FWHM1,amp2,sigma2,mu2,T2,FWHM2,amp3,sigma3,mu3,T3,FWHM3,amp4,sigma4,mu4,T4,FWHM4]
            fitted_parameters_errs =[amp1_er,sigma1_er,gamma_er,mu1_er,bg_er,T_er,area_er,amp2_er,sigma2_er,mu2_er,amp3_er,sigma3_er,mu3_er,amp4_er,sigma4_er,mu4_er]
        
        elif num_peaks==5:
           
                    
            vf = lambda x,a1,sigma1,gamma,mu1,bg,a2,mu2,sigma2,a3,mu3,sigma3,a4,mu4,sigma4,a5,mu5,sigma5 : Voigt(x,a1,sigma1,gamma,mu1,bg)+Voigt(x,a2,sigma2,gamma,mu2,bg)+Voigt(x,a3,sigma3, gamma,mu3,bg)+Voigt(x,a4,sigma4,gamma,mu4,bg)+Voigt(x,a5,sigma5,gamma,mu5,bg)

            est_voigt = [peaks[0][0],sigma_est,gamma_est,peaks[0][1],bg_est,peaks[1][0],peaks[1][1],sigma_est,peaks[2][0],peaks[2][1],sigma_est,peaks[3][0],peaks[3][1],sigma_est,peaks[4][0],peaks[4][1],sigma_est]
                    
            coeff,cov = scipy.optimize.curve_fit(vf,x,spec_smdata,est_voigt)
                
            cov = np.sqrt(cov)
            amp1 = coeff[0]
            sigma1 = coeff[1]
            gamma = coeff[2]
            mu1 = coeff[3]
            bg = coeff[4]
            amp2 = coeff[5]
            sigma2 = coeff[7]
            mu2 = coeff[6]
            amp3 = coeff[8]
            mu3 = coeff[9]
            sigma3 = coeff[10]
            amp4 = coeff[11]
            mu4 = coeff[12]
            sigma4 = coeff[13]
            amp5 = coeff[14]
            mu5 = coeff[15]
            sigma5 = coeff[16]
                    
                    
            amp1_er = cov[0][0]
            sigma1_er = cov[1][1]
            gamma_er = cov[2][2]
            mu1_er = cov[3][3]
            bg_er = cov[4][4]
            amp2_er = cov[5][5]
            sigma2_er = cov[7][7]
            mu2_er = cov[6][6]
            amp3_er = cov[8][8]
            mu3_er = cov[9][9]
            sigma3_er = cov[10][10]
            amp4_er = cov[11][11]
            mu4_er = cov[12][12]
            sigma4_er = cov[13][13]
            amp5_er = cov[14][14]
            mu5_er = cov[15][15]
            sigma5_er = cov[16][16]
            
                    
            fit = Voigt(x,coeff[0],coeff[1],coeff[2],coeff[3],coeff[4])+Voigt(x,coeff[5],coeff[7],coeff[2],coeff[6],coeff[4])+Voigt(x,coeff[8],coeff[10],coeff[2],coeff[9],coeff[4])+Voigt(x,coeff[11],coeff[13],coeff[2],coeff[12],coeff[4])+Voigt(x,coeff[14],coeff[16],coeff[2],coeff[15],coeff[4])-bg
            '''
            amps = [amp1,amp2,amp3,amp4,amp5]
            amp0 = max(amps)
            
            if amps.index(max(amps))==0:
                sigma0 = sigma1
                amp0_er = amp1_er
                sigma0_er = sigma1_er
            
            elif amps.index(max(amps))==1:
                sigma0 = sigma2
                amp0_er = amp2_er
                sigma0_er = sigma2_er
            
            elif amps.index(max(amps))==2:
                sigma0 = sigma3
                amp0_er = amp3_er
                sigma0_er = sigma3_er
            
            elif amps.index(max(amps))==3:
                sigma0 = sigma4
                amp0_er = amp4_er
                sigma0_er = sigma4_er
            
            elif amps.index(max(amps))==4:
                sigma0 = sigma5
                amp0_er = amp5_er
                sigma0_er = sigma5_er
            '''    

            fit_area = amp1*sigma1*np.sqrt(2*math.pi)

            area_er = fit_area*np.sqrt((amp1_er/amp1)**2+(sigma1_er/sigma1)**2)

            T1 = (((sigma1/mu1)**2)*(c**2)*m)/(2*k_b)#need real calc
            T2 = (((sigma2/mu2)**2)*(c**2)*m)/(2*k_b)
            T3 = (((sigma3/mu3)**2)*(c**2)*m)/(2*k_b)
            T4 = (((sigma4/mu4)**2)*(c**2)*m)/(2*k_b)
            T5 = (((sigma5/mu5)**2)*(c**2)*m)/(2*k_b)
                    
            T_er = np.sqrt((((((2*sigma1/mu1**2))*(c**2)*m)/(2*k_b))*sigma1_er**2)+((((-2*(sigma1**2)/mu1))*(c**2)*m)/(2*k_b)*mu1_er**2))#dodgy
            FWHM1 = (0.5346*2*gamma) + np.sqrt((2*sigma1*np.sqrt(2*np.log(2)))**2+(0.2166*(2*gamma)**2))
            FWHM2 = (0.5346*2*gamma) + np.sqrt((2*sigma2*np.sqrt(2*np.log(2)))**2+(0.2166*(2*gamma)**2))
            FWHM3 = (0.5346*2*gamma) + np.sqrt((2*sigma3*np.sqrt(2*np.log(2)))**2+(0.2166*(2*gamma)**2))
            FWHM4 = (0.5346*2*gamma) + np.sqrt((2*sigma4*np.sqrt(2*np.log(2)))**2+(0.2166*(2*gamma)**2))
            FWHM5 = (0.5346*2*gamma) + np.sqrt((2*sigma5*np.sqrt(2*np.log(2)))**2+(0.2166*(2*gamma)**2))
                    
            chi2 = (((spec1d-fit)**2).sum())/(len(spec1d)-len(est_voigt))
                    
            fitted_parameters = [amp1,sigma1,gamma,mu1,bg,T1,fit_area,chi2,FWHM1,amp2,sigma2,mu2,T2,FWHM2,amp3,sigma3,mu3,T3,FWHM3,amp4,sigma4,mu4,T4,FWHM4,amp5,sigma5,mu5,T5,FWHM5]
            
            fitted_parameters_errs =[amp1_er,sigma1_er,gamma_er,mu1_er,bg_er,T_er,area_er,amp2_er,sigma2_er,mu2_er,amp3_er,sigma3_er,mu3_er,amp4_er,sigma4_er,mu4_er,amp5_er,sigma5_er,mu5_er]
                
        elif num_peaks==6:
            
                    
            vf = lambda x,a1,sigma1,gamma,mu1,bg,a2,mu2,sigma2,a3,mu3,sigma3,a4,mu4,sigma4,a5,mu5,sigma5,a6,mu6,sigma6 : Voigt(x,a1,sigma1,gamma,mu1,bg)+Voigt(x,a2,sigma2,gamma,mu2,bg)+Voigt(x,a3,sigma3, gamma,mu3,bg)+Voigt(x,a4,sigma4,gamma,mu4,bg)+Voigt(x,a5,sigma5,gamma,mu5,bg)+Voigt(x,a6,sigma6,gamma,mu6,bg)
                    
            est_voigt = [peaks[0][0],sigma_est,gamma_est,peaks[0][1],bg_est,peaks[1][0],peaks[1][1],sigma_est,peaks[2][0],peaks[2][1],sigma_est,peaks[3][0],peaks[3][1],sigma_est,peaks[4][0],peaks[4][1],sigma_est,peaks[5][0],peaks[5][1],sigma_est]
                    
            coeff,cov = scipy.optimize.curve_fit(vf,x,spec_smdata,est_voigt)
                    
            cov = np.sqrt(cov)
            amp1 = coeff[0]
            sigma1 = coeff[1]
            gamma = coeff[2]
            mu1 = coeff[3]
            bg = coeff[4]
            amp2 = coeff[5]
            sigma2 = coeff[7]
            mu2 = coeff[6]
            amp3 = coeff[8]
            mu3 = coeff[9]
            sigma3 = coeff[10]
            amp4 = coeff[11]
            mu4 = coeff[12]
            sigma4 = coeff[13]
            amp5 = coeff[14]
            mu5 = coeff[15]
            sigma5 = coeff[16]
            amp6 = coeff[17]
            mu6 = coeff[18]
            sigma6 = coeff[19]
            
            amp1_er = cov[0][0]
            sigma1_er = cov[1][1]
            gamma_er = cov[2][2]
            mu1_er = cov[3][3]
            bg_er = cov[4][4]
            amp2_er = cov[5][5]
            sigma2_er = cov[7][7]
            mu2_er = cov[6][6]
            amp3_er = cov[8][8]
            mu3_er = cov[9][9]
            sigma3_er = cov[10][10]
            amp4_er = cov[11][11]
            mu4_er = cov[12][12]
            sigma4_er = cov[13][13]
            amp5_er = cov[14][14]
            mu5_er = cov[15][15]
            sigma5_er = cov[16][16]
            amp6_er = cov[17][17]
            mu6_er = cov[18][18]
            sigma6_er = cov[19][19]
            
            
            fit = Voigt(x,coeff[0],coeff[1],coeff[2],coeff[3],coeff[4])+Voigt(x,coeff[5],coeff[7],coeff[2],coeff[6],coeff[4])+Voigt(x,coeff[8],coeff[10],coeff[2],coeff[9],coeff[4])+Voigt(x,coeff[11],coeff[13],coeff[2],coeff[12],coeff[4])+Voigt(x,coeff[14],coeff[16],coeff[2],coeff[15],coeff[4])+Voigt(x,coeff[17],coeff[19],coeff[2],coeff[18],coeff[4])-bg
            '''        
            amps = [amp1,amp2,amp3,amp4,amp5]
            amp0 = max(amps)
            
            if amps.index(max(amps))==0:
                sigma0 = sigma1
                amp0_er = amp1_er
                sigma0_er = sigma1_er
            
            elif amps.index(max(amps))==1:
                sigma0 = sigma2
                amp0_er = amp2_er
                sigma0_er = sigma2_er
            
            elif amps.index(max(amps))==2:
                sigma0 = sigma3
                amp0_er = amp3_er
                sigma0_er = sigma3_er
            
            elif amps.index(max(amps))==3:
                sigma0 = sigma4
                amp0_er = amp4_er
                sigma0_er = sigma4_er
            
            elif amps.index(max(amps))==4:
                sigma0 = sigma5
                amp0_er = amp5_er
                sigma0_er = sigma5_er

            elif amps.index(max(amps))==5:
                sigma0 = sigma6
                amp0_er = amp6_er
                sigma0_er = sigma6_er
            '''    

            fit_area = amp1*sigma1*np.sqrt(2*math.pi)

            area_er = fit_area*np.sqrt((amp1_er/amp1)**2+(sigma1_er/sigma1)**2)

            T1 = (((sigma1/mu1)**2)*(c**2)*m)/(2*k_b)#need real calc
            T2 = (((sigma2/mu2)**2)*(c**2)*m)/(2*k_b)
            T3 = (((sigma3/mu3)**2)*(c**2)*m)/(2*k_b)
            T4 = (((sigma4/mu4)**2)*(c**2)*m)/(2*k_b)
            T5 = (((sigma5/mu5)**2)*(c**2)*m)/(2*k_b)
            T6 = (((sigma6/mu6)**2)*(c**2)*m)/(2*k_b)
                    
            T_er = np.sqrt((((((2*sigma1/mu1**2))*(c**2)*m)/(2*k_b))*sigma1_er**2)+((((-2*(sigma1**2)/mu1))*(c**2)*m)/(2*k_b)*mu1_er**2))#dodgy
            FWHM1 = (0.5346*2*gamma) + np.sqrt((2*sigma1*np.sqrt(2*np.log(2)))**2+(0.2166*(2*gamma)**2))
            FWHM2 = (0.5346*2*gamma) + np.sqrt((2*sigma2*np.sqrt(2*np.log(2)))**2+(0.2166*(2*gamma)**2))
            FWHM3 = (0.5346*2*gamma) + np.sqrt((2*sigma3*np.sqrt(2*np.log(2)))**2+(0.2166*(2*gamma)**2))
            FWHM4 = (0.5346*2*gamma) + np.sqrt((2*sigma4*np.sqrt(2*np.log(2)))**2+(0.2166*(2*gamma)**2))
            FWHM5 = (0.5346*2*gamma) + np.sqrt((2*sigma5*np.sqrt(2*np.log(2)))**2+(0.2166*(2*gamma)**2))
            FWHM6 = (0.5346*2*gamma) + np.sqrt((2*sigma6*np.sqrt(2*np.log(2)))**2+(0.2166*(2*gamma)**2))
                    
            chi2 = (((spec1d-fit)**2).sum())/(len(spec1d)-len(est_voigt))
                    
            fitted_parameters = [amp1,sigma1,gamma,mu1,bg,T1,fit_area,chi2,FWHM1,amp2,sigma2,mu2,T2,FWHM2,amp3,sigma3,mu3,T3,FWHM3,amp4,sigma4,mu4,T4,FWHM4,amp5,sigma5,mu5,T5,FWHM5,amp6,sigma6,mu6,T6,FWHM6]
                    
            fitted_parameters_errs =[amp1_er,sigma1_er,gamma_er,mu1_er,bg_er,T_er,area_er,amp2_er,sigma2_er,mu2_er,amp3_er,sigma3_er,mu3_er,amp4_er,sigma4_er,mu4_er,amp5_er,sigma5_er,mu5_er,amp6_er,sigma6_er,mu6_er]
            
        elif num_peaks==7:
                    
            vf = lambda x,a1,sigma1,gamma,mu1,bg,a2,mu2,sigma2,a3,mu3,sigma3,a4,mu4,sigma4,a5,mu5,sigma5,a6,mu6,sigma6,a7,mu7,sigma7 : Voigt(x,a1,sigma1,gamma,mu1,bg)+Voigt(x,a2,sigma2,gamma,mu2,bg)+Voigt(x,a3,sigma3, gamma,mu3,bg)+Voigt(x,a4,sigma4,gamma,mu4,bg)+Voigt(x,a5,sigma5,gamma,mu5,bg)+Voigt(x,a6,sigma6,gamma,mu6,bg)+Voigt(x,a7,sigma7,gamma,mu7,bg)
                    
            est_voigt = [peaks[0][0],sigma_est,gamma_est,peaks[0][1],bg_est,peaks[1][0],peaks[1][1],sigma_est,peaks[2][0],peaks[2][1],sigma_est,peaks[3][0],peaks[3][1],sigma_est,peaks[4][0],peaks[4][1],sigma_est,peaks[5][0],peaks[5][1],sigma_est,peaks[6][0],peaks[6][1],sigma_est]
                    
            coeff,cov = scipy.optimize.curve_fit(vf,x,spec_smdata,est_voigt)
                    
            cov = np.sqrt(cov)
            amp1 = coeff[0]
            sigma1 = coeff[1]
            gamma = coeff[2]
            mu1 = coeff[3]
            bg = coeff[4]
            amp2 = coeff[5]
            sigma2 = coeff[7]
            mu2 = coeff[6]
            amp3 = coeff[8]
            mu3 = coeff[9]
            sigma3 = coeff[10]
            amp4 = coeff[11]
            mu4 = coeff[12]
            sigma4 = coeff[13]
            amp5 = coeff[14]
            mu5 = coeff[15]
            sigma5 = coeff[16]
            amp6 = coeff[17]
            mu6 = coeff[18]
            sigma6 = coeff[19]
            amp7 = coeff[20]
            mu7 = coeff[21]
            sigma7 = coeff[22]
                    
            amp1_er = cov[0][0]
            sigma1_er = cov[1][1]
            gamma_er = cov[2][2]
            mu1_er = cov[3][3]
            bg_er = cov[4][4]
            amp2_er = cov[5][5]
            sigma2_er = cov[7][7]
            mu2_er = cov[6][6]
            amp3_er = cov[8][8]
            mu3_er = cov[9][9]
            sigma3_er = cov[10][10]
            amp4_er = cov[11][11]
            mu4_er = cov[12][12]
            sigma4_er = cov[13][13]
            amp5_er = cov[14][14]
            mu5_er = cov[15][15]
            sigma5_er = cov[16][16]
            amp6_er = cov[17][17]
            mu6_er = cov[18][18]
            sigma6_er = cov[19][19]
            amp7_er = cov[20][20]
            mu7_er = cov[21][21]
            sigma7_er = cov[22][22]
                
                    
            fit = Voigt(x,coeff[0],coeff[1],coeff[2],coeff[3],coeff[4])+Voigt(x,coeff[5],coeff[7],coeff[2],coeff[6],coeff[4])+Voigt(x,coeff[8],coeff[10],coeff[2],coeff[9],coeff[4])+Voigt(x,coeff[11],coeff[13],coeff[2],coeff[12],coeff[4])+Voigt(x,coeff[14],coeff[16],coeff[2],coeff[15],coeff[4])+Voigt(x,coeff[17],coeff[19],coeff[2],coeff[18],coeff[4])+Voigt(x,coeff[20],coeff[22],coeff[2],coeff[21],coeff[4])-bg
                    
            '''
            amps = [amp1,amp2,amp3,amp4,amp5]
            amp0 = max(amps)
            
            if amps.index(max(amps))==0:
                sigma0 = sigma1
                amp0_er = amp1_er
                sigma0_er = sigma1_er
            
            elif amps.index(max(amps))==1:
                sigma0 = sigma2
                amp0_er = amp2_er
                sigma0_er = sigma2_er
            
            elif amps.index(max(amps))==2:
                sigma0 = sigma3
                amp0_er = amp3_er
                sigma0_er = sigma3_er
            
            elif amps.index(max(amps))==3:
                sigma0 = sigma4
                amp0_er = amp4_er
                sigma0_er = sigma4_er
            
            elif amps.index(max(amps))==4:
                sigma0 = sigma5
                amp0_er = amp5_er
                sigma0_er = sigma5_er
            
            elif amps.index(max(amps))==5:
                sigma0 = sigma6
                amp0_er = amp6_er
                sigma0_er = sigma6_er
            
            elif amps.index(max(amps))==6:
                sigma0 = sigma7
                amp0_er = amp7_er
                sigma0_er = sigma7_er
            '''    


            fit_area = amp1*sigma1*np.sqrt(2*math.pi)

            area_er = fit_area*np.sqrt((amp1_er/amp1)**2+(sigma1_er/sigma1)**2)

            T1 = (((sigma1/mu1)**2)*(c**2)*m)/(2*k_b)#need real calc
            T2 = (((sigma2/mu2)**2)*(c**2)*m)/(2*k_b)
            T3 = (((sigma3/mu3)**2)*(c**2)*m)/(2*k_b)
            T4 = (((sigma4/mu4)**2)*(c**2)*m)/(2*k_b)
            T5 = (((sigma5/mu5)**2)*(c**2)*m)/(2*k_b)
            T6 = (((sigma6/mu6)**2)*(c**2)*m)/(2*k_b)
            T7 = (((sigma7/mu7)**2)*(c**2)*m)/(2*k_b)
                    
            T_er = np.sqrt((((((2*sigma1/mu1**2))*(c**2)*m)/(2*k_b))*sigma1_er**2)+((((-2*(sigma1**2)/mu1))*(c**2)*m)/(2*k_b)*mu1_er**2))#dodgy
            FWHM1 = (0.5346*2*gamma) + np.sqrt((2*sigma1*np.sqrt(2*np.log(2)))**2+(0.2166*(2*gamma)**2))
            FWHM2 = (0.5346*2*gamma) + np.sqrt((2*sigma2*np.sqrt(2*np.log(2)))**2+(0.2166*(2*gamma)**2))
            FWHM3 = (0.5346*2*gamma) + np.sqrt((2*sigma3*np.sqrt(2*np.log(2)))**2+(0.2166*(2*gamma)**2))
            FWHM4 = (0.5346*2*gamma) + np.sqrt((2*sigma4*np.sqrt(2*np.log(2)))**2+(0.2166*(2*gamma)**2))
            FWHM5 = (0.5346*2*gamma) + np.sqrt((2*sigma5*np.sqrt(2*np.log(2)))**2+(0.2166*(2*gamma)**2))
            FWHM6 = (0.5346*2*gamma) + np.sqrt((2*sigma6*np.sqrt(2*np.log(2)))**2+(0.2166*(2*gamma)**2))
            FWHM7 = (0.5346*2*gamma) + np.sqrt((2*sigma7*np.sqrt(2*np.log(2)))**2+(0.2166*(2*gamma)**2))

            chi2 = (((spec1d-fit)**2).sum())/(len(spec1d)-len(est_voigt))
                    
            fitted_parameters = [amp1,sigma1,gamma,mu1,bg,T1,fit_area,chi2,FWHM1,amp2,sigma2,mu2,T2,FWHM2,amp3,sigma3,mu3,T3,FWHM3,amp4,sigma4,mu4,T4,FWHM4,amp5,sigma5,mu5,T5,FWHM5,amp6,sigma6,mu6,T6,FWHM6,amp7,sigma7,mu7,T7,FWHM7]
                    
            fitted_parameters_errs =[amp1_er,sigma1_er,gamma_er,mu1_er,bg_er,T_er,area_er,amp2_er,sigma2_er,mu2_er,amp3_er,sigma3_er,mu3_er,amp4_er,sigma4_er,mu4_er,amp5_er,sigma5_er,mu5_er,amp6_er,sigma6_er,mu6_er,amp7_er,sigma7_er,mu7_er]
                

    elif lorentzian:

        est_lorentz = [amp_est,gamma_est,mu_est,bg_est]
        coeff,cov = scipy.optimize.curve_fit(lorentz,x,spec_smdata,est_lorentz)
        fit = lorentz(x,coeff[0],coeff[1],coeff[2],coeff[3])-coeff[3]

        amp = coeff[0]
        gamma = coeff[1]
        mu = coeff[1]
        bg = coeff[3]
        
        amp_er = cov[0][0]
        gamma_er = cov[1][1]
        mu_er = cov[2][2]
        bg_er = cov[3][3]

        fitted_parameters = [amp,gamma,mu,bg]
        fitted_parameters_errs =[amp_er,gamma_er,mu_er,bg_er]

    elif gaussian:
        
        amp_est=spec1d.max()
        mu_est = x[spec1d.argmax()]
        est_gauss = [amp_est,sigma_est,mu_est,bg_est]
        coeff, cov = scipy.optimize.curve_fit(gauss,x,spec_smdata,est_gauss)
        
        amp = coeff[0]
        sigma = coeff[1]
        mu = coeff[2]
        bg = coeff[3]

        fit = gauss(x,coeff[0],coeff[1],coeff[2],coeff[3])-bg

        amp_er = cov[0][0]
        sigma_er = cov[1][1]
        mu_er = cov[2][2]
        bg_er = cov[3][3] 

        fit_area = amp*sigma*np.sqrt(2*math.pi)
        area_er = fit_area*np.sqrt((amp_er/amp)**2+(sigma_er/sigma)**2)
        spec1d=np.array(spec1d, dtype=np.float)
        fit=np.array(fit, dtype=np.float)
        
        T = (((sigma/mu)**2)*(c**2)*m)/(2*k_b)
        T_er = np.sqrt((((((2*sigma/mu**2))*(c**2)*m)/(2*k_b))*sigma_er**2)+((((-2*(sigma**2)/mu))*(c**2)*m)/(2*k_b)*mu_er**2))
        FWHM = (2*np.sqrt(2*np.log(2))*sigma)
        
        chi2 = (((spec1d-fit)**2).sum())/(len(spec1d)-5)

        fitted_parameters = [amp,sigma,mu,bg,T,fit_area,chi2,FWHM]
        fitted_parameters_errs =[amp_er,sigma_er,mu_er,bg_er,T_er,area_er]
   
    if plot:
        plt.figure()
        plt.plot(x,spec_smdata-bg)
        plt.plot(x,fit)
            
    return fit, fitted_parameters, fitted_parameters_errs

def fitted_ratio(x1,x2,y11,y12,y21,y22,n1,plot=True) :
    # Change path file when switching files
    # Calculates the ratio of the fitted emission lines Si IV 13 & 14 
    
    path = ('/Users/krishnamooroogen/Documents/PHYSICS/MSSL_projectfiles/images/iris/v7/2014_01_31/ratio/average/inhomogenous/%s%s_%s%s'%(x1,x2,y11,y12))
    if not os.path.exists(path):
        os.makedirs(path)

    header =  file_info(file='/Users/krishnamooroogen/Documents/PHYSICS/MSSL_projectfiles/instrument_data/iris/rasterfiles/iris_l2_20140131_084053_3803257203_raster_t000_r00000.fits')
    
    spec1d13,x13 = collapse(x1,x2,y11,y12,rasterfile='/Users/krishnamooroogen/Documents/PHYSICS/MSSL_projectfiles/instrument_data/iris/data_for_python/iris_l2_20140131_084053_si13.fits',line=1394,plot=False)
    spec1d13, Error13 = Iris_prep(spec1d13,header[0],line=1394,plot=False)

    peaks13,num_peaks13 = peakdet(spec1d13,x13,n=n1,plot=False)

    fit13,fitted_parameters13,fitted_parameters_errs13=fit(spec1d13, x13,0,peaks13,num_peaks13,gamma_est = 0.01, sigma_est = 0.2, bg_est = 0, voigtian=True,lorentzian=False, gaussian=False, plot=False)
    
    # my need changing for errors and multiple voigt files
    spec1d14 ,x14 = collapse(x1,x2,y21,y22,rasterfile='/Users/krishnamooroogen/Documents/PHYSICS/MSSL_projectfiles/instrument_data/iris/data_for_python/iris_l2_20140131_084053_si14.fits',line=1403,plot=False)
    spec1d14, Error14 = Iris_prep(spec1d14,header[0],line=1403,plot=False)

    peaks14,num_peaks14 = peakdet(spec1d14,x14,n=n1,plot=False)

    fit14,fitted_parameters14,fitted_parameters_errs14 = fit(spec1d14, x14,0,peaks14,num_peaks14, gamma_est = 0.01, sigma_est = 0.07, bg_est = 0, voigtian=True,lorentzian=False, gaussian=False, plot=False)

    sigma13 = fitted_parameters13[1]
    gamma13 = fitted_parameters13[2]
    sig13er = fitted_parameters_errs13[1]
    gam13er = fitted_parameters_errs13[2]

    area13 = fitted_parameters13[6]
    area13_er = fitted_parameters_errs13[6]

    area14 = fitted_parameters14[6]
    area14_er = fitted_parameters_errs14[6]

    sigma14 = fitted_parameters14[1]
    gamma14 = fitted_parameters14[2]
    sig14er = fitted_parameters_errs14[1]
    gam14er = fitted_parameters_errs14[2]


    ratio = area13/area14
    ratio_error = ratio*np.sqrt((area13_er/area13)**2+(area14_er/area14)**2)
    
    FWHM13 = fitted_parameters13[8]
    FWHM14 = fitted_parameters14[8]
    mu13 = fitted_parameters13[3]
    mu14 = fitted_parameters14[3]
    bg13 = fitted_parameters13[4]
    bg14 = fitted_parameters14[4]

    print 'Area13: ',('%.2f'%area13),'+/-',('%.2f'%area13_er),'\n','Area14: ',('%.2f'%area14),'+/-',('%.2f'%area14_er),'\n','Fitted emission line ratio: ', ('%.2f'%ratio),'+/-',('%.2f'%ratio_error),'\n','sigma13: ',('%.2f'%sigma13),'+/-',('%.2f'%sig13er),'\n','gamma13: ',('%.2f'%gamma13),'+/-',('%.2f'%gam13er),'\n','sigma14: ',('%.2f'%sigma14),'+/-',('%.2f'%sig14er),'\n','gamma14: ',('%.2f'%gamma14),'+/-',('%.2f'%gam14er),'\n','Chi2_13',('%.2f'%fitted_parameters13[7]),'\n','Chi2_14',('%.2f'%fitted_parameters14[7])

    if plot:
        plt.figure()
        plt.clf()
        plt.title('Iris spectra Si IV 13.')
        plt.errorbar(x13,(spec1d13-bg13),Error13,ecolor='r',label='Area = %s\nError=%1.2f'%(area13,area13_er))
        plt.plot(x13,fit13,label='Voigt fit')
        plt.xlabel('Spectral index')
        plt.ylabel('No. Photons/second.')
        plt.legend(loc='upper left',prop={'size':10})
        filename=os.path.join(path,'2014_01_31_si13_%s%s_%s%s.png'%(x1,x2,y11,y12))
        plt.savefig(filename,bbox_inches='tight')
        
        plt.figure()
        plt.clf()
        plt.title('Iris spectra Si IV 14.')
        plt.xlabel('Spectral index')
        plt.ylabel('No. Photons/second.')
        plt.errorbar(x14,(spec1d14-bg14),Error14,ecolor='r',label='Area = %s\n Error=%1.2f'%(area14,area14_er))
        plt.plot(x14,fit14,label='Voigt fit')
        plt.legend(loc='upper left',prop={'size':10})
        filename=os.path.join(path,'2014_01_31_si14_%s%s_%s%s.png'%(x1,x2,y11,y12))
        pylab.savefig(filename,bbox_inches='tight')

    return [area13,area13_er,area14,area14_er,ratio, ratio_error]
    
def spectral_plot(x1,x2,y1,y2):
    # Displays spectral plot with errors
    # Displays Fitted parmaters and line information
    path0 = '/Users/krishnamooroogen/Documents/PHYSICS/MSSL_projectfiles/Exelfiles/v6/2014_01_30_21/spectral_lines'
    if not os.path.exists(path0):
        os.makedirs(path0)
    path1 = '/Users/krishnamooroogen/Documents/PHYSICS/MSSL_projectfiles/images/iris/v6/2014_01_30_21/spectral_lines'
    if not os.path.exists(path1):
        os.makedirs(path1)
    
    spec1d,xl = collapse(x1,x2,y1,y2,rasterfile='/Users/krishnamooroogen/Documents/PHYSICS/MSSL_projectfiles/instrument_data/iris/data_for_python/iris_l2_20140130_215650_si13.fits',line=1394,plot=True)
    
    EXPTIME =  file_info(file='/Users/krishnamooroogen/Documents/PHYSICS/MSSL_projectfiles/instrument_data/iris/rasterfiles/iris_l2_20140130_215650_3800256103_raster_t000_r00000.fits')

    spec1d, Error = Iris_prep(spec1d,header[0],line=1394,plot=False)
    
    peaks1, num_peaks1 = line_indent(spec1d,xl)

    fit0,fitted_parameters0,fitted_parameters_errs0=fit(spec1d,xl,0,peaks1, num_peaks1, gamma_est = 0, sigma_est = 1, bg_est =0, voigtian=False, lorentzian=False, gaussian=True, plot=False)

    xwl = setting_x_to_wavelength(spec1d,fitted_parameters0,Lambda=1394)

    peaks, num_peaks = line_indent(spec1d,xwl)

    fit1,fitted_parameters1,fitted_parameters_errs1=fit(spec1d,xwl,0,peaks, num_peaks, gamma_est = 0.01, sigma_est = 0.07, bg_est=0, voigtian=True, lorentzian=False, gaussian=False, plot=False)
    
    amp = fitted_parameters1[0]
    chi2 = fitted_parameters1[7]
    temp = fitted_parameters1[5]
    FWHM = fitted_parameters1[8]
    bg = fitted_parameters1[4]
    mu = fitted_parameters1[3]
    sigma = fitted_parameters1[1]
    gamma = fitted_parameters1[2]
    amp_er = fitted_parameters_errs1[0]
    mu_er = fitted_parameters_errs1[3]
    sigma_er = fitted_parameters_errs1[1]
    gamma_er = fitted_parameters_errs1[2]
    
    plt.figure()
    plt.clf()
    plt.title('Si IV IRIS spectra.')
    plt.xlabel('Wavelengths $\AA$.')
    plt.ylabel('No.Photons/second.')
    plt.errorbar(xwl,(spec1d-bg),Error,ecolor='r',label='Spectral data.')
    plt.plot(xwl,fit1,color= 'g',label='Voit fit: \n Chi2 %1.3f \n FWHM %1.3f \n Temp %1.3f K \n $\sigma$ %1.3f \n $\gamma$ %1.3f '%(chi2,FWHM,temp,sigma,gamma))
    label(peaks)
    plt.legend(loc='upper left',prop={'size':8})
    filename=os.path.join(path1,'2014_01_30_21_si13_%s%s_%s%s.png'%(x1,x2,y1,y2))
    pylab.savefig(filename,bbox_inches='tight')

    if num_peaks == 1:
        print 'Amplitude Si IV',('%.2f'%amp),'+/-',('%.2f'%amp_er),'\n','Mu Si IV',('%.3f'%mu),'+/-',('%.3f'%mu_er),'\n','Sigma Si IV: ',('%.3f'%sigma),'+/-',('%.3f'%sigma_er),'\n','Gamma Si IV: ',('%.3f'%gamma),'+/-',('%.3f'%gamma_er),'\n','Temperature Si IV: ',('%.2f'%temp),'K','\n','FWHM Si IV',('%.3f'%FWHM)
    
        df = DataFrame({'Amplitude Si IV':(amp,None),'Amplitude error Si IV':(amp_er,None),'Mu Si IV':(mu,None),'Mu Si IV error':(mu_er,None),'Sigma Si IV':(sigma,None),'Sigma Si IV error': (sigma_er,None),'Gamma Si IV':(gamma,None),'Gamma Si IV error':(gamma_er,None),'FWHM Si IV':(FWHM,None),'Chi2 Si IV':(chi2,None),'Temperature Si IV':(temp,None)})
    
        file = ('%s%s_%s%s_13_spectral_lines.xlsx'%(x1,x2,y1,y2))
        j=os.path.join(path0,file)
        df.to_excel(j, sheet_name='sheet1', index=False)

    if num_peaks == 2:
        
        amp2 = fitted_parameters1[9]
        temp2 = fitted_parameters1[12]
        FWHM2 = fitted_parameters1[13]
        mu2 = fitted_parameters1[11]
        sigma2 = fitted_parameters1[10]
        amp2_er = fitted_parameters_errs1[7]
        mu2_er = fitted_parameters_errs1[9]
        sigma2_er = fitted_parameters_errs1[8]
        

        print 'Amplitude Si IV',('%.2f'%amp),'+/-',('%.2f'%amp_er),'\n','Mu Si IV',('%.3f'%mu),'+/-',('%.3f'%mu_er),'\n','Sigma Si IV: ',('%.3f'%sigma),'+/-',('%.3f'%sigma_er),'\n','Gamma Si IV: ',('%.3f'%gamma),'+/-',('%.3f'%gamma_er),'\n','Temperature Si IV: ',('%.2f'%temp),'K','\n','FWHM Si IV',('%.3f'%FWHM),'\n', 'Amplitude 2nd line',('%.2f'%amp2),'+/-',('%.2f'%amp2_er),'\n','Mu 2nd line: ',('%.3f'%mu2),'+/-',('%.3f'%mu2_er),'\n','Sigma 2nd line: ',('%.3f'%sigma2),'+/-',('%.3f'%sigma2_er),'\n','Chi2: ',('%.2f'%chi2),'\n','Temperature 2nd line: ',('%.2f'%temp2),'K','\n','FWHM 2nd line: ',('%.3f'%FWHM2)

        df = DataFrame({'Amplitude Si IV':(amp,None),'Amplitude error Si IV':(amp_er,None),'Mu Si IV':(mu,None),'Mu Si IV error':(mu_er,None),'Sigma Si IV':(sigma,None),'Sigma Si IV error': (sigma_er,None),'Gamma Si IV':(gamma,None),'Gamma Si IV error':(gamma_er,None),'FWHM Si IV':(FWHM,None),'Chi2 Si IV':(chi2,None),'Temperature Si IV':(temp,None),'Amplitude 2nd line':(amp2,None),'Amplitude error 2nd line':(amp2_er,None),'Mu 2nd line':(mu2,None),'Mu 2nd line error':(mu_er,None),'Sigma 2nd line':(sigma2,None),'Sigma 2nd line error': (sigma2_er,None),'FWHM 2nd line':(FWHM2,None),'Temperature 2nd line':(temp2,None)})

        file = ('%s%s_%s%s_13_spectral_lines.xlsx'%(x1,x2,y1,y2))
        j=os.path.join(path0,file)
        df.to_excel(j, sheet_name='sheet1', index=False)

    if num_peaks == 3:

        amp2 = fitted_parameters1[9]
        temp2 = fitted_parameters1[12]
        FWHM2 = fitted_parameters1[13]
        mu2 = fitted_parameters1[11]
        sigma2 = fitted_parameters1[10]
        amp2_er = fitted_parameters_errs1[7]
        mu2_er = fitted_parameters_errs1[10]
        sigma2_er = fitted_parameters_errs1[8]
        
        
        amp3 = fitted_parameters1[14]
        temp3 = fitted_parameters1[17]
        FWHM3 = fitted_parameters1[18]
        mu3 = fitted_parameters1[16]
        sigma3 = fitted_parameters1[15]
        
        amp3_er = fitted_parameters_errs1[10]
        mu3_er = fitted_parameters_errs1[12]
        sigma3_er = fitted_parameters_errs1[11]
        

        
        print 'Amplitude Si IV',('%.2f'%amp),'+/-',('%.2f'%amp_er),'\n','Mu',('%.2f'%mu),'+/-',('%.2f'%mu_er),'\n','Sigma Si IV: ',('%.2f'%sigma),'+/-',('%.2f'%sigma_er),'\n','Gamma Si IV: ',('%.2f'%gamma),'+/-',('%.2f'%gamma_er),'\n','Temperature Si IV: ',temp,'K','\n','FWHM Si IV',FWHM,'\n', 'Amplitude 2nd line',amp2,'+/-',amp2_er,'\n','Mu 2nd line',mu2,'+/-',mu2_er,'\n','Sigma 2nd line: ',('%.2f'%sigma2),'+/-',('%.2f'%sigma2_er),'\n','Chi2: ',('%.2f'%chi2),'\n','Temperature 2nd line: ',('%.2f'%temp2),'\n','FWHM 2nd line: ',('%.2f'%FWHM2),'\n','Amplitude 3rd line',('%.2f'%amp3),'+/-',('%.2f'%amp3_er),'\n','Mu 3rd line',('%.2f'%mu3),'+/-',('%.2f'%mu3_er),'\n','Sigma 3rd line: ',('%.2f'%sigma3),'+/-',('%.2f'%sigma3_er),'\n','Temperature 3rd line: ',('%.2f'%temp3),'\n','FWHM 3rd line: ',('%.2f'%FWHM3)
        
        df = DataFrame({'Amplitude Si IV':(amp,None),'Amplitude error Si IV':(amp_er,None),'Mu Si IV':(mu,None),'Mu Si IV error':(mu_er,None),'Sigma Si IV':(sigma,None),'Sigma Si IV error': (sigma_er,None),'Gamma Si IV':(gamma,None),'Gamma Si IV error':(gamma_er,None),'FWHM Si IV':(FWHM,None),'Chi2 Si IV':(chi2,None),'Temperature Si IV':(temp,None),'Amplitude 2nd line':(amp2,None),'Amplitude error 2nd line':(amp2_er,None),'Mu 2nd line':(mu2,None),'Mu 2nd line error':(mu2_er,None),'Sigma 2nd line':(sigma2,None),'Sigma 2nd line error': (sigma2_er,None),'FWHM 2nd line':(FWHM2,None),'Temperature 2nd line':(temp2,None),'Amplitude 3rd line':(amp3,None),'Amplitude error 3rd line':(amp3_er,None),'Mu 3rd line':(mu3,None),'Mu 3rd line error':(mu3_er,None),'Sigma 3rd line':(sigma3,None),'Sigma 3rd line error': (sigma3_er,None),'FWHM 3rd line':(FWHM3,None),'Temperature 3rd line':(temp3,None)})
        
        file = ('%s%s_%s%s_13_spectral_lines.xlsx'%(x1,x2,y1,y2))
        j=os.path.join(path0,file)
        df.to_excel(j, sheet_name='sheet1', index=False)
                                                                                                                                                                                           
    if num_peaks == 4:
                                                                                                                                                                                        
        amp2 = fitted_parameters1[9]
        temp2 = fitted_parameters1[12]
        FWHM2 = fitted_parameters1[13]
        mu2 = fitted_parameters1[11]
        sigma2 = fitted_parameters1[10]                                                                                                                                                                                
        amp2_er = fitted_parameters_errs1[7]                                                                                                                                                                                   
        mu2_er = fitted_parameters_errs1[10]                                                                                                                                                                                   
        sigma2_er = fitted_parameters_errs1[8]                                                                                                                                                                                   
                                                                                                                                                                                           
        amp3 = fitted_parameters1[14]                                                                                                                                                                                   
        temp3 = fitted_parameters1[17]                                                                                                                                                                                   
        FWHM3 = fitted_parameters1[18]                                                                                                                                                                                   
        mu3 = fitted_parameters1[16]                                                                                                                                                                                   
        sigma3 = fitted_parameters1[15]                                                                                                                                                                                  
        amp3_er = fitted_parameters_errs1[10]                                                                                                                                                                                   
        mu3_er = fitted_parameters_errs1[12]
        sigma3_er = fitted_parameters_errs1[11]
                                                                                                                                                                                           
        amp4 = fitted_parameters1[19]
        temp4 = fitted_parameters1[22]
        FWHM4 = fitted_parameters1[23]
        mu4 = fitted_parameters1[21]
        sigma4 = fitted_parameters1[20]
        amp4_er = fitted_parameters_errs1[13]
        mu4_er = fitted_parameters_errs1[15]
        sigma4_er = fitted_parameters_errs1[14]
                                                                                                                                                                                           
        print 'Amplitude Si IV',('%.2f'%amp),'+/-',('%.2f'%amp_er),'\n','Mu',('%.2f'%mu),'+/-',('%.2f'%mu_er),'\n','Sigma Si IV: ',('%.2f'%sigma),'+/-',('%.2f'%sigma_er),'\n','Gamma Si IV: ',('%.2f'%gamma),'+/-',('%.2f'%gamma_er),'\n','Temperature Si IV: ',temp,'K','\n','FWHM Si IV',FWHM,'\n', 'Amplitude 2nd line',amp2,'+/-',amp2_er,'\n','Mu 2nd line',mu2,'+/-',mu2_er,'\n','Sigma 2nd line: ',('%.2f'%sigma2),'+/-',('%.2f'%sigma2_er),'\n','Chi2: ',('%.2f'%chi2),'\n','Temperature 2nd line: ',('%.2f'%temp2),'\n','FWHM 2nd line: ',('%.2f'%FWHM2),'Amplitude 3rd line',('%.2f'%amp3),'+/-',('%.2f'%amp3_er),'\n','Mu 3rd line',('%.2f'%mu3),'+/-',('%.2f'%mu3_er),'\n''Sigma 3rd line: ',('%.2f'%sigma3),'+/-',('%.2f'%sigma3_er),'\n','Temperature 3rd line: ',('%.2f'%temp3),'\n','FWHM 3rd line: ',('%.2f'%FWHM3),'Amplitude 4th line',('%.2f'%amp4),'+/-',('%.2f'%amp4_er),'\n','Mu 4th line',('%.2f'%mu4),'+/-',('%.2f'%mu4_er),'\n''Sigma 4th line: ',('%.2f'%sigma4),'+/-',('%.2f'%sigma4_er),'\n','Temperature 4th line: ',('%.2f'%temp4),'\n','FWHM 4th line: ',('%.2f'%FWHM4)
         
        df = DataFrame({'Amplitude Si IV':(amp,None),'Amplitude error Si IV':(amp_er,None),'Mu Si IV':(mu,None),'Mu Si IV error':(mu_er,None),'Sigma Si IV':(sigma,None),'Sigma Si IV error': (sigma_er,None),'Gamma Si IV':(gamma,None),'Gamma Si IV error':(gamma_er,None),'FWHM Si IV':(FWHM,None),'Chi2 Si IV':(chi2,None),'Temperature Si IV':(temp,None),'Amplitude 2nd line':(amp2,None),'Amplitude error 2nd line':(amp2_er,None),'Mu 2nd line':(mu2,None),'Mu 2nd line error':(mu2_er,None),'Sigma 2nd line':(sigma2,None),'Sigma 2nd line error': (sigma2_er,None),'FWHM 2nd line':(FWHM2,None),'Temperature 2nd line':(temp2,None),'Amplitude 3rd line':(amp3,None),'Amplitude error 3rd line':(amp3_er,None),'Mu 3rd line':(mu3,None),'Mu 3rd line error':(mu3_er,None),r'Sigma 3rd line':(sigma3,None),'Sigma 3rd line error': (sigma3_er,None),'FWHM 3rd line':(FWHM3,None),'Temperature 3rd line':(temp3,None),'Amplitude 4th line':(amp4,None),'Amplitude error 4th line':(amp4_er,None),'Mu 4th line':(mu4,None),'Mu 4th line error':(mu4_er,None),'Sigma 4th line':(sigma4,None),'Sigma 4th line error': (sigma4_er,None),'FWHM 4th line':(FWHM4,None),'Temperature 4th line':(temp4,None)})

        file = ('%s%s_%s%s_13_spectral_lines.xlsx'%(x1,x2,y1,y2))
        j=os.path.join(path0,file)
        df.to_excel(j, sheet_name='sheet1', index=False)
                                                                                                                                                                                           
    if num_peaks == 5:

        amp2 = fitted_parameters1[9]
        temp2 = fitted_parameters1[12]                                                                                                                                                                                       
        FWHM2 = fitted_parameters1[13]
        mu2 = fitted_parameters1[11]                                                                                                                                                                                   
        sigma2 = fitted_parameters1[10]                                                                                                                                                                                   
        amp2_er = fitted_parameters_errs1[7]                                                                                                                                                                                   
        mu2_er = fitted_parameters_errs1[10]                                                                                                                                                                                   
        sigma2_er = fitted_parameters_errs1[8]
                                                                                                                                                                                           
        amp3 = fitted_parameters1[14]                                                                                                                                                                                   
        temp3 = fitted_parameters1[17]                                                                                                                                                                                   
        FWHM3 = fitted_parameters1[18]                                                                                                                                                                                   
        mu3 = fitted_parameters1[16]                                                                                                                                                                                    
        sigma3 = fitted_parameters1[15]                                                                                                                                                                                                                                                                                                                                                                      
        amp3_er = fitted_parameters_errs1[10]                                                                                                                                                                                                                                                                                                                                                                     
        mu3_er = fitted_parameters_errs1[12]                                                                                                                                                                                                                                                                                                                                                                     
        sigma3_er = fitted_parameters_errs1[11]
                                                                                                                                                                                           
        amp4 = fitted_parameters1[19]                                                                                                                                                                                   
        temp4 = fitted_parameters1[22]                                                                                                                                                                                   
        FWHM4 = fitted_parameters1[23]                                                                                                                                                                                   
        mu4 = fitted_parameters1[21]                                                                                                                                                                                   
        sigma4 = fitted_parameters1[20]                                                                                                                                                                                   
        amp4_er = fitted_parameters_errs1[13]                                                                                                                                                                                   
        mu4_er = fitted_parameters_errs1[15]                                                                                                                                                                                   
        sigma4_er = fitted_parameters_errs1[14]
                                                                                                                                                                                           
        amp5 = fitted_parameters1[24]
        temp5 = fitted_parameters1[27]
        FWHM5 = fitted_parameters1[28]
        mu5 = fitted_parameters1[26]
        sigma5 = fitted_parameters1[25]
        amp5_er = fitted_parameters_errs1[16]
        mu5_er = fitted_parameters_errs1[17]
        sigma5_er = fitted_parameters_errs1[17]
                                                                                                                                                                                           
        print 'Amplitude Si IV',('%.2f'%amp),'+/-',('%.2f'%amp_er),'\n','Mu',('%.2f'%mu),'+/-',('%.2f'%mu_er),'\n','Sigma Si IV: ',('%.2f'%sigma),'+/-',('%.2f'%sigma_er),'\n','Gamma Si IV: ',('%.2f'%gamma),'+/-',('%.2f'%gamma_er),'\n','Temperature Si IV: ',temp,'K','\n','FWHM Si IV',FWHM,'\n', 'Amplitude 2nd line',amp2,'+/-',amp2_er,'\n','Mu 2nd line',mu2,'+/-',mu2_er,'\n','Sigma 2nd line: ',('%.2f'%sigma2),'+/-',('%.2f'%sigma2_er),'\n','Chi2: ',('%.2f'%chi2),'\n','Temperature 2nd line: ',('%.2f'%temp2),'\n','FWHM 2nd line: ',('%.2f'%FWHM2),'Amplitude 3rd line',('%.2f'%amp3),'+/-',('%.2f'%amp3_er),'\n','Mu 3rd line',('%.2f'%mu3),'+/-',('%.2f'%mu3_er),'\n''Sigma 3rd line: ',('%.2f'%sigma3),'+/-',('%.2f'%sigma3_er),'\n','Temperature 3rd line: ',('%.2f'%temp3),'\n','FWHM 3rd line: ',('%.2f'%FWHM3),'Amplitude 4th line',('%.2f'%amp4),'+/-',('%.2f'%amp4_er),'\n','Mu 4th line',('%.2f'%mu4),'+/-',('%.2f'%mu4_er),'\n''Sigma 4th line: ',('%.2f'%sigma4),'+/-',('%.2f'%sigma4_er),'\n','Temperature 4th line: ',('%.2f'%temp4),'\n','FWHM 4thd line: ',('%.2f'%FWHM4),'Amplitude 5th line',('%.2f'%amp5),'+/-',('%.2f'%amp5_er),'\n','Mu 5th line',('%.2f'%mu5),'+/-',('%.2f'%mu5_er),'\n''Sigma 5th line: ',('%.2f'%sigma5),'+/-',('%.2f'%sigma5_er),'\n','Temperature 5th line: ',('%.2f'%temp5),'\n','FWHM 5th line: ',('%.2f'%FWHM5)
                                                                                                                                                                                      
        df = DataFrame({'Amplitude Si IV':(amp,None),'Amplitude error Si IV':(amp_er,None),'Mu Si IV':(mu,None),'Mu Si IV error':(mu_er,None),'Sigma Si IV':(sigma,None),'Sigma Si IV error': (sigma_er,None),'Gamma Si IV':(gamma,None),'Gamma Si IV error':(gamma_er,None),'FWHM Si IV':(FWHM,None),'Chi2 Si IV':(chi2,None),'Temperature Si IV':(temp,None),'Amplitude 2nd line':(amp2,None),'Amplitude error 2nd line':(amp2_er,None),'Mu 2nd line':(mu2,None),'Mu 2nd line error':(mu2_er,None),'Sigma 2nd line':(sigma2,None),'Sigma 2nd line error': (sigma2_er,None),'FWHM 2nd line':(FWHM2,None),'Temperature 2nd line':(temp2,None),'Amplitude 3rd line':(amp3,None),'Amplitude error 3rd line':(amp3_er,None),'Mu 3rd line':(mu3,None),'Mu 3rd line error':(mu3_er,None),r'Sigma 3rd line':(sigma3,None),'Sigma 3rd line error': (sigma3_er,None),'FWHM 3rd line':(FWHM3,None),'Temperature 3rd line':(temp3,None),'Amplitude 4th line':(amp4,None),'Amplitude error 4th line':(amp4_er,None),'Mu 4rd line':(mu4,None),'Mu 4th line error':(mu4_er,None),'Sigma 4th line':(sigma4,None),'Sigma 4th line error': (sigma3_er,None),'FWHM 4th line':(FWHM4,None),'Temperature 4th line':(temp4,None),'Amplitude 5th line':(amp5,None),'Amplitude error 5th line':(amp5_er,None),'Mu 5th line':(mu5,None),'Mu 5th line error':(mu5_er,None),'Sigma 5th line':(sigma5,None),'Sigma 5th line error': (sigma5_er,None),'FWHM 5th line':(FWHM5,None),'Temperature 5th line':(temp5,None)})

        file = ('%s%s_%s%s_13_spectral_lines.xlsx'%(x1,x2,y1,y2))
        j=os.path.join(path0,file)
        df.to_excel(j, sheet_name='sheet1', index=False)

    if num_peaks == 6:

        amp2 = fitted_parameters1[9]
        temp2 = fitted_parameters1[12]
        FWHM2 = fitted_parameters1[13]
        mu2 = fitted_parameters1[11]
        sigma2 = fitted_parameters1[10]
        amp2_er = fitted_parameters_errs1[7]
        mu2_er = fitted_parameters_errs1[10]
        sigma2_er = fitted_parameters_errs1[8]
        
        amp3 = fitted_parameters1[14]
        temp3 = fitted_parameters1[17]
        FWHM3 = fitted_parameters1[18]
        mu3 = fitted_parameters1[16]
        sigma3 = fitted_parameters1[15]
        amp3_er = fitted_parameters_errs1[10]
        mu3_er = fitted_parameters_errs1[12]
        sigma3_er = fitted_parameters_errs1[11]
            
        amp4 = fitted_parameters1[19]
        temp4 = fitted_parameters1[22]
        FWHM4 = fitted_parameters1[23]
        mu4 = fitted_parameters1[21]
        sigma4 = fitted_parameters1[20]
        amp4_er = fitted_parameters_errs1[13]
        mu4_er = fitted_parameters_errs1[15]
        sigma4_er = fitted_parameters_errs1[14]
                
        amp5 = fitted_parameters1[24]
        temp5 = fitted_parameters1[27]
        FWHM5 = fitted_parameters1[28]
        mu5 = fitted_parameters1[26]
        sigma5 = fitted_parameters1[25]
        amp5_er = fitted_parameters_errs1[16]
        mu5_er = fitted_parameters_errs1[18]
        sigma5_er = fitted_parameters_errs1[17]
        
        amp6 = fitted_parameters1[29]
        temp6 = fitted_parameters1[32]
        FWHM6 = fitted_parameters1[33]
        mu6 = fitted_parameters1[31]
        sigma6 = fitted_parameters1[30]
        amp6_er = fitted_parameters_errs1[19]
        mu6_er = fitted_parameters_errs1[21]
        sigma6_er = fitted_parameters_errs1[20]
                
        print 'Amplitude Si IV',('%.2f'%amp),'+/-',('%.2f'%amp_er),'\n','Mu',('%.2f'%mu),'+/-',('%.2f'%mu_er),'\n','Sigma Si IV: ',('%.2f'%sigma),'+/-',('%.2f'%sigma_er),'\n','Gamma Si IV: ',('%.2f'%gamma),'+/-',('%.2f'%gamma_er),'\n','Temperature Si IV: ',temp,'K','\n','FWHM Si IV',FWHM,'\n', 'Amplitude 2nd line',amp2,'+/-',amp2_er,'\n','Mu 2nd line',mu2,'+/-',mu2_er,'\n','Sigma 2nd line: ',('%.2f'%sigma2),'+/-',('%.2f'%sigma2_er),'\n','Chi2: ',('%.2f'%chi2),'\n','Temperature 2nd line: ',('%.2f'%temp2),'\n','FWHM 2nd line: ',('%.2f'%FWHM2),'Amplitude 3rd line',('%.2f'%amp3),'+/-',('%.2f'%amp3_er),'\n','Mu 3rd line',('%.2f'%mu3),'+/-',('%.2f'%mu3_er),'\n''Sigma 3rd line: ',('%.2f'%sigma3),'+/-',('%.2f'%sigma3_er),'\n','Temperature 3rd line: ',('%.2f'%temp3),'\n','FWHM 3rd line: ',('%.2f'%FWHM3),'Amplitude 4th line',('%.2f'%amp4),'+/-',('%.2f'%amp4_er),'\n','Mu 4th line',('%.2f'%mu4),'+/-',('%.2f'%mu4_er),'\n''Sigma 4th line: ',('%.2f'%sigma4),'+/-',('%.2f'%sigma4_er),'\n','Temperature 4th line: ',('%.2f'%temp4),'\n','FWHM 4thd line: ',('%.2f'%FWHM4),'Amplitude 5th line',('%.2f'%amp5),'+/-',('%.2f'%amp5_er),'\n','Mu 5th line',('%.2f'%mu5),'+/-',('%.2f'%mu5_er),'\n''Sigma 5th line: ',('%.2f'%sigma5),'+/-',('%.2f'%sigma5_er),'\n','Temperature 5th line: ',('%.2f'%temp5),'\n','FWHM 5th line: ',('%.2f'%FWHM5),'Amplitude 6th line',('%.2f'%amp6),'+/-',('%.2f'%amp6_er),'\n','Mu 6th line',('%.2f'%mu6),'+/-',('%.2f'%mu6_er),'\n''Sigma 6th line: ',('%.2f'%sigma6),'+/-',('%.2f'%sigma6_er),'\n','Temperature 6th line: ',('%.2f'%temp6),'\n','FWHM 6th line: ',('%.2f'%FWHM6)
                
        df = DataFrame({'Amplitude Si IV':(amp,None),'Amplitude error Si IV':(amp_er,None),'Mu Si IV':(mu,None),'Mu Si IV error':(mu_er,None),'Sigma Si IV':(sigma,None),'Sigma Si IV error': (sigma_er,None),'Gamma Si IV':(gamma,None),'Gamma Si IV error':(gamma_er,None),'FWHM Si IV':(FWHM,None),'Chi2 Si IV':(chi2,None),'Temperature Si IV':(temp,None),'Amplitude 2nd line':(amp2,None),'Amplitude error 2nd line':(amp2_er,None),'Mu 2nd line':(mu2,None),'Mu 2nd line error':(mu2_er,None),'Sigma 2nd line':(sigma2,None),'Sigma 2nd line error': (sigma2_er,None),'FWHM 2nd line':(FWHM2,None),'Temperature 2nd line':(temp2,None),'Amplitude 3rd line':(amp3,None),'Amplitude error 3rd line':(amp3_er,None),'Mu 3rd line':(mu3,None),'Mu 3rd line error':(mu3_er,None),r'Sigma 3rd line':(sigma3,None),'Sigma 3rd line error': (sigma3_er,None),'FWHM 3rd line':(FWHM3,None),'Temperature 3rd line':(temp3,None),'Amplitude 4th line':(amp4,None),'Amplitude error 4th line':(amp4_er,None),'Mu 4rd line':(mu4,None),'Mu 4th line error':(mu4_er,None),'Sigma 4th line':(sigma4,None),'Sigma 4th line error': (sigma3_er,None),'FWHM 4th line':(FWHM4,None),'Temperature 4th line':(temp4,None),'Amplitude 5th line':(amp5,None),'Amplitude error 5th line':(amp5_er,None),'Mu 5th line':(mu5,None),'Mu 5th line error':(mu5_er,None),'Sigma 5th line':(sigma5,None),'Sigma 5th line error': (sigma5_er,None),'FWHM 5th line':(FWHM5,None),'Temperature 5th line':(temp5,None),'Amplitude 6th line':(amp6,None),'Amplitude error 6th line':(amp6_er,None),'Mu 6th line':(mu6,None),'Mu 6th line error':(mu6_er,None),'Sigma 6th line':(sigma6,None),'Sigma 6th line error': (sigma6_er,None),'FWHM 6th line':(FWHM6,None),'Temperature 6th line':(temp6,None)})

        file = ('%s%s_%s%s_13_spectral_lines.xlsx'%(x1,x2,y1,y2))
        j=os.path.join(path0,file)
        df.to_excel(j, sheet_name='sheet1', index=False)

    if num_peaks == 7:
    
        amp2 = fitted_parameters1[9]
        temp2 = fitted_parameters1[12]
        FWHM2 = fitted_parameters1[13]
        mu2 = fitted_parameters1[11]
        sigma2 = fitted_parameters1[10]
        amp2_er = fitted_parameters_errs1[7]
        mu2_er = fitted_parameters_errs1[10]
        sigma2_er = fitted_parameters_errs1[8]
    
        amp3 = fitted_parameters1[14]
        temp3 = fitted_parameters1[17]
        FWHM3 = fitted_parameters1[18]
        mu3 = fitted_parameters1[16]
        sigma3 = fitted_parameters1[15]
        amp3_er = fitted_parameters_errs1[10]
        mu3_er = fitted_parameters_errs1[12]
        sigma3_er = fitted_parameters_errs1[11]
    
        amp4 = fitted_parameters1[19]
        temp4 = fitted_parameters1[22]
        FWHM4 = fitted_parameters1[23]
        mu4 = fitted_parameters1[21]
        sigma4 = fitted_parameters1[20]
        amp4_er = fitted_parameters_errs1[13]
        mu4_er = fitted_parameters_errs1[15]
        sigma4_er = fitted_parameters_errs1[14]
    
        amp5 = fitted_parameters1[24]
        temp5 = fitted_parameters1[27]
        FWHM5 = fitted_parameters1[28]
        mu5 = fitted_parameters1[26]
        sigma5 = fitted_parameters1[25]
        amp5_er = fitted_parameters_errs1[16]
        mu5_er = fitted_parameters_errs1[18]
        sigma5_er = fitted_parameters_errs1[17]
    
        amp6 = fitted_parameters1[29]
        temp6 = fitted_parameters1[32]
        FWHM6 = fitted_parameters1[33]
        mu6 = fitted_parameters1[31]
        sigma6 = fitted_parameters1[30]
        amp6_er = fitted_parameters_errs1[19]
        mu6_er = fitted_parameters_errs1[21]
        sigma6_er = fitted_parameters_errs1[20]

        amp7 = fitted_parameters1[34]
        temp7 = fitted_parameters1[37]
        FWHM7 = fitted_parameters1[38]
        mu7 = fitted_parameters1[36]
        sigma7 = fitted_parameters1[35]
        amp7_er = fitted_parameters_errs1[22]
        mu7_er = fitted_parameters_errs1[24]
        sigma7_er = fitted_parameters_errs1[23]


        print 'Amplitude Si IV',('%.2f'%amp),'+/-',('%.2f'%amp_er),'\n','Mu',('%.2f'%mu),'+/-',('%.2f'%mu_er),'\n','Sigma Si IV: ',('%.2f'%sigma),'+/-',('%.2f'%sigma_er),'\n','Gamma Si IV: ',('%.2f'%gamma),'+/-',('%.2f'%gamma_er),'\n','Temperature Si IV: ',temp,'K','\n','FWHM Si IV',FWHM,'\n', 'Amplitude 2nd line',amp2,'+/-',amp2_er,'\n','Mu 2nd line',mu2,'+/-',mu2_er,'\n','Sigma 2nd line: ',('%.2f'%sigma2),'+/-',('%.2f'%sigma2_er),'\n','Chi2: ',('%.2f'%chi2),'\n','Temperature 2nd line: ',('%.2f'%temp2),'\n','FWHM 2nd line: ',('%.2f'%FWHM2),'Amplitude 3rd line',('%.2f'%amp3),'+/-',('%.2f'%amp3_er),'\n','Mu 3rd line',('%.2f'%mu3),'+/-',('%.2f'%mu3_er),'\n''Sigma 3rd line: ',('%.2f'%sigma3),'+/-',('%.2f'%sigma3_er),'\n','Temperature 3rd line: ',('%.2f'%temp3),'\n','FWHM 3rd line: ',('%.2f'%FWHM3),'Amplitude 4th line',('%.2f'%amp4),'+/-',('%.2f'%amp4_er),'\n','Mu 4th line',('%.2f'%mu4),'+/-',('%.2f'%mu4_er),'\n''Sigma 4th line: ',('%.2f'%sigma4),'+/-',('%.2f'%sigma4_er),'\n','Temperature 4th line: ',('%.2f'%temp4),'\n','FWHM 4thd line: ',('%.2f'%FWHM4),'Amplitude 5th line',('%.2f'%amp5),'+/-',('%.2f'%amp5_er),'\n','Mu 5th line',('%.2f'%mu5),'+/-',('%.2f'%mu5_er),'\n''Sigma 5th line: ',('%.2f'%sigma5),'+/-',('%.2f'%sigma5_er),'\n','Temperature 5th line: ',('%.2f'%temp5),'\n','FWHM 5th line: ',('%.2f'%FWHM5),'Amplitude 6th line',('%.2f'%amp6),'+/-',('%.2f'%amp6_er),'\n','Mu 6th line',('%.2f'%mu6),'+/-',('%.2f'%mu6_er),'\n''Sigma 6th line: ',('%.2f'%sigma6),'+/-',('%.2f'%sigma6_er),'\n','Temperature 6th line: ',('%.2f'%temp6),'\n','FWHM 6th line: ',('%.2f'%FWHM6),'Amplitude 7th line',('%.2f'%amp7),'+/-',('%.2f'%amp7_er),'\n','Mu 7th line',('%.2f'%mu7),'+/-',('%.2f'%mu7_er),'\n''Sigma 7th line: ',('%.2f'%sigma7),'+/-',('%.2f'%sigma7_er),'\n','Temperature 7th line: ',('%.2f'%temp7),'\n','FWHM 7th line: ',('%.2f'%FWHM7)
    
        df = DataFrame({'Amplitude Si IV':(amp,None),'Amplitude error Si IV':(amp_er,None),'Mu Si IV':(mu,None),'Mu Si IV error':(mu_er,None),'Sigma Si IV':(sigma,None),'Sigma Si IV error': (sigma_er,None),'Gamma Si IV':(gamma,None),'Gamma Si IV error':(gamma_er,None),'FWHM Si IV':(FWHM,None),'Chi2 Si IV':(chi2,None),'Temperature Si IV':(temp,None),'Amplitude 2nd line':(amp2,None),'Amplitude error 2nd line':(amp2_er,None),'Mu 2nd line':(mu2,None),'Mu 2nd line error':(mu2_er,None),'Sigma 2nd line':(sigma2,None),'Sigma 2nd line error': (sigma2_er,None),'FWHM 2nd line':(FWHM2,None),'Temperature 2nd line':(temp2,None),'Amplitude 3rd line':(amp3,None),'Amplitude error 3rd line':(amp3_er,None),'Mu 3rd line':(mu3,None),'Mu 3rd line error':(mu3_er,None),r'Sigma 3rd line':(sigma3,None),'Sigma 3rd line error': (sigma3_er,None),'FWHM 3rd line':(FWHM3,None),'Temperature 3rd line':(temp3,None),'Amplitude 4th line':(amp4,None),'Amplitude error 4th line':(amp4_er,None),'Mu 4rd line':(mu4,None),'Mu 4th line error':(mu4_er,None),'Sigma 4th line':(sigma4,None),'Sigma 4th line error': (sigma3_er,None),'FWHM 4th line':(FWHM4,None),'Temperature 4th line':(temp4,None),'Amplitude 5th line':(amp5,None),'Amplitude error 5th line':(amp5_er,None),'Mu 5th line':(mu5,None),'Mu 5th line error':(mu5_er,None),'Sigma 5th line':(sigma5,None),'Sigma 5th line error': (sigma5_er,None),'FWHM 5th line':(FWHM5,None),'Temperature 5th line':(temp5,None),'Amplitude 6th line':(amp6,None),'Amplitude error 6th line':(amp6_er,None),'Mu 6th line':(mu6,None),'Mu 6th line error':(mu6_er,None),'Sigma 6th line':(sigma6,None),'Sigma 6th line error': (sigma6_er,None),'FWHM 6th line':(FWHM6,None),'Temperature 6th line':(temp6,None),'Amplitude 7th line':(amp7,None),'Amplitude error 7th line':(amp7_er,None),'Mu 7th line':(mu7,None),'Mu 7th line error':(mu7_er,None),'Sigma 7th line':(sigma7,None),'Sigma 7th line error': (sigma7_er,None),'FWHM 7th line':(FWHM7,None),'Temperature 7th line':(temp7,None)})
    
        file = ('%s%s_%s%s_13_spectral_lines.xlsx'%(x1,x2,y1,y2))
        j=os.path.join(path0,file)
        df.to_excel(j, sheet_name='sheet1', index=False)

def draw_region(x1,x2,y1,y2) :
    
    rasterfile='/Users/krishnamooroogen/Documents/PHYSICS/MSSL_projectfiles/instrument_data/iris/data_for_python/iris_l2_20140131_084053_si13.fits'
    file1='/Users/krishnamooroogen/Documents/PHYSICS/MSSL_projectfiles/VSOdata/AIA/aia_lev1_211a_2014_01_31t08_40_59_62z_image_lev1_fits.fits'
    #draw subplot of specified regions of interest
    #either plot of disk alongside with ROI
    #or series of ROI
    #0252_424584
    xpixlen = pyfits.open(rasterfile)[0].data.shape[0]
    ypixlen = pyfits.open(rasterfile)[0].data.shape[1]
    print xpixlen,ypixlen
    header =  file_info(file='/Users/krishnamooroogen/Documents/PHYSICS/MSSL_projectfiles/instrument_data/iris/rasterfiles/iris_l2_20140131_084053_3803257203_raster_t000_r00000.fits')

    FOVX = header[1]
    FOVY = header[2]
    XCEN = header[3]
    YCEN = header[4]
    
    x_st = (XCEN-(FOVX/2))
    x_en = (XCEN+(FOVX/2))
    y_st = YCEN-(FOVY/2)
    y_en = YCEN+(FOVY/2)
    
    xpix_diff = x2-x1
    ypix_diff = y2-y1
    
    dadpx = FOVX/xpixlen
    dadpy = FOVY/ypixlen

    arcdiffx = dadpx*xpix_diff
    arcdiffy = dadpy*ypix_diff
    px_xst = dadpx*x1
    px_xen = dadpx*x2
    px_yst = dadpy*y1
    px_yen = dadpy*y2
    smap = sunpy.Map(file1)
        
    submap0 = smap.submap([x_st-10,x_en+10],[y_st,y_en]) #creates a submap between defined points(bottom left x,y)
    rect = patches.Rectangle([x_st, y_st], FOVX, FOVY, color = 'black', fill=False) #creates ROI sqaure
    gs = gridspec.GridSpec(2, 2,width_ratios=[1,1],height_ratios=[1,1]) #plot ratio

    fig = plt.figure(4,figsize = (10,10)) #opens new figure
    ax = plt.subplot(gs[0]) # plots in position 1
    smap.plot()
    plt.colorbar()
    smap.draw_limb()
    ax.add_artist(rect) #adds rect to plot
    ax.set_title('')

    ax1 = plt.subplot(gs[1]) #plots in position 0
    submap0.plot()
    plt.colorbar()
    #submap0.draw_grid(grid_spacing=10)#draw grid on image
    ax1.set_title('')
    
    submap1 = smap.submap([px_xst-1,px_xen+1],[px_yst,px_yen]) #creates a submap between defined points(bottom left x,y)
    rect1 = patches.Rectangle([x_st+px_xst, y_st+px_yst], arcdiffx*2, arcdiffy*1, color = 'black', fill=False) #creates ROI sqaure
    ax2 = plt.subplot(gs[2])
    submap0.plot()
    plt.colorbar()
    ax2.add_artist(rect1)
    ax2.set_title('')
    
    ax3 = plt.subplot(gs[3])
    submap1.plot()
    #submap1.draw_grid(grid_spacing=10)
    plt.colorbar()
    ax3.set_title('')
    
    
    plt.suptitle('FOV')
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(10)
        tick.label.set_rotation('vertical')
    
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(10)
        tick.label.set_rotation('vertical')

    for tick in ax3.xaxis.get_major_ticks():
        tick.label.set_fontsize(10)
        tick.label.set_rotation('vertical')

    fig.subplots_adjust(top=0.3, wspace=None, hspace=0.4)#space between top
    plt.tight_layout()
    #plt.savefig('submap_test.pdf', bbox_inches='tight', pad_inches = 0.0)
    plt.show()

def p_0(num,t) :
    #inhomogenous
    esc_p = np.sum((((-1)**n)*(np.exp(t)**n)*((math.factorial(n)*((n+1)**0.5))**-1)) for n in range(1,num,1))
    return esc_p+1

def p_1(num,t) :
    #homogeneous
    esc_p = np.sum((((np.exp(-t))**n)*((math.factorial(n+1)*(n+1)**0.5)**-1)) for n in range(0,num,1) )
    return esc_p

def tau_calc(ratio,ratio_error,x1,x2,y1,y2,homogenous=True):
    
    path = ('/Users/krishnamooroogen/Documents/PHYSICS/MSSL_projectfiles/images/iris/v7/2014_01_12_23/inhomogenous/ratio/%s%s_%s%s'%(x1,x2,y1,y2))
    
    # Calculate optical depth from esacape probablity contours
    # f_res,f_ref oscillator strengths
    # p, escape prob ratio
    # m_1/m_2 esc prob ivanov
    # p_1/p_0 esc prob kastner
    # t, optical depth
    # r, theoretical p ratio
    # working in natural logs 

    esc_prob=[]
    
    f_res = 0.26
    f_ref = 0.52
    l13 = 1393.76
    l14 = 1402.77

    k = (l13*f_ref)/(l14*f_res)
    p=ratio*(f_res/f_ref)
    p_er=p*(ratio_error/ratio)
    print 'p: ',('%.2f'%p),'+/-',('%.2f'%p_er)
    print 'k: ',('%.2f'%k)
    esc_prob.append(p)
    esc_prob.append(p_er)
    tau = np.arange(-2,3,0.1)
    p_erp = p+p_er
    p_erm = p-p_er
    
    x,y= np.meshgrid(tau,tau)
    
    k1 = tau+np.log(k)
    k2 = tau-np.log(k)
    
    
    if homogenous:
        r=np.log(p_1(51, y)/p_1(51, x))
    else:
        r=np.log(p_0(51, y)/p_0(51, x))
    
    plt.figure()
    CS0 = plt.contour(x, y, r,[p],colors='r')
    CS1 = plt.contour(x, y, r,[p_erp],colors='r')
    CS2 = plt.contour(x, y, r,[p_erm],colors='r')
    plt.close()
    pa = CS0.collections[0].get_paths()[0].vertices
    pa1 = CS1.collections[0].get_paths()[0].vertices
    pa2 = CS2.collections[0].get_paths()[0].vertices
    
    xx = pa[:, 0]
    yy = pa[:, 1]
    
    plt.figure()
    plt.title(r'Contour plot of R($\tau_1$,$\tau_2$) in ln($\tau$) plane')
    plt.xlabel(r'ln($\tau_1$)')
    plt.ylabel(r'ln($\tau_2$)')
    plt.plot(tau,k1,'b--',label=r'Theoretical R($\tau_1$,$\tau_2$) gradient const K')
    plt.plot(tau,k2,'b--')
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
    CS = plt.contour(x, y, r,40,colors='k')
    plt.plot(xx, yy, 'r-', label='Crossing contour %s'%(p))
    plt.clabel(CS, inline=0, fontsize=10)
    plt.legend(loc='best',prop={'size':8})
    
    filename=os.path.join(path,'2014_01_12_23_%s%s_contour.png'%(y1,y2))
    pylab.savefig(filename,bbox_inches='tight')
    plt.close()
    
    plt.figure()
    ax=plt.subplot(1,1,1)
    figure_title = r'Contour plot of R($\tau_1$,$\tau_2$)crossing with escape proability line in the ln($\tau$) plane.'
    plt.text(0.5, 1.08, figure_title,horizontalalignment='center',fontsize=12,transform = ax.transAxes)
    plt.xlabel(r'ln($\tau_1$)')
    plt.ylabel(r'ln($\tau_2$)')
    plt.plot(xx, yy, 'r-', label='Crossing contour %1.2f'%(p))
    plt.plot(tau,k2,'b--',label='Theory line K')
    plt.legend(loc='best',prop={'size':8})
    
    ls1 = LineString(pa)
    ls3 = LineString(pa1)
    ls4 = LineString(pa2)
    ls2 = LineString(np.c_[tau, k2])
    
    points = ls1.intersection(ls2)
    points_per = ls3.intersection(ls2)
    points_mer = ls4.intersection(ls2)
    
    try:
        if len(points)==0:
            print 'No intersection.'
            tau1 = None
            tau2 = None
            tau1_per = None
            tau2_per = None
            tau1_mer = None
            tau2_mer = None
            plt.close()
            return esc_prob,tau1,tau2,tau1_per,tau1_mer,tau2_per,tau2_mer
        
        else:
            x, y = points[0].x, points[0].y
            
            plt.plot(x,y, "ro")
                
                
            print'tau 1: ',('%.2f'%np.exp(x)), 'tau 2: ',('%.2f'%np.exp(y))
            tau1 = np.exp(x)
            tau2 = np.exp(y)
            
    
    except:
        x, y = points.x, points.y
        
        plt.plot(x,y, "ro")
        
        print'tau 1: ',('%.2f'%np.exp(x)), 'tau 2: ',('%.2f'%np.exp(y))
        tau1 = np.exp(x)
        tau2 = np.exp(y)
    
    try:
        if len(points_per)==0:
            print 'No positive error intersection.'
            tau1_per = None
            tau2_per = None 
                        
        else:
            xe, ye = points_per[0].x, points_per[0].y
            plt.plot(xe,ye, "gx")
                        
            print'tau positive error 1: ',('%.2f'%np.exp(xe)), 'tau positive error 2: ',('%.2f'%np.exp(ye))
            tau1_per = np.exp(xe)
            tau2_per = np.exp(ye)
    
    except:
        
        xe, ye = points_per.x, points_per.y
            
        plt.plot(xe,ye, "gx")
        
        print'tau positive error 1: ',('%.2f'%np.exp(xe)), 'tau positive error 2: ',('%.2f'%np.exp(ye))
        tau1_per = np.exp(xe)
        tau2_per = np.exp(ye)

    try:
        if len(points_mer)==0:
            print 'No negative intersection.'
            tau1_mer = None
            tau2_mer = None
            
        else:
            xe1, ye1 = points_mer[0].x, points_mer[0].y
        
        
            print'tau negative error 1: ',('%.2f'%np.exp(xe1)), 'tau negative error 2: ',('%.2f'%np.exp(ye1))
            tau1_mer = np.exp(xe1)
            tau2_mer = np.exp(ye1)
            

    except:
        
        xe1, ye1 = points_mer.x, points_mer.y
    
        plt.plot(xe1,ye1, "gx")
        print'tau negative error 1: ',('%.2f'%np.exp(xe1)), 'tau negative error 2: ',('%.2f'%np.exp(ye1))
        tau1_mer = np.exp(xe1)
        tau2_mer = np.exp(ye1)
        
    

    filename=os.path.join(path,'2014_01_12_23_%s%s.png'%(y1,y2))
    pylab.savefig(filename,bbox_inches='tight')
    plt.close()
    return esc_prob,tau1,tau2,tau1_per,tau1_mer,tau2_per,tau2_mer

def ratio_tau_out(x1,x2,y11,y12,y21,y22,n,run_tau=False) :
    path = '/Users/krishnamooroogen/Documents/PHYSICS/MSSL_projectfiles/Exelfiles/v7/2014_01_31/averaged_inhomo'
    if not os.path.exists(path):
        os.makedirs(path)
    
    area13,area13_er,area14,area14_er,ratio, ratio_error = fitted_ratio(x1,x2,y11,y12,y21,y22,n,plot=True)

    if run_tau:
        
        
        esc_prob,tau1,tau2,tau1_per,tau1_mer,tau2_per,tau2_mer = tau_calc(ratio,ratio_error,x1,x2,y11,y12,homogenous=False)
        df = DataFrame({'Area13': (area13,None), 'Area13 error': (area13_er,None),'Area14':(area14,None),'Area14 error':(area14_er,None),'Ratio':(ratio,None),'Ratio error':(ratio_error,None),'p':(esc_prob[0],None),'p_er':(esc_prob[1],None),'Tau1':(tau1,None),'Tau1 positive crossing point':(tau1_per,None),'Tau1 negative crossing point':(tau1_mer,None) ,'Tau2':(tau2,None),'Tau2 positive crossing point':(tau2_per,None),'Tau2 negative crossing point':(tau2_mer,None)})
    
        file = ('%s%s_%s%s_ratio.xlsx'%(x1,x2,y11,y12))
        j=os.path.join(path,file)
        df.to_excel(j, sheet_name='sheet1', index=False)
        
def run_priori(x1,x2,y1,y2):
    # Test function for pixel by pixel fitting 
    # Same as spectral plot but simpler 
    
    spec1d,x = collapse(x1,x2,y1,y2,rasterfile='/Users/krishnamooroogen/Documents/PHYSICS/MSSL_projectfiles/instrument_data/iris/data_for_python/iris_l2_20140130_202110_si13.fits',line=1394,plot=False)
    header = file_info(file='/Users/krishnamooroogen/Documents/PHYSICS/MSSL_projectfiles/instrument_data/iris/rasterfiles/iris_l2_20140130_202110_3800256103_raster_t000_r00000.fits')
    spec1d,Error = Iris_prep(spec1d,header[0],line=1394,plot=False)
    
    #peaks, num_peaks = line_indent(spec1d,x)
    peaks,num_peaks = peakdet(spec1d,x)

    
    gfit, fitted_parameters, fitted_parameters_errs= fit(spec1d, x,0,peaks,num_peaks,gamma_est = 0, sigma_est = 2, bg_est = 0, voigtian=True,lorentzian=False, gaussian=False, plot=False)

    bg = fitted_parameters[3]
    sigma = fitted_parameters[1]
    fit_area = fitted_parameters[6]
    area_er=fitted_parameters_errs[6]

    spec1d = spec1d - bg
    
    plt.figure()
    plt.errorbar(x,spec1d,Error,ecolor='r',label='Raw data.')
    plt.plot(gfit,label='Gaussian fit')
    plt.title('Gaussian fit over smoothed spectral data.')
    plt.ylabel('Photons/sec')
    plt.xlabel('Spectral index')
    plt.legend(loc='best')
    
    print 'fit area para',fit_area,'+/-', area_er
    print 'sigma',sigma,'+/-', fitted_parameters_errs[1]
        
def pixel_run(exptime,file = '/Users/krishnamooroogen/Documents/PHYSICS/MSSL_projectfiles/instrument_data/iris/data_for_python/iris_l2_20140110_015513_si13.fits',yst=19,yen=1069,ydelta=10,xdelta=12,line0=1394,plot=False):
    # Estimate for pixel by pixel ratio as squares
    # Loops of whole spectral image and caluclates area from fitted paramters
    # Sigma boundries may need changing 
    
    spectral_data = pyfits.open(file)[0].data
    print spectral_data.shape
    Area = []
    Area_er = []
    displace = np.arange(1.5,10,1)
    global speclen
    speclen = len(spectral_data)
    for xi in range(0,len(spectral_data),xdelta):
        for yi in range(yst,yen,ydelta):
            spec1d,xl = collapse(xi,(xi+xdelta),yi,(yi+ydelta),file,line=line0,plot=False)

            for displ in displace:
                try:
                    if line0==1394:
                        path = ('/Users/krishnamooroogen/Documents/PHYSICS/MSSL_projectfiles/images/iris/v7/2014_01_12_23/inhomogenous/ratio/%s%s_%s%s'%(xi,(xi+xdelta),yi,(yi+ydelta)))
                        if not os.path.exists(path):
                            os.makedirs(path)
                    if line0==1403:
                        path = ('/Users/krishnamooroogen/Documents/PHYSICS/MSSL_projectfiles/images/iris/v7/2014_01_12_23/inhomogenous/ratio/%s%s_%s%s'%(xi,(xi+xdelta),(yi+6),(yi+ydelta+6)))
                        if not os.path.exists(path):
                            os.makedirs(path)
                    check = 1
                    spectral_smdata_sum, error = Iris_prep(spec1d,exptime,line=line0,plot=False)
                    check = 1
                    peaks,num_peaks = peakdet(spectral_smdata_sum,xl,n=(displ),plot=False)
                    
                    fitted_spec,fit_param,fit_param_er = fit(spectral_smdata_sum, xl,0,peaks,num_peaks, gamma_est=0.01, sigma_est=0.07, bg_est=0, voigtian=True,lorentzian=False, gaussian=False, plot=False)
                    
                    fit_area = fit_param[6]
                    fit_area_er = fit_param_er[6]
                    bg = fit_param[4]
    
                    if fit_area<1:
                        check = 0
                        raise np.Exception
                    
                except:
                    if check==1:
                        #print 1
                        continue
                    if check==0:
                        #print 0
                        Area.append(1)
                        Area_er.append(1)
                        break
                    
                else:
                    #print 'fit'
                    Area.append(fit_area)
                    Area_er.append(fit_area_er)
                    if plot:
                        fig = plt.figure()
                        ax = plt.subplot(111)
                        ax.set_title('Iris spectra Si IV %s.'%(line0,))
                        ax.errorbar(xl,(spectral_smdata_sum-bg),error,ecolor='r',label='Area = %s\nError=%1.2f'%(fit_area,fit_area_er))
                        ax.plot(xl,fitted_spec,label='Voigt fit')
                        ax.set_xlabel('Spectral index')
                        ax.set_ylabel('No. Photons/second.')
                        handles, labels = ax.get_legend_handles_labels()
                        ax.legend(handles,labels,loc='upper left',prop={'size':10})
                        filename=os.path.join(path,'2014_01_12_23_si%s_%s%s_%s%s.png'%(line0,xi,(xi+xdelta),yi,(yi+ydelta)))
                        pylab.savefig(filename)
                        plt.close()

                    break
            else:
                #print 'ran out'
                Area.append(1)
                Area_er.append(1)
    
    return Area,Area_er

def ratio_map(xsh=8,ysh=105,y0st=0,y0en=0,y1st=0,y1en=0,yd=0,xd=0):
    #plt.gca().invert_yaxis()
    # Produces map of pixel by pixel ratios
    # Ratios of 0 are where the fit has over/under exgerated the area
    # Ratios of 1 could be where the fit has failed on both lines or high opacity, can be checked manually. More likely to a failiure of fit.
    # Files must be inputted at script 
    header =  file_info(file='/Users/krishnamooroogen/Documents/PHYSICS/MSSL_projectfiles/instrument_data/iris/rasterfiles/iris_l2_20140112_235123_3820257474_raster_t000_r00000.fits')

    
    Area13,Area13_er = pixel_run(header[0],file= '/Users/krishnamooroogen/Documents/PHYSICS/MSSL_projectfiles/instrument_data/iris/data_for_python/iris_l2_20140112_235123_si13.fits',yst=y0st,yen=y0en,ydelta=yd,xdelta=xd,line0=1394,plot=False)

    Area14,Area14_er = pixel_run(header[0],file= '/Users/krishnamooroogen/Documents/PHYSICS/MSSL_projectfiles/instrument_data/iris/data_for_python/iris_l2_20140112_235123_si14.fits',yst=y1st,yen=y1en,ydelta=yd,xdelta=xd,line0=1403,plot=False)
    
    area_13=np.array(Area13, dtype=np.float)
    area_14=np.array(Area14, dtype=np.float)
    
    area13_er=np.array(Area13_er, dtype=np.float)
    area14_er=np.array(Area14_er, dtype=np.float)

    
    ratio = area_13/area_14
    ratio_error = ratio*np.sqrt((area13_er/area_13)**2+(area14_er/area_14)**2)

    ratio2 = [x if 1<x<1.9 else 0 for x in ratio]
    
    data = np.array(ratio2).reshape(xsh,ysh)
    
    x=np.arange(y0st,y0en,yd)
    y=np.arange(0,(speclen+1),xd)
    
    plt.figure(figsize=(10,10))
    ax=plt.subplot(1,1,1)
    plt.pcolor(x,y,data, cmap='Greys', vmin=data.min(), vmax=data.max())
    figure_title = 'Ratio Map.'
    plt.text(-0.012, 0.6, figure_title,horizontalalignment='center',fontsize=15,transform = ax.transAxes,rotation=90)

    plt.xlabel('Pixels.',rotation=180)
    plt.ylabel('Pixels.')
    plt.xlim(y0st,y0en)
    plt.ylim(0,(speclen+1))
    ax.set_xticks(x)
    ax.set_yticks(y)
    
    for i in y:
       ax.axhline(i,color='g',ls='--')
    for i in x:
        ax.axvline(i,color='g',ls='--')

    
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(10)
        tick.label.set_rotation('vertical')

    ax.yaxis.set_label_position("right")    
    ax.yaxis.tick_right()
    ax.set_yticklabels(y, rotation=90,fontsize=10)

    cbar = plt.colorbar(orientation="horizontal")

    
    ratio_without0values = filter(lambda x: x != 0, ratio2)
            
    indicies = [index for index,i in enumerate(ratio2) if i==0]
    
    ratio_err_without0values = [i for j, i in enumerate(ratio_error) if j not in indicies]
            
    data_coords= []
    for xi in range(0,speclen,xd):
        for yi in x:
            data_coords.append(xi)
            data_coords.append((xi+xd))
            data_coords.append(yi)
            data_coords.append((yi+yd))
            
    
    coords = np.array(data_coords).reshape((len(data_coords)/4),4)
    
    
    co_ords = [i for j, i in enumerate(coords) if j not in indicies]
    print len(ratio_without0values),len(ratio_err_without0values),len(co_ords)
    return [ratio_without0values,ratio_err_without0values,co_ords]
    
def peakdet(spec1d,x,n=1.5,plot=False) :

    if x==None:
        x = np.arange(0,len(spec1d),1)

    if len(spec1d)!= len(x):
        sys.exit('Input vectors v and x must have same length')
            
    ft = scipy.ndimage.filters.gaussian_filter(spec1d,2)
    #ft = np.fft.fftshift(np.fft.fft(spec))
    #ft = np.diff(spec,n=1)
    
            
    #print'x', x[argrelextrema(ft, np.greater)[0]]
    #print 'y',spec1d[argrelextrema(spec1d, np.greater)[0]]

           
    x_pos = x[argrelextrema(ft, np.greater)[0]][ft[argrelextrema(ft, np.greater)[0]]>=n*stats.mode(ft[argrelextrema(ft, np.greater)[0]])[0]]
    
    if len(x_pos)==0:
        x_pos = x[argrelextrema(ft, np.greater)[0]]
    
    num_peaks = len(x_pos)
    peaks=[]
    
    for i in x_pos:
        peaks.append(spec1d[i])
        peaks.append(i)
        
    peaks = np.reshape(peaks,(len(x_pos),2))
    peaks = sorted(peaks,key=itemgetter(0),reverse=True)
        
    #print'peaks', peaks
    #print 'number of peaks found', num_peaks

    if plot:
        plt.figure()
        plt.plot(spec1d)
        for i in peaks:
            plt.axvline(i[1],color='r',ls='--')
            plt.axhline(i[0],color='r',ls='--')

    return peaks, num_peaks

def tau_auto() :
    
    path ='/Users/krishnamooroogen/Documents/PHYSICS/MSSL_projectfiles/Exelfiles/v7/2014_01_12_23/inhomogenous/ratio'
    if not os.path.exists(path):
        os.makedirs(path)
    
    values = ratio_map(xsh=4,ysh=21,y0st=8,y0en=365,y1st=2,y1en=354,yd=17,xd=2)

    #values = [[1.8],[0.5],[[0,26,27,50]]]
    for i in range(0,len(values[0])):
        print i 
        esc_prob,tau1,tau2,tau1_per,tau1_mer,tau2_per,tau2_mer = tau_calc(values[0][i],values[1][i],values[2][i][0],values[2][i][1],values[2][i][2],values[2][i][3],homogenous=False)
        print 'Ratio', values[0][i]
        print 'Ratio Error', values[1][i]
        df = DataFrame({'Ratio':(values[0][i],None),'Ratio error':(values[1][i],None),'p':(esc_prob[0],None),'p_er':(esc_prob[1],None),'Tau 1':(tau1,None),'Tau 1 positive crossing point':(tau1_per,None),'Tau 1 negative crossing point':(tau1_mer,None) ,'Tau 2':(tau2,None),'Tau 2 positive crossing point':(tau2_per,None),'Tau 2 negative crossing point':(tau2_mer,None)})
        
        file = ('%s%s_%s%s_ratio.xlsx'%((values[2][i][0]),values[2][i][1],(values[2][i][2]),values[2][i][3]))
        k=os.path.join(path,file)
        df.to_excel(k, sheet_name='sheet1', index=False)

    