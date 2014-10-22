from __future__ import division
import sunpy
import matplotlib.pyplot as plt
from matplotlib import patches 
import matplotlib.gridspec as gridspec
from sunpy.net import vso
from sunpy.time import *
import numpy 
import scipy
import pyfits
import itertools
import pylab
import math
from sunpy.cm import cm
import Image
import ImageEnhance
from scipy import signal
import minuit



#Script with various functions for manipulating and analysing of solar data using the sunpy module as a base
#might add a plot specific function, get code to work first 

def output_fits(input,filename):

    hdu = pyfits.PrimaryHDU(input)
    hdulist = pyfits.HDUList([hdu])
    hdulist.writeto(filename)
'''
def Plot_It(smap,*args,**kwargs,sunMap=False,image_sh=False,submap_plot=False,side_by_side_plot=False,):
    #optional arguments for submaps, need to set layout options
    #add tracking
    if sunMap:
        plt.figure(110, figsize = (10,10))
        smap.plot()
        plt.colorbar()
        smap.draw_limb()
        plt.show() 

    if image_sh:
        pylab.imshow(smap)

    if submap_plot:
        submap = smap.submap([x-shiftx1,-x+shiftx2],[y-shifty3,y+shifty4])
        rect = patches.Rectangle([-x, y], width, height, color = 'white', fill=False)
        gs = gridspec.GridSpec(2, 2,width_ratios=[1,1],height_ratios=[1,1])
        fig = plt.figure(4,figsize = (10,10)) 
        ax = plt.subplot(gs[1]) 
        smap.plot()
        plt.colorbar()
        smap.draw_limb()
        ax.add_artist(rect) 
        
        ax1 = plt.subplot(gs[0]) 
        submap.plot()
        plt.colorbar()
        submap.draw_grid(grid_spacing=10)
        ax1.set_title('Region of Interest')
        fig.subplots_adjust(hspace=0.5)
        plt.tight_layout()
        plt.show()

    if ROI_plot:
        submap = smap.submap([x-shiftx1,-x+shiftx2],[y-shifty3,y+shifty4])
        rect = patches.Rectangle([-x, y], width, height, color = 'white', fill=False)
        fig = plt.figure(2,figsize=(10,10))
        ax1 = plt.subplot(1,1,1)
        smap.plot()
        plt.colorbar()
        smap.draw_limb()
        ax1.add_artist(rect)
        
        fig2 = plt.figure(3,figsize = (10,10))
        ax2 = plt.subplot(1,1,1)
        submap.plot()
        plt.colorbar()
        submap.draw_grid(grid_spacing=10)
        ax2.set_title('ROI')
        plt.show()

    if side_by_side_plot:

        gs1 = gridspec.GridSpec(2, 2,width_ratios=[1,1],height_ratios=[1,1])
        fig = plt.figure(401,figsize = (10,10))
        ax3 = plt.subplot(gs1[1])
        smap.plot()
        smap.draw_limb()
        plt.show()
        
        ax2 = plt.subplot(gs1[0])
        pylab.imshow(smap2)
        ax2.set_title(title)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        fig.subplots_adjust(hspace=0.5)
        plt.tight_layout()
'''

def draw(file='/Users/krishnamooroogen/Documents/PHYSICS/MSSL_projectfiles/instrument_data/AIA/2011_06_07/aia_lev1_211a_2011_06_07t06_27_36_63z_image_lev1_fits.fits', sub_plot = False):
    #draw subplot of specified regions of interest
    #either plot of disk alongside with ROI
    #or series of ROI

    smap = sunpy.Map(file)
    if sub_plot:
            

        submap = smap.submap([-20-100,-20+600],[90-200,90+500]) #creates a submap between defined points(bottom left x,y)
        rect = patches.Rectangle([-20, 90], 500, 400, color = 'white', fill=False) #creates ROI sqaure
        gs = gridspec.GridSpec(2, 2,width_ratios=[1,1],height_ratios=[1,1]) #plot ratio
        fig = plt.figure(4,figsize = (10,10)) #opens new figure
        ax = plt.subplot(gs[1]) # plots in position 1
        smap.plot() 
        plt.colorbar()
        smap.draw_limb()
        ax.add_artist(rect) #adds rect to plot

        ax1 = plt.subplot(gs[0]) #plots in position 0
        submap.plot() 
        plt.colorbar()
        submap.draw_grid(grid_spacing=10)#draw grid on image
        ax1.set_title('Region of Interest') 
        fig.subplots_adjust(hspace=0.5)#space between top
        plt.tight_layout()  
        plt.savefig('submap_test.pdf', bbox_inches='tight', pad_inches = 0.0)
        plt.show() 

    else:
        submap = smap.submap([-20-100,-20+600],[90-500,90+500])
        rect = patches.Rectangle([-20, 90], 500, 400, color = 'white', fill=False)
        fig = plt.figure(2,figsize=(10,10))
        ax1 = plt.subplot(1,1,1)
        smap.plot()
        plt.colorbar()
        smap.draw_limb()
        ax1.add_artist(rect)
    
        fig2 = plt.figure(3,figsize = (10,10))
        ax2 = plt.subplot(1,1,1)
        submap.plot()
        plt.colorbar()
        submap.draw_grid(grid_spacing=10)
        ax2.set_title('ROI')

        plt.show()

def vso_q(st ='2014/04/26 01:00:36' ,end = '2014/04/26 01:00:50',inst = 'AIA', min_w = '211', max_w = '304'):
    #starts vso client
    #sends query to vso, can add extra search paramters
    #prints number of found records
    #show files found
    #download query results, wait function pauses the terminal until download is finshed 

    client = vso.VSOClient()
    qr = client.query(vso.attrs.Time(st,end), vso.attrs.Instrument(inst),vso.attrs.Wave(min_w,max_w))
    print 'Number of records', qr.num_records()
    qr.show()
    res    = client.get(qr, path='/Users/krishnamooroogen/Documents/physics/mssl_projectfiles/VSOdata/{instrument}/{file}.fits').wait()
    print 'Download complete'

'''     pixels : str
        number of pixels (numeric range, see below)
        resolution : str
        effective resolution (1 = full, 0.5 = 2x2 binned, etc)
        numeric range, see below.
        pscale : str
        pixel scale, in arcseconds (numeric range, see below)
        extent(FULLDISK,CORONA,LIMB) Source and or detector can be specified layout(spectra,time series, )
        #use the '|' key to join searches for example two instruments or two dates
        new_date  = date.replace('-','/')
        date_mod  = new_date.replace('T',' ')
        date_mod1 = date_mod[:-3]
    
'''

def rot_pos(date_start,date_end,x = -600,y=0):
    #calculates days of rotation using date strings
    #converts hpc coord to hg coord
    #applies Howard, Harvey, and Forgach model for differential rotation as used in IDL rot_xy
    #converts hg back into hpc
    #updates and returns coordinate position 
    
        days              = (sunpy.time.parse_time(date_end)-sunpy.time.parse_time(date_start)).days
        longd,latt        = sunpy.wcs.wcs.convert_hpc_hg(x,y)
        sin2l             = (math.sin(math.radians(float(longd))))**2
        sin4l             = sin2l*sin2l
        rott_longd        = (1e-6)*days*(2.894-0.428*sin2l-0.37*sin4l)*24*3600/0.0174532925
        nposx,nposy       = sunpy.wcs.wcs.convert_hg_hpc(rott_longd,latt)
        nposx             = nposx+x
        
        return nposx
        
def Tracking_ROI(solarfiles = 'solarfiles.txt', x=-700,y=0,ROI_plot=False):
    #track a coordinate through succesive days
    #create a file with file paths to the data that is to be tracked
    #opens file list
    #loops through file list opens each in turn and obtains datetime meta info
    #applies the rot function to the found datetimes
    #rot returns new coordinate
    #updated coordinates used to plot new ROI's i.e. tracking
    
    file_list = [i.strip().split() for i in open(solarfiles).readlines()]
    coordsROI =[]
    coordsROI.append(x)
    i=1
    j=2
    for index, line in enumerate(file_list):
        if index==len(file_list)-1:
            break
        smap          = sunpy.Map(line)
        smap2         = sunpy.Map(file_list[index+1])
        header        = smap.meta
        header2       = smap2.meta
        date_start    = header['date-obs']
        date_end      = header2['date-obs']

        nposx         = rot_pos(date_start,date_end,x,y)
        coordsROI.append(nposx)
        x             = nposx

    if ROI_plot:
        for line, x1 in itertools.izip(file_list,coordsROI):
            smap      = sunpy.Map(line)
            submap    = smap.submap([x1,x1+600],[y-300,y+400])
            fig       = plt.figure(101,figsize = (10,10))
            ax        = plt.subplot(len(file_list),2,i)
            submap.plot()
            plt.colorbar()
            #submap.draw_grid(grid_spacing=10)
            fig.subplots_adjust(hspace=0.5)
            plt.tight_layout()
            ax.set_title('ROI evolution')
            i+=1

    else:
        for line, x1 in itertools.izip(file_list,coordsROI):

            smap      = sunpy.Map(line)
            submap    = smap.submap([x1,x1+600],[y-300,y+400])
            rect      = patches.Rectangle([x1, y], 700, 500, color = 'white', fill=False)
            fig       = plt.figure(101,figsize = (10,10))
            ax        = plt.subplot(len(file_list),2,i)
            smap.plot()
            plt.colorbar()
            smap.draw_limb()
            ax.add_artist(rect)

            ax1         = plt.subplot(len(file_list),2,j)
            submap.plot()
            plt.colorbar()
            #submap.draw_grid(grid_spacing=10)
            plt.tight_layout()
            fig.subplots_adjust(hspace=0.5)
            ax1.set_title('ROI')
            j+=2
            i+=2

def Image_reduction_fft(f='/Users/krishnamooroogen/Documents/physics/MSSL_projectfiles/psf/psf_211_AIA.fits',d='/Users/krishnamooroogen/Documents/physics/MSSL_projectfiles/VSOdata/AIA/2011_06_07/aia_lev1_211a_2011_06_07t06_27_36_63z_image_lev1_fits.fits', PSF_plot=False, prior_post_plot=True):
    #Image deconvolution using FFT 
    #opens PSF(AIA instrument point spread function) and Data(AIA image data)
    #Plot options prior to deconvolution 
    #Applies FFT and divide in fouier space, convert back into real space, log10 is taken for plotting simplicity 
    #outputs deconvoluted image to fits file
    #plots deconvoluted image
    #colours neeed adjusting...

    cmap = cm.cmlist.get('sdoaia211')
    datamap = sunpy.Map(d)
    psf  = pyfits.open(f)[0].data
    data = pyfits.open(d)[0].data
    
    if PSF_plot:
        gs1 = gridspec.GridSpec(2, 2,width_ratios=[1,1],height_ratios=[1,1])
        fig = plt.figure(401,figsize = (10,10))
        ax3 = plt.subplot(gs1[1])
        datamap.plot()
        datamap.draw_limb()
        plt.show()

        ax2 = plt.subplot(gs1[0])
        pylab.imshow(numpy.log10(psf))
        ax2.set_title('PSF')
        ax2.set_xlabel('Pixel number')
        ax2.set_ylabel('Pixel number')
        fig.subplots_adjust(hspace=0.5)
        plt.tight_layout()
        plt.savefig('psf_and_solardisk.pdf', bbox_inches='tight', pad_inches = 0.0)
    

    #convert_realspace = numpy.log10(convert_realspace)
    deconvoluted_sun = deconvolute(data,psf)
    output_fits(deconvoluted_sun,'deconvoluted.fits')
    
    deconvoluted_sun = sunpy.Map('deconvoluted.fits')

    if prior_post_plot:
            gs = gridspec.GridSpec(2, 2,width_ratios=[1,1],height_ratios=[1,1])
            fig1 = plt.figure(400,figsize = (10,10))
            ax = plt.subplot(gs[1])
            deconvoluted_sun.plot(cmap=cmap)
            deconvoluted_sun.draw_limb()
            ax.set_title('After deconvolution, 2011/06/07 AIA')
            
            ax1 = plt.subplot(gs[0]) 
            datamap.plot(cmap=cmap)
            datamap.draw_limb()
            ax1.set_title('Before deconvolution, 2011/06/07 AIA')
            fig1.subplots_adjust(hspace=0.5)
            plt.tight_layout()
            plt.show()  
            plt.savefig('deconvolutionfft.pdf', bbox_inches='tight', pad_inches = 0.0)
      
    else:
       
        fig1 = plt.figure(405,figsize = (10,10))
        axs = plt.subplot(1,1,1)
        deconvoluted_sun.plot(cmap=cmap)
        deconvoluted_sun.draw_limb()
        axs.set_title('After deconvolution 211$\AA$, 2011/06/07 AIA')
        plt.show()

def Richardsonlucy_deconvolve_min(image_data='/Users/krishnamooroogen/Documents/physics/MSSL_projectfiles/VSOdata/AIA/2011_06_07/aia_lev1_211a_2011_06_07t06_27_36_63z_image_lev1_fits.fits' ,psf='/Users/krishnamooroogen/Documents/physics/MSSL_projectfiles/psf/psf_211_AIA.fits'):
    #opens image data and psf
    #creates inverse psf
    #sets lambda function (ricahrdson lucy)
    #sets start value
    #iterates gradient
       
        image_data = numpy.float64(pyfits.open(image_data)[0].data)
        psf = numpy.float64(pyfits.open(psf)[0].data)
        psf_hat = psf[::-1]
    
        #RL = lambda latent: numpy.array(numpy.float64(latent))*(convolute((numpy.array(numpy.float64(image_data))/(convolute(numpy.array(numpy.float64(latent)),numpy.array(numpy.float64(psf))))),numpy.array(numpy.float64(psf_hat))))
        RL = lambda latent: numpy.array(numpy.float64(latent))*(((numpy.array(numpy.float64(image_data))/(numpy.array(numpy.float64(latent))*numpy.array(numpy.float64(psf))).sum())*numpy.array(numpy.float64(psf))).sum())

        m = minuit.Minuit(RL,latent = (numpy.array([numpy.array([0]*len(image_data)),numpy.array([0]*len(image_data))]*int(len(image_data)/2))))
        m.tol = 1000000000
        m.printMode = 1
        m.migrad()
        m.hesse()
        latent = m.values['latent']

        print'fitLines> Covariance Matrix:\n', m.covariance
        
        #latent = numpy.log10(latent)
        output_fits(latent_image_est,'richard_lucy_minuit.fits')
    
        f = pyfits.open('richard_lucy_minuit.fits')[0].data
        pylab.imshow(f)
        deconvoluted_sun = sunpy.Map('richard_lucy.fits')
        fig1 = plt.figure(605,figsize = (10,10))
        axs = plt.subplot(1,1,1)
        deconvoluted_sun.plot(cmap=cmap)
        axs.set_title('After deconvolution 211$\AA$, 2011/06/07 AIA')
        plt.show()

def deconvolute(input,divider):

    fourier_space_division = (numpy.fft.rfft2(input,input.shape))/(numpy.fft.rfft2(divider,divider.shape))
    convert_realspace = numpy.fft.fftshift(numpy.fft.irfft2(fourier_space_division,fourier_space_division.shape))
    return convert_realspace

def convolute(input,input2):

    fourier_space_convolution = (numpy.fft.rfft2(input,input.shape))*(numpy.fft.rfft2(input2,input2.shape))
    convert_realspace_convolution = numpy.fft.fftshift(numpy.fft.irfft2(fourier_space_convolution,fourier_space_convolution.shape))
    return convert_realspace_convolution
    

def richardsonlucy(image_data='/Users/krishnamooroogen/Documents/physics/MSSL_projectfiles/VSOdata/AIA/2011_06_07/aia_lev1_211a_2011_06_07t06_27_36_63z_image_lev1_fits.fits' ,psf='/Users/krishnamooroogen/Documents/physics/MSSL_projectfiles/psf/psf_211_AIA.fits',niter=25):
    #Iterative Richardosn-Lucy Algorotihm for image deconvolution
    #opens image and psf data
    #create dummy start value for latent image
    #inverse the PSF
    #apply iteration
    #plot result
    #u^t+1=(u^t)*(d/(u^tXpsf)Xpsf^)
    
    cmap = cm.cmlist.get('sdoaia211')
    image_data = pyfits.open(image_data)[0].data
    psf = pyfits.open(psf)[0].data
    print 'Creating dummy array'
    latent_image_est = numpy.array([numpy.array([1.0]*len(image_data)),numpy.array([1.0]*len(image_data))]*int(len(image_data)/2))
    print 'Creating inverse psf'
    psf_hat = psf[::-1]
    print 'Starting Richardson-Lucy deconvolution'
    for i in range(niter):

        latent_image_est = latent_image_est*(convolute((image_data/convolute(latent_image_est,psf)),psf_hat))
        print 'Performing Richardson-Lucy Deconvolution, iteration number..',i

    print 'Deconvolution finished, begining plotting.'
    latent_image_est = numpy.log10(latent_image_est)
    output_fits(latent_image_est,'richard_lucy3.fits')

    deconvoluted_sun = sunpy.Map('richard_lucy3.fits')
    datamap = sunpy.Map('/Users/krishnamooroogen/Documents/physics/MSSL_projectfiles/VSOdata/AIA/2011_06_07/aia_lev1_211a_2011_06_07t06_27_36_63z_image_lev1_fits.fits')
    gs = gridspec.GridSpec(2, 2,width_ratios=[1,1],height_ratios=[1,1])
    fig1 = plt.figure(400,figsize = (10,10))
    ax = plt.subplot(gs[1])
    deconvoluted_sun.plot(cmap=cmap)
    deconvoluted_sun.draw_limb()
    ax.set_title('After deconvolution, 2011/06/07 AIA')
    
    ax1 = plt.subplot(gs[0])
    datamap.plot(cmap=cmap)
    datamap.draw_limb()
    ax1.set_title('Before deconvolution, 2011/06/07 AIA')
    fig1.subplots_adjust(hspace=0.5)
    plt.tight_layout()
    plt.show()
    plt.savefig('deconvolutionrl1.pdf', bbox_inches='tight', pad_inches = 0.0)


    '''
    f = pyfits.open('richard_lucy.fits')[0].data
    pylab.imshow(f)
    deconvoluted_sun = sunpy.Map('richard_lucy.fits')
    fig1 = plt.figure(605,figsize = (10,10))
    axs = plt.subplot(1,1,1)
    deconvoluted_sun.plot(cmap=cmap)
    axs.set_title('After deconvolution 211$\AA$, 2011/06/07 AIA')
    plt.show()
    '''        
            
    return latent_image_est

def out_txt(mylist,filename):
    with open(filename,'w') as file:
        for item in mylist:
            print>>file, item

def RUN():
    #drawsmap,sub_plot=True)
    #vso_q()
    #Tracking_ROI()
    #Image_reduction_fft()
    richardsonlucy()
    #Richardsonlucy_deconvolve_min()