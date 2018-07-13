# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 23:19:28 2017

@author: iwong
"""
from __future__ import division
import os
import sys

import numpy as np
import numpy.ma as ma
import astropy.io.fits as fits
import pdb
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
import scipy.optimize
from scipy.spatial import distance
import itertools
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib
from matplotlib.patches import Polygon
import matplotlib.colors as colors
import skimage.transform as sk
from astroquery.jplhorizons import Horizons
import urllib
import copy

import pickle

arcsectodeg = 1/60./60.

def checkpaths(direcs):
    '''
    Create necessary directories
    '''

    for direc in direcs:
        if not os.path.exists(direc):
            os.makedirs(direc)
        
def dist(point1,point2):
    '''
    Calculate distance between points, 
    where point1 and point2 are arrays of coordinates
    '''
    
    dist = np.sqrt((point1[:,0]-point2[:,0])**2+(point1[:,1]-point2[:,1])**2)
    return dist

def savepickle(data,filename):
    '''
    Save to pickle file
    '''

    f = open(filename,'w')
    pickle.dump(data,f)
    f.close()
    
def loadpickle(filename):
    '''
    Load from pickle file
    '''

    f = open(filename,'r')
    data = pickle.load(f)
    f.close()
    
    return data      

def makebiasflat(files,imtype='flat',bias=None):
    '''
    Create a median bias or flat image
    '''

    n = len(files)
    for i in range(n):
        hdulist = fits.open(files[i])
        if i == 0:
            frames = np.zeros((hdulist[0].header['NAXIS2'],hdulist[0].header['NAXIS1'],n))
        frames[:,:,i] = hdulist[0].data
    out = np.median(frames,axis=2)

    if imtype == 'flat':
        out = (out-bias)

    return out

def flatten(files,filters,flats,bias,flatdir,FILTlabel='FILTER'):
    '''
    Flatten images by filter
    '''

    n = len(files)
    for i in range(n):
        hdulist = fits.open(files[i])
        filt = hdulist[0].header[FILTlabel]
        biascorr = (hdulist[0].data-bias)
        w = np.where(filters == filt)
        flatimage = biascorr/flats[w][0]
        filename = files[i][files[i].rfind('/')+1:]
        hdu = fits.PrimaryHDU(flatimage,header=hdulist[0].header)
        hdu.writeto(flatdir+'f'+filename,clobber=True)

def invert_mask(file,output,axis=None,mask=None):
    '''
    Flip image along an axis and/or mask away the region [x1,x2,y1,y2]
    '''

    hdulist = fits.open(file)
    im = hdulist[0].data
    if axis != None:
        if axis == 'x':
            im = im[:,::-1]
        if axis == 'y':
            im = im[::-1,:]
        elif axis == 'xy':
            im = im[::-1,::-1]
    if mask != None:
        num  = len(mask)
        imx = np.isinf(im)
        for i in range(num):
            mask1 = mask[i]
            x1,x2,y1,y2 = mask1
            masked = im[y1:y2,x1:x2]    
            imx[y1:y2,x1:x2] = True
        nonmasked = ma.array(im, mask = imx, fill_value = float('NaN') )
        med = ma.median(nonmasked)
        pd_nonmasked = pd.DataFrame(nonmasked)
        M = len(pd_nonmasked.index)
        N = len(pd_nonmasked.columns)
        random = pd.DataFrame(np.random.normal(loc = med, scale = med*0.5), columns=pd_nonmasked.columns, index=pd_nonmasked.index)
        pd_nonmasked.update(random)
        np_final  = pd_nonmasked.as_matrix()
        hdulist[0].data = np_final
    hdu = fits.PrimaryHDU(im,header=hdulist[0].header)
    hdu.writeto(output,clobber=True)
                
def fit_intercept(x,b):
    return b+x

#==========================================================================

class image(object):
    '''
    A generic data object for a single astronomical image
    '''
    
    def __init__(self,filename,astrometrydir,stardir,sourcedir,flipxy=False,
                 RAlabel='RA',DEClabel='DEC',TIMElabel='JD',FILTlabel='FILTER',
                 OBJlabel='OBJECT',EXPTIMElabel='EXPTIME',
                 pointingpixel=None,pixelscale=None,tolerance=2):
            
        self.filename = filename
        self.shortname = self.filename[self.filename.rfind('/')+1:]
        self.astrometrydir = astrometrydir
        self.sourcedir = sourcedir
        self.stardir = stardir
        self.sourcefile = self.sourcedir+self.shortname+'.cat'
        self.starfile = self.stardir+self.shortname+'.star'
        self.astrofile = self.astrometrydir+self.shortname+'.ast'
        
        hdulist = fits.open(self.filename)
        header = hdulist[0].header
        self.header = header
        
        RA,DEC = header[RAlabel],header[DEClabel]
        coord = SkyCoord(RA+' '+DEC,unit=(u.hourangle,u.deg))
        self.RA_image = coord.ra.degree     #pointing RA,DEC in decimal degrees
        self.DEC_image = coord.dec.degree
        self.nx = header['NAXIS1']
        self.ny = header['NAXIS2']
        self.flipxy = flipxy
        self.tolerance = tolerance/100.      #tolerance (in percent)
        self.pxscale = pixelscale     #pixel scale in arcseconds
        self.time = header[TIMElabel]
        if TIMElabel[0:3] == 'MJD':
            self.time = float(self.time)+2400000.5
        self.exptime = header[EXPTIMElabel]
        self.object = header[OBJlabel]
        self.airmass = header['AIRMASS']
        self.filter = header[FILTlabel]
        if pointingpixel is None:
            self.pointx = int(self.nx/2)
            self.pointy = int(self.ny/2)
        else:
            self.pointx,self.pointy = pointingpixel[0],pointingpixel[1]
            
    def runSExtractor(self,sextractfile,alias='sex'):
        '''
        Run SEXtractor on the image using the given configuration file
        '''
        
        inp = self.filename
        cmd = alias+' '+inp+' -c '+sextractfile+' -catalog_name '+self.sourcefile
        os.system(cmd)
        
    def getstars(self):
        '''
        Query the PANSTARRS catalog for stars in the vicinity of the image
        '''
        
        #Find RA,DEC of image center and extent of image (long axis)
        if self.flipxy:
            RA = self.RA_image+(int(self.ny/2)-self.pointy)*self.pxscale*arcsectodeg
            DEC = self.DEC_image+(int(self.nx/2)-self.pointx)*self.pxscale*arcsectodeg
        else:
            RA = self.RA_image-(int(self.nx/2)-self.pointx)*self.pxscale*arcsectodeg
            DEC = self.DEC_image+(int(self.ny/2)-self.pointy)*self.pxscale*arcsectodeg
        RADIUS = 1.2*max(self.nx,self.ny)/2*self.pxscale*arcsectodeg      
              
        url = 'http://gsss.stsci.edu/webservices/vo/CatalogSearch.aspx?RA='+str(RA)+'&DEC='+str(DEC)+'&SR='+str(RADIUS)+'&FORMAT=CSV&CAT=PS1V3OBJECTS&MINDET=10&MAXOBJ=500'
        out = urllib.urlopen(url)
        f = open(self.starfile,'w')
        f.write(out.read())
        f.close()

    def getstars_manual(self,coords):
        '''
        Query the PANSTARRS catalog, given user-provided coordinates (sexagesimal)
        for the pointing location
        '''

        if coords == None:
            coords = [self.RA_image,self.DEC_image]
        coord = SkyCoord(coords[0]+' '+coords[1],unit=(u.hourangle,u.deg))
        
        #Find RA,DEC of image center and extent of image (long axis)
        if self.flipxy:
            RA = coord.ra.degree+(int(self.ny/2)-self.pointy)*self.pxscale*arcsectodeg
            DEC = coord.dec.degree+(int(self.nx/2)-self.pointx)*self.pxscale*arcsectodeg
        else:
            RA = coord.ra.degree-(int(self.nx/2)-self.pointx)*self.pxscale*arcsectodeg
            DEC = coord.dec.degree+(int(self.ny/2)-self.pointy)*self.pxscale*arcsectodeg
        RADIUS = 1.2*max(self.nx,self.ny)/2*self.pxscale*arcsectodeg

        url = 'http://gsss.stsci.edu/webservices/vo/CatalogSearch.aspx?RA='+str(RA)+'&DEC='+str(DEC)+'&SR='+str(RADIUS)+'&FORMAT=CSV&CAT=PS1V3OBJECTS&MINDET=10&MAXOBJ=500'
        out = urllib.urlopen(url)
        f = open(self.starfile,'w')
        f.write(out.read())
        f.close()
        
    def solveastro(self,order,plotting):
        '''
        Solve the astrometry, assuming maximum tolerated scale deviation
        '''
        
        #Create ordered source and star lists
        minfwhm = 0.5/self.pxscale
        maxfwhm = 5.0/self.pxscale
        self.sources = compilesources(self,minfwhm,maxfwhm)
        self.stars = compilestars(self,no_galaxies=False)
        pdb.set_trace()

        #Choose 200 3-source triplets that span most of the image in both x and y directions
        triplets = np.asarray(list(itertools.combinations(self.sources.index,3)))
        xorder = np.array([[np.where(self.sources.x.argsort() == i)[0][0] for i in j] for j in triplets])
        w = np.where(((np.max(triplets,axis=1)-np.min(triplets,axis=1))>len(self.sources.y)/2)
                     &((np.max(xorder,axis=1)-np.min(xorder,axis=1))>len(self.sources.x)/2))
        triplets = triplets[w]
        xorder = xorder[w]
        source_triplets = triplets[np.random.choice(np.arange(len(triplets)),200)]
        
        #Try to solve
        solved = False
        att = 0
        while not solved and att < len(source_triplets):
            #Look for matching star triplets
            result = self.findmatch(source_triplets[att],self.sources.dist,self.stars.dist)
            if result is False:
                att += 1
            else:
                print "Match found!"
                solved = True
                
                #Transform and save
                passed = self.transform(order)
                if not passed:
                    solved = False
                    att += 1
                else:
                    #Plot
                    if plotting:
                        self.plotsolution()
                #Save
                if solved:
                    sav = raw_input("Acceptable? (y/n)")
                    if sav == 'y':
                        savepickle(self,self.astrofile)
                    else:
                        solved = False
                        att += 1
        
        return solved

    def solveastro_manual(self,order,plotting,reffile):
        '''
        Solve the astrometry by clicking sources and stars that match
        '''
        
        self.starfile = reffile

        #Create ordered source and star lists
        minfwhm = 0.5/self.pxscale
        maxfwhm = 5.0/self.pxscale
        self.sources = compilesources(self,minfwhm,maxfwhm)
        self.stars = compilestars(self,no_galaxies=False)

        #Click on matching sources and stars and establish ballpark zeropoint
        click_sources(self)
        click_stars(self,self.starfile+'.fits')
        sourcex = self.match_sources[:,0]
        sourcey = self.match_sources[:,1]
        if self.flipxy:
            starx = self.match_stars[:,1]
            stary = self.match_stars[:,0]
        else:
            starx = self.match_stars[:,0]
            stary = self.match_stars[:,1]
        self.src = np.asarray(zip(sourcex,sourcey))
        self.dst = np.asarray(zip(starx,stary))
        self.zpguess = np.median(self.match_stars[:,2]+2.5*np.log10(self.match_sources[:,2]))

        #Transform
        done = self.transform(order)

        #Plot
        if plotting and done:
            self.plotsolution()

        #Save
        if done:
            sav = raw_input("Acceptable? (y/n)")
            if sav == 'y':
                savepickle(self,self.astrofile)
            
    def findmatch(self,triplet,sourcedist,stardist):
        '''
        Look for a matching star triplet given a source triplet
        '''
 
        a,b,c = triplet

        #Find first pair matches
        ww = np.where(abs(stardist-sourcedist[a,b])/sourcedist[a,b] < self.tolerance)
        if len(ww) < 1:
            return False
 
        #Look for other two pair matches
        for i in range(len(ww[0])):
            w1 = np.where(abs(stardist[ww[0][i],:]-sourcedist[a,c])/sourcedist[a,c] < self.tolerance)
            w2 = np.where(abs(stardist[ww[1][i],:]-sourcedist[b,c])/sourcedist[b,c] < self.tolerance)
            if len(w1) < 1 or len(w2) < 1 or len(np.intersect1d(w1,w2)) < 1:
                pass
            else:
                for j in np.intersect1d(w1,w2):
                    #Verify
                    candidate = np.array([ww[0][i],ww[1][i],j])
                    verify = self.verify(triplet,candidate)
                    if np.sum(verify) < 6:
                        pass
                    else:
                        return candidate
                
        return False
            
    def verify(self,sourcetrip,startrip):
        '''
        Verify relative x-positions of matched triplet,
        predicted pointing (within 20% of long axis),
        and estimated photometric zeropoints
        to prevent false positive matches
        '''
        
        #Check x-distance
        sourcex = self.sources.x[sourcetrip]
        sourcey = self.sources.y[sourcetrip]
        sourceflux = self.sources.flux[sourcetrip]
        if self.flipxy:
            starx = self.stars.dec[startrip]
            stary = self.stars.ra[startrip]
        else:
            starx = self.stars.ra[startrip]  
            stary = self.stars.dec[startrip]
        starmag = self.stars.mag[startrip]
        check1 = abs(abs((sourcex[1]-sourcex[0])*self.pxscale)-abs((starx[1]-starx[0])/arcsectodeg))/abs((sourcex[1]-sourcex[0])/self.pxscale) < self.tolerance
        check2 = abs(abs((sourcex[2]-sourcex[1])*self.pxscale)-abs((starx[2]-starx[1])/arcsectodeg))/abs((sourcex[2]-sourcex[1])/self.pxscale) < self.tolerance
        check3 = abs(abs((sourcex[2]-sourcex[0])*self.pxscale)-abs((starx[2]-starx[0])/arcsectodeg))/abs((sourcex[2]-sourcex[0])/self.pxscale) < self.tolerance
        
        #Check relative x-position
        source_order = sourcex.argsort()
        if self.flipxy:
            star_order = starx.argsort()
        else:
            star_order = (-starx).argsort()        
        check4 = np.sum(source_order == star_order) == 3
        if sum(np.array([check1,check2,check3,check4])) < 4:
            return np.array([check1,check2,check3,check4,False])
            
        #Check predicted pointing
        self.src = np.asarray(zip(sourcex,sourcey))
        self.dst = np.asarray(zip(starx,stary))
        trans = sk.estimate_transform('polynomial',self.dst,self.src,order=1)
        if self.flipxy:
            calc_point = trans.__call__(np.array([[self.DEC_image,self.RA_image]]))
        else:
            calc_point = trans.__call__(np.array([[self.RA_image,self.DEC_image]]))
        error_tol = 0.2*max(self.nx,self.ny)
        shift = dist(calc_point,np.array([[self.pointx,self.pointy]]))
        check5 = shift < error_tol

        #Check relative photometry
        zps = starmag+2.5*np.log10(sourceflux)
        self.zpguess = np.median(zps)
        check6 = (max(zps)-min(zps)) < 4

        return np.array([check1,check2,check3,check4,check5,check6])
        
    def transform(self,order):
        '''
        Calculate transformation from X,Y to RA,DEC starting from matched triplet
        '''
        
        #Initial linear transformation based on initial matches
        self.rlsrc = self.src
        initialtrans = sk.estimate_transform('polynomial',self.src,self.dst,order=1)
        
        #Look for more matches (based on both distance and brightness)
        sourcecoords = np.asarray(zip(self.sources.x,self.sources.y))
        if self.flipxy:
            starcoords = np.asarray(zip(self.stars.dec,self.stars.ra))
        else:
            starcoords = np.asarray(zip(self.stars.ra,self.stars.dec))
        calc_points = initialtrans.__call__(sourcecoords)
        sourceidx,staridx = [],[]
        for i in range(len(starcoords)):
            dev = abs(-2.5*np.log10(self.sources.flux)+self.zpguess-self.stars.mag[i])
            w1 = np.where((dist(np.array([starcoords[i,:]]),calc_points)/arcsectodeg < 3) & (dev < 3))
            if len(w1[0]) >= 1:
                dev = abs(-2.5*np.log10(self.sources.flux[w1])+self.zpguess-self.stars.mag[i])
                w2 = np.where(dev == min(dev))
                if w1[0][w2[0][0]] not in sourceidx:
                    sourceidx.append(w1[0][w2[0][0]])
                    staridx.append(i)
            
        if len(sourceidx)<5:
            print "Few matching stars. Trying to re-solve astrometry..."
            return False
        else:
            #Create final transformation
            self.matchidx = sourceidx
            self.src = np.asarray(zip(self.sources.x[sourceidx],self.sources.y[sourceidx]))
            if self.flipxy:
                self.dst = np.asarray(zip(self.stars.dec[staridx],self.stars.ra[staridx]))
            else:
                self.dst = np.asarray(zip(self.stars.ra[staridx],self.stars.dec[staridx]))
            self.trans = sk.estimate_transform('polynomial',self.src,self.dst,order=order)

            #Compute median error in position and shift from predicted pointing
            est = self.trans.__call__(self.src)
            self.error = np.median(dist(est,self.dst)/arcsectodeg)
            if self.error > 2:
                return False
            calc_point = self.trans.__call__(np.array([[self.pointx,self.pointy]]))
            if self.flipxy:
                self.shift = dist(calc_point,np.array([[self.DEC_image,self.RA_image]]))/arcsectodeg/self.pxscale
            else:
                self.shift = dist(calc_point,np.array([[self.RA_image,self.DEC_image]]))/arcsectodeg/self.pxscale
            if self.shift > 0.2*max(self.nx,self.ny):
                return False
            print "Matches = "+str(len(self.src))+"   Error = "+str(round(self.error,3))+" arcsec"+"    Shift = "+str(round(self.shift,1))+" pixels"

            return True          
        
    def plotsolution(self):
        '''
        Comparison plot of sources and stars with matched triplet marked
        '''
        
        hdulist = fits.open(self.filename)
        flux = hdulist[0].data
        ima = copy.deepcopy(flux)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        flat = np.ndarray.flatten(ima)
        med,std = np.median(flat),np.std(flat)
        ima[np.where((ima-med)>3*std)] = med+3*std
        ima[np.where((med-ima)>3*std)] = med-3*std

        plt.triplot(self.rlsrc[:,0], self.rlsrc[:,1])
        ax.imshow(ima,norm=colors.LogNorm())
        ax.scatter(self.sources.x,self.sources.y,s=80,facecolors='none',edgecolors='black')
        ax.scatter(self.sources.x[self.matchidx],self.sources.y[self.matchidx],s=50,facecolors='none',edgecolors='blue')
        ax.set_xlim(-1,ima.shape[1])
        ax.set_ylim(-1,ima.shape[0])
        ax.set_xlabel('x [px]')
        ax.set_ylabel('y [px]')
        plt.show()
        
#==========================================================================
            
        
class sources(object):
    '''
    Object containing 2D array of the pairwise distances between the 
    brightest max. [numb] sources on the image (in arcseconds)
    with increasing y position
    '''
    
    def __init__(self,table,pxscale,numb=30):
        table = table[table[:,2].argsort()[::-1]]
        n = min(len(table),numb)
        x = table[:n,0]
        y = table[:n,1]
        flux = table[:n,2]
        fluxerr = table[:n,3]
        fwhm = table[:n,4]
        self.y = y[y.argsort()]
        self.x = x[y.argsort()]
        self.flux = flux[y.argsort()]
        self.fluxerr = fluxerr[y.argsort()]
        self.fwhm = fwhm[y.argsort()]
        self.index = np.arange(n)
        coords = zip(self.x,self.y)
        dist = distance.cdist(coords,coords,'euclidean')*pxscale
        dist[np.arange(dist.shape[0])[:,None] >= np.arange(dist.shape[1])] = np.nan
        self.dist = dist
 
#==========================================================================
        
class stars(object):
    '''
    Object containing 2D array of the pairwise distances between the 
    brightest max. [numb] stars in the query catalog (in arcseconds)
    with increasing DEC or RA position
    '''
    
    def __init__(self,table,flipxy,numb=100):
        table = table[table[:,2].argsort()]
        n = min(len(table),numb)
        ra = table[:n,0]
        dec = table[:n,1]
        mag = table[:n,2]
        gmag = table[:n,3]
        gmagerr = table[:n,4]
        rmag = table[:n,5]
        rmagerr = table[:n,6]
        imag = table[:n,7]
        imagerr = table[:n,8]
        zmag = table[:n,9]
        zmagerr = table[:n,10]
        if flipxy:
            sorting = ra.argsort()
        else:
            sorting = dec.argsort()
        self.ra = ra[sorting]
        self.dec = dec[sorting]
        self.mag = mag[sorting]
        self.gmag = gmag[sorting]
        self.gmagerr = gmagerr[sorting]
        self.rmag = rmag[sorting]
        self.rmagerr = rmagerr[sorting]
        self.imag = imag[sorting]
        self.imagerr = imagerr[sorting]
        self.zmag = zmag[sorting]
        self.zmagerr = zmagerr[sorting]
        self.index = np.arange(n)
        coords = zip(self.ra,self.dec)
        dist = distance.cdist(coords,coords,'euclidean')*60*60    
        dist[np.arange(dist.shape[0])[:,None] >= np.arange(dist.shape[1])] = np.nan
        self.dist = dist

#==========================================================================

def compilesources(im,minfwhm,maxfwhm,numb=30):
    '''
    Create list of sources that are likely astrophysical
    '''
    
    data = np.genfromtxt(im.sourcefile)
    x = data[:,1]
    y = data[:,2]
    flux = data[:,3]
    fluxerr = data[:,4]
    fwhm = data[:,7]
    flag = data[:,5].astype('int')
    
    w = np.where((flux > 0)&(fwhm > minfwhm)&(fwhm < maxfwhm)&((flag == 0)|(flag == 2)))
    return sources(np.column_stack([x[w],y[w],flux[w],fluxerr[w],fwhm[w]]),im.pxscale,numb=numb)
    
def compilestars(im,no_galaxies=True,numb=100):
    '''
    Create list of catalog stars for astrometry or photometry
    '''
    
    data = np.genfromtxt(im.starfile,delimiter=",",skip_header=2)
    ra = data[:,11]
    dec = data[:,12]
    mag = data[:,-2]
    gmag = data[:,25]
    gmagerr = data[:,26]
    rmag = data[:,31]
    rmagerr = data[:,32]
    imag = data[:,37]
    imagerr = data[:,38]
    zmag = data[:,43]
    zmagerr = data[:,44]
    ikronmag = data[:,83]
    flag = data[:,10]
    if no_galaxies:
        w = np.where((mag > 0) & (gmag > 0) & (imag > 0) & (rmag > 0) & (zmag > 0) & (imag-ikronmag < 0.05))
    else:
        w = np.where((mag != None))      
    return stars(np.column_stack([ra[w],dec[w],mag[w],gmag[w],gmagerr[w],
                                  rmag[w],rmagerr[w],imag[w],imagerr[w],
                                  zmag[w],zmagerr[w]]),im.flipxy,numb=numb)

def click_sources(im,target=False):
    '''
    Click sources to establish photometric references
    '''
    
    sources = im.sources
    hdulist = fits.open(im.filename)
    flux = hdulist[0].data

    def onclick_source(event):
        dst = dist(np.array([[event.xdata, event.ydata]]),np.asarray(zip(sources.x,sources.y)))
        w = np.where(dst<25)[0]
        if len(w) == 0:
            print "Carefully click within 25 pixels of desired source!"
        else:
            print "x = "+str(round(sources.x[w[0]],2))+", y = "+str(round(sources.y[w[0]],2))
            im.match_sources.append([sources.x[w[0]],sources.y[w[0]],sources.flux[w[0]],sources.fluxerr[w[0]],sources.fwhm[w[0]]])

    ima = copy.deepcopy(flux)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    flat = np.ndarray.flatten(ima)
    med,std = np.median(flat),np.std(flat)
    ima[np.where((ima-med)>3*std)] = med+3*std
    ima[np.where((med-ima)>3*std)] = med-3*std
    ax.imshow(ima,norm=colors.LogNorm())
    ax.scatter(sources.x,sources.y,s=80,facecolors='none',edgecolors='black')
    ax.set_xlim(-1,ima.shape[1])
    ax.set_ylim(-1,ima.shape[0])
    ax.set_xlabel('x [px]')
    ax.set_ylabel('y [px]')

    im.match_sources = []
    cid = fig.canvas.mpl_connect('button_press_event',onclick_source)
    plt.show()

    if target:
        im.x,im.y,im.flux,im.fluxerr,im.fwhm = im.match_sources[0]
        im.mag = -2.5*np.log10(im.flux)+im.zp
        im.magerr = im.zperr

    im.match_sources = np.asarray(im.match_sources)

def click_stars(im,refimage):
    '''
    Click stars that match the selected sources
    '''
    
    stars = im.stars

    def onclick_star(event):
        radec = np.asarray([im.wcs.wcs_pix2world(event.xdata,event.ydata,1)])
        dst = dist(radec,np.asarray(zip(stars.ra,stars.dec)))
        w = np.where(dst<4*arcsectodeg)[0]
        if len(w) == 0:
            print "Carefully click within 4 arcsec of desired source!"
        else:
            print "ra = "+str(round(stars.ra[w[0]],3))+", dec = "+str(round(stars.dec[w[0]],3))
            im.match_stars.append([stars.ra[w[0]],stars.dec[w[0]],stars.mag[w[0]],
                                     stars.gmag[w[0]],stars.gmagerr[w[0]],
                                     stars.rmag[w[0]],stars.rmagerr[w[0]],
                                     stars.imag[w[0]],stars.imagerr[w[0]],
                                     stars.zmag[w[0]],stars.zmagerr[w[0]]])

    hdulist = fits.open(refimage)
    wcs = WCS(hdulist[0].header)
    im.wcs = wcs
    ima = hdulist[0].data

    fig = plt.figure()
    ax = fig.add_subplot(111,projection=wcs)
    flat = np.ndarray.flatten(ima)
    med,std = np.median(flat),np.std(flat)
    ima[np.where((ima-med)>3*std)] = 3*std
    ax.imshow(ima,origin='lower',norm=colors.LogNorm())
    lon,lat = ax.coords
    lon.set_major_formatter('d.ddd')
    lat.set_major_formatter('d.ddd')
    xx,yy = wcs.wcs_world2pix(stars.ra,stars.dec,1)
    ax.scatter(xx,yy,s=80,facecolors='none',edgecolors='blue')
    ax.set_xlabel('ra [deg]')
    ax.set_ylabel('dec [deg]')

    im.match_stars = []
    cid = fig.canvas.mpl_connect('button_press_event',onclick_star)
    plt.show()

    im.match_stars = np.asarray(im.match_stars)

#==========================================================================

class photometry(object):
    '''
    A generic data object for photometric calibration
    '''
    
    def __init__(self,filename,stardir,sourcedir,astrometrydir,photometrydir,flipxy=False,
                 FILTlabel='FILTER',TIMElabel='JD',OBJlabel='OBJECT',starfile=None):
            
        self.filename = filename
        self.shortname = self.filename[self.filename.rfind('/')+1:]
        self.sourcedir = sourcedir
        self.stardir = stardir
        self.sourcefile = self.sourcedir+self.shortname+'.cat'
        self.photometrydir = photometrydir
        self.astrometrydir = astrometrydir
        astro = loadpickle(self.astrometrydir+self.shortname+'.ast')
        self.error = astro.error
        self.zpguess = astro.zpguess
        self.pxscale = astro.pxscale
        self.trans = astro.trans
        if starfile != None:
            self.starfile = starfile
        else:
            self.starfile = self.stardir+self.shortname+'.star' 
        
        hdulist = fits.open(self.filename)
        header = hdulist[0].header
        self.header = header
        self.time = header[TIMElabel]
        self.object = header[OBJlabel]
        if TIMElabel[0:3] == 'MJD':
            self.time = float(self.time)+2400000.5
        self.flipxy = flipxy
        self.filter = header[FILTlabel]

        self.sources = compilesources(self,0.5/self.pxscale,numb=1000)
        self.stars = compilestars(self,no_galaxies=True,numb=1000)

    def transform(self):
        '''
        Apply astrometric transformation and find source-star calibration pairs
        '''

        sources = self.sources
        stars = self.stars

        #Search for matches within 20x median positional error
        sourcecoords = np.asarray(zip(sources.x,sources.y))
        if self.flipxy:
            starcoords = np.asarray(zip(stars.dec,stars.ra))
        else:
            starcoords = np.asarray(zip(stars.ra,stars.dec))
        self.calc_points = self.trans.__call__(sourcecoords)
        self.match_sources,self.match_stars = [],[]
        for i in range(len(starcoords)):
            dev = abs(-2.5*np.log10(sources.flux)+self.zpguess-stars.mag[i])
            w1 = np.where((dist(np.array([starcoords[i,:]]),self.calc_points)/arcsectodeg < 20*self.error) & (dev < 3))
            if len(w1[0]) >= 1:
                dev = abs(-2.5*np.log10(sources.flux[w1])+self.zpguess-stars.mag[i])
                w2 = np.where(dev == min(dev))
                self.match_sources.append([sources.x[w1[0][w2[0][0]]],sources.y[w1[0][w2[0][0]]],
                                           sources.flux[w1[0][w2[0][0]]],sources.fluxerr[w1[0][w2[0][0]]],
                                           sources.fwhm[w1[0][w2[0][0]]]])
                self.match_stars.append([stars.ra[i],stars.dec[i],stars.mag[i],
                                     stars.gmag[i],stars.gmagerr[i],
                                     stars.rmag[i],stars.rmagerr[i],
                                     stars.imag[i],stars.imagerr[i],
                                     stars.zmag[i],stars.zmagerr[i]])
    
    def matching(self):
        '''
        Create array of sources and corresponding catalog stars
        '''

        n = len(self.match_sources)
        matches = np.zeros((n,16))
        for i in range(n):
            matches[i,0:5] = self.match_sources[i][:]
            matches[i,5:] = self.match_stars[i][:]
        self.matches = matches
        

    def rematching(self,oldphotometrydir,filters,justmatch=False):
        '''
        Create array of sources and corresponding catalog stars
        and recalibrate using a previous photometric extraction
        '''

        sources = self.sources

        #Replace previous matches with new photometry
        olddata = loadpickle(oldphotometrydir+self.shortname+'.phot')
        oldmatches = olddata.matches
        n = len(oldmatches)
        matches = np.zeros((n,16))
        for i in range(n):
            oldx,oldy,oldflux = oldmatches[i,0:3]
            w1 = np.where((abs(sources.x-oldx)<10) & (abs(sources.y-oldy)<10))
            if len(w1[0]) >= 1:
                dev = abs(sources.flux[w1]/oldflux-1)
                w2 = np.where(dev==min(dev))
                matches[i,0:5] = [sources.x[w1[0][w2[0][0]]],sources.y[w1[0][w2[0][0]]],sources.flux[w1[0][w2[0][0]]],sources.fluxerr[w1[0][w2[0][0]]],sources.fwhm[w1[0][w2[0][0]]]]
                matches[i,5:] = oldmatches[i,5:]
        w = np.where(matches[:,0] != 0)[0]
        matches = matches[w,:]
        self.matches = matches

        self.zeropoint(filters,justmatch=justmatch)

        w = np.where((abs(sources.x-olddata.x)<5) & (abs(sources.y-olddata.y)<5) & (0.2<sources.flux/olddata.flux) & (sources.flux/olddata.flux<5))[0]
        if len(w) == 1:
            self.found = True
            self.x = sources.x[w[0]]
            self.y = sources.y[w[0]]
            self.flux = sources.flux[w[0]]
            self.fluxerr = sources.fluxerr[w[0]]
            self.fwhm = sources.fwhm[w[0]]
            self.mag = -2.5*np.log10(self.flux)+self.zp
            self.magerr = self.zperr
 
    def zeropoint(self,filters,justmatch=False):
        '''
        Fit for the zeropoint in each image using the matched sources
        '''
        if justmatch:
            self.zp,self.zperr = 0,0
            return

        #Choose correct filter magnitudes
        self.which_filter(filters)
        immag = -2.5*np.log10(self.matches[:,2])
        catmag = self.matches[:,self.filtindex]
        catmagerr = self.matches[:,self.filtindex+1]

        #Remove bad catalog stars
        w = np.where(abs(catmagerr) <= 0.1)
        catmag = catmag[w]
        catmagerr = catmagerr[w]
        immag = immag[w]

        #Initial guess and filter
        zpguess1 = catmag[0]-immag[0]
        j = 1
        done = False
        while not done:
            zpguess2 = catmag[j]-immag[j]
            if abs(zpguess1-zpguess2) < 1:
                done = True
            else:
                j += 1
        zpguess = np.mean([zpguess1,zpguess2])
        w = np.where(abs(catmag-immag-zpguess) < 1)
        catmag = catmag[w]
        catmagerr = catmagerr[w]
        immag = immag[w]

##        #Initial fit
##        fit,cov = scipy.optimize.curve_fit(fit_intercept,immag,catmag,sigma=catmagerr)
##        self.zp = fit[0]
##        self.zperr = np.sqrt(cov[0][0])
##
##        #Remove extreme 25-sigma outliers and re-fit
##        dev = abs((catmag-(self.zp+immag))/catmagerr)
##        w = np.where(dev <= 25)
##        catmag = catmag[w]
##        catmagerr = catmagerr[w]
##        immag = immag[w]
        fit,cov = scipy.optimize.curve_fit(fit_intercept,immag,catmag,sigma=catmagerr)
        self.zp = fit[0]
        self.zperr = np.sqrt(cov[0][0])

        #Plot
        plt.figure()
        plt.errorbar(immag,catmag,yerr=catmagerr,fmt='o')
        plt.plot(immag,self.zp+immag,'r-')
        plt.xlabel('Image magnitude',fontsize=14)
        plt.ylabel('PANSTARRS magnitude',fontsize=14)
        plt.savefig(self.photometrydir+self.shortname+'.png')
        plt.close()

    def which_filter(self,filters):
        '''
        Choose correct column index in matches array for the given filter
        '''

        w = np.where(filters == self.filter)[0][0]
        self.filtindex = 8+2*w

    def autotarget(self):
        '''
        Query JPL Horizons for the position of the target and match to a source
        '''

        #Execute query and retrieve position information
        eph = Horizons(id=self.object,epochs=self.time).ephemerides()
        ra,dec,raerr,decerr,mag = eph['RA'][0],eph['DEC'][0],eph['RA_3sigma'][0],eph['DEC_3sigma'][0],eph['V'][0]

        #Find matching source (within 9 sigma)
        sources = self.sources
        sourcera = self.calc_points[:,0]
        sourcedec = self.calc_points[:,1]
        seeing = np.median(self.matches[:,4])
        seeingvar = np.std(self.matches[:,4])
        if self.zp == 0:
            sourcemag = -2.5*np.log10(sources.flux)+self.zpguess
        else:
            sourcemag = -2.5*np.log10(sources.flux)+self.zp
        w = np.where((abs(sourcera-ra)/arcsectodeg<raerr*3) & (abs(sourcedec-dec)/arcsectodeg<decerr*3) & (abs(sourcemag-mag) < 2))[0]
        if len(w) == 1:
            print "Target found!"
            self.found = True
            self.x = sources.x[w[0]]
            self.y = sources.y[w[0]]
            self.flux = sources.flux[w[0]]
            self.fluxerr = sources.fluxerr[w[0]]
            self.fwhm = sources.fwhm[w[0]]
            self.mag = -2.5*np.log10(self.flux)+self.zp
            self.magerr = self.zperr
        elif len(w) > 1:
            print "WARNING: multiple possible matches found!"
            fwhm = sources.fwhm[w]
            w1 = np.where(fwhm>seeing-3*seeingvar)[0]
            if len(w1) == 1:
                self.found = True
                self.x = sources.x[w[w1[0]]]
                self.y = sources.y[w[w1[0]]]
                self.flux = sources.flux[w[w1[0]]]
                self.fluxerr = sources.fluxerr[w[w1[0]]]
                self.fwhm = sources.fwhm[w[w1[0]]]
                self.mag = -2.5*np.log10(self.flux)+self.zp
                self.magerr = self.zperr
            else:
                self.found = False
        else:
            self.found = False
