# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 23:19:28 2017

@author: iwong
"""
import pdb
import os
import time
import numpy as np

import aux


def do_astrometry_manual(files,resultsdir,astrometrydir,stardir,sourcedir,sextractfile,
                        coords=None,refindex=0,flipxy=False,RAlabel='RA',DEClabel='DEC',
                          TIMElabel='JD',FILTlabel='FILTER',OBJlabel='OBJECT',EXPTIMElabel='EXPTIME',
                          pixelscale=None,order=2,tolerance=2,num_sources=30,plotting=False,
                         reextract=False,requery=False,resolve=False,skipastro=False,alias='sex'):
    '''
    Find sources on images and query PANSTARRS for catalog stars
    '''
    
    nfiles = len(files)
    print "Processing "+str(nfiles)+" files..."

    for i,fi in enumerate(sorted(files)):
        #Load files into image objects
        print "Working on "+fi+"  "+str(i+1)+"/"+str(nfiles)
        im = aux.image(fi,resultsdir,astrometrydir,stardir,sourcedir,flipxy,
                       RAlabel,DEClabel,TIMElabel,FILTlabel,OBJlabel,EXPTIMElabel,
                       pixelscale=pixelscale)
        
        #Run SExtractor on images, if needed
        done_extract = os.path.exists(im.sourcefile)
        if reextract is True or done_extract is False:
            print "Running SExtractor..."
            im.runSExtractor(sextractfile,alias=alias)
            
        if i == refindex:
            refim = im
            
        #Query PANSTARRS for catalog stars, if needed
        done_query = os.path.exists(refim.starfile)
        if requery is True or done_query is False:
            print "Querying star catalog..."
            refim.getstars_manual(coords)
            done_query = True

        #Click matching stars and sources to solve astrometry, if needed
        if not skipastro:
            done_astro = os.path.exists(im.astrofile)
            if resolve is True or done_astro is False:
                print "Manually solving astrometry..."
                im.solveastro_manual(order,plotting,refim.starfile,num_sources=num_sources)

    return refim.starfile

def do_photometry_manual(files,resultsdir,stardir,sourcedir,photometrydir,
                oldphotometrydir=None,rephot=False,starfile=None,flipxy=False,
                FILTlabel='FILTER',TIMElabel='JD',OBJlabel='OBJECT'):
    '''
    Match sources by clicking on sources on the reference image
    and stars in the catalog OR rematch sources given a different aperture
    extraction based on a previous click_match run
    '''

    nfiles = len(files)
    files = sorted(files)
    filters = np.array(["g","r","i","z","B","V","R","I"])

    #Manually match for reference image (refindex)
    for i,file in enumerate(files):
        phot = aux.photometry(file,resultsdir,stardir,sourcedir,photometrydir,flipxy=flipxy,
                          FILTlabel=FILTlabel,TIMElabel=TIMElabel,starfile=starfile)
        done_phot = os.path.exists(phot.photometrydir+phot.shortname+'.phot')
        if rephot is True or done_phot is False:
            print "Working on "+phot.shortname
            if oldphotometrydir is None:
                aux.click_sources(phot)
                aux.click_stars(phot,phot.starfile+'.fits')
                phot.matching()
                phot.zeropoint(filters)
                phot.click_sources(target=True)
            else:
                phot.rematching(oldphotometrydir,filters)
            aux.savepickle(phot,phot.photometrydir+phot.shortname+'.phot')
            print "mag = "+str(round(phot.mag,3))+", magerr = "+str(round(phot.magerr,3))

