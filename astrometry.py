# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 23:19:28 2017

@author: iwong
"""
import pdb
import os
import time

import aux


def do_astrometry(files,astrometrydir,stardir,sourcedir,sextractfile,
                  flipxy=False,RAlabel='RA',DEClabel='DEC',
                  TIMElabel='JD',FILTlabel='FILTER',OBJlabel='OBJECT',EXPTIMElabel='EXPTIME',
                  pointingpixel=None,pixelscale=None,tolerance=5,order=3,
                  reextract=False,requery=False,resolve=False,plotting=True,skipastro=False,alias='sex'):
    '''
    Solve astrometry for a list of images with the aid of user inputs:
        flipxy: if True, up = East, right = North
        pointingpixel: image X,Y coordinates of approximate pointing
        pixelscale: pixel scale in arcseconds
        tolerance: expected maximum deviation of image scale (percent)
    '''
    
    if pixelscale is None:
        print "No pixel scale given!!"
        pdb.set_trace()
    if pointingpixel is None:
        print "Assuming center pointing"
    nfiles = len(files)
    print "Processing "+str(nfiles)+" files..."
 
    for i,fi in enumerate(sorted(files)):
        #Load files into image objects
        print "Working on "+fi+"  "+str(i+1)+"/"+str(nfiles)
        im = aux.image(fi,astrometrydir,stardir,sourcedir,flipxy,
                       RAlabel,DEClabel,TIMElabel,FILTlabel,OBJlabel,EXPTIMElabel,
                       pointingpixel,pixelscale,tolerance)
        
        #Run SExtractor on images, if needed
        done_extract = os.path.exists(im.sourcefile)
        if reextract is True or done_extract is False:
            print "Running SExtractor..."
            im.runSExtractor(sextractfile,alias=alias)
            
        #Query PANSTARRS for catalog stars, if needed
        done_query = os.path.exists(im.starfile)
        if requery is True or done_query is False:
            print "Querying star catalog..."
            im.getstars()
            
        #Solve astrometry, if needed
        if not skipastro:
            done_solve = os.path.exists(im.astrofile)
            if resolve is True or done_solve is False:
                print "Solving astrometry..."
                it = 0
                solved = False
                while it < 3 and not solved:    #Try up to 3 times
                    print "Iteration "+str(it+1)
                    solved = im.solveastro(order,plotting)
                    it += 1



