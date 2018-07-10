# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 23:19:28 2017

@author: iwong
"""
import pdb
import os
import time

import aux

def do_photometry(files,resultsdir,stardir,sourcedir,astrometrydir,photometrydir,filters,
                oldphotometrydir=None,rephot=False,justmatch=False,flipxy=False,
                FILTlabel='FILTER',TIMElabel='TIME',OBJlabel='OBJECT'):
    '''
    Calibrate photometry using solved astrometry OR rematch sources given
    a different aperture extraction based on a previous do_photometry run
    '''

    nfiles = len(files)
    files = sorted(files)

    for i,file in enumerate(sorted(files)[222:224]):
        phot = aux.photometry(file,resultsdir,stardir,sourcedir,astrometrydir,photometrydir,flipxy=flipxy,
                          FILTlabel=FILTlabel,TIMElabel=TIMElabel,OBJlabel=OBJlabel)
        done_phot = os.path.exists(phot.photometrydir+phot.shortname+'.phot')
        if rephot is True or done_phot is False:
            print "Working on "+phot.shortname
            if oldphotometrydir is None:
                phot.transform()
                phot.matching()
                phot.zeropoint(filters,justmatch=justmatch)
                phot.autotarget()
            else:
                phot.rematching(oldphotometrydir,filters,justmatch=justmatch)
            aux.savepickle(phot,phot.photometrydir+phot.shortname+'.phot')
            if phot.found:
                print "mag = "+str(round(phot.mag,3))+", magerr = "+str(round(phot.magerr,3))
