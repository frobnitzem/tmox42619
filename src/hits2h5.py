#!/sdf/group/lcls/ds/ana/sw/conda2/inst/envs/ps-4.6.3/bin/python3


import psana
import numpy as np
import sys
import re
import h5py
from scipy.fftpack import dct,dst
import os
from pathlib import Path
from typing import List

import typer
import tqdm

from Ports import *
from Ebeam import *
from Vls import *
from Gmd import *
from utils import *
from config import Config, read_config

from stream import stream, sink, take, item, apply

def xtcav_crop(inimg,win=(256,256)):
    # hard coded factor of 2 scale down
    xprof = np.mean(inimg,axis=0)
    yprof = np.mean(inimg,axis=1)
    y0 = np.argmax(xprof)
    x0 = np.argmax(yprof)
    resimg = (np.roll(inimg,(-x0+win[0]//2,-y0+win[1]//2),axis=(0,1)))[:win[0],:win[1]]
    tmp= np.column_stack((resimg,np.flip(resimg,axis=1)))
    outimg=np.row_stack((tmp,np.flip(tmp,axis=0)))
    W = dct(dct(outimg,axis=1,type=2),axis=0,type=2)
    xenv = np.zeros(W.shape[0])
    yenv = np.zeros(W.shape[1])
    xenv[:win[0]//2] = 0.5*(1+np.cos(np.arange(win[0]//2)*np.pi/(win[0]/2)))
    yenv[:win[1]//2] = 0.5*(1+np.cos(np.arange(win[1]//2)*np.pi/(win[1]/2)))
    for i in range(W.shape[1]//2):
        W[:,i] *= xenv
    for i in range(W.shape[0]//2):
        W[i,:] *= yenv
    W *= 4.0/np.product(W.shape)
    out = dct( dct(W[:win[0]//2,:win[1]//2],type=3,axis=0),type=3,axis=1)[:win[0]//4,:win[1]//4]
    return out,x0//2,y0//2
    #return dct(dct(W,axis=2,type=3),axis=1,type=3),x0,y0
    #print(x0,y0)
    #return inimg[:win[0],:win[1]],x0,y0
    

def main(cfgname : Path,
         scratchdir : Path,
         nshots : int,
         expname : str,
         runnums : List[int]) -> None:
    #    print('scratchpath=<path to file/h5files>')
    #    print('expname=tmox42619')
    #    print('nshots=100')
    #    print('configfile=<path/to/config.h5>')
    #    print('syntax: ./hits2h5.py <list of run numbers>')
    params = read_config(cfgname)

    # TODO: parallelize this loop
    for run in runnums:
        outname = scratchdir/f"hits.{expnname}.run_{run:03d}.h5"
        handle_run(run, outname, params)
        print('Finished with run %d'%run)
    print("Hello, I'm done now!")

def handle_run(run : int, outname : Path, params : Config,
                runhsd = True
                runvls = True
                runebeam = True
                runxtcav = False
                rungmd = True
              ) -> None
    chans = params.chans
    t0s = params.t0s
    logicthresh = params.logicthresh
    nr_expand = params.expand
    inflate = params.inflate

    #_ = [print('runnum %i'%int(r)) for r in runnums ]

    print('starting analysis exp %s for run %d'%(expname,run)
    #cfgname = '%s/%s.hsdconfigs.h5'%(scratchdir,expname)

    spect = Vls(params.vlsthresh)
    #spect.setthresh(params.vlsthresh)
    spect.setwin(*params.vlswin)
    ebunch = Ebeam()
    gmd = Gmd()
    ebunch.setoffset(params.l3offset)
    port = { key: Port(key, chan,
                       t0 = t0s[key],
                       logicthresh = logicthresh[key],
                       inflate = inflate,
                       expand = nr_expand,
                       nrolloff = 2**6)
             for key, chan in chans.items() }

    ds = psana.DataSource(exp=expname,run=run)
    #for run in ds.runs():
    runs = next(ds.runs())
    #np.savetxt('%s/waveforms.%s.%i.%i.dat'%(scratchdir,expname,runnum,key),wv[key],fmt='%i')#,header=headstring)
    print("detectors: ", runs.detnames)

    hsd = None
    vls = None
    ebeam = None
    xtcav = None
    xgmd = None

    eventnum = 0
    print('processing run %d'%run)
    if runhsd and 'hsd' in runs.detnames:
        hsd=runs.Detector('hsd')
    if runvls and 'andor' in runs.detnames:
        vls=runs.Detector('andor')
    if runebeam and 'ebeam' in runs.detnames:
        ebeam=runs.Detector('ebeam')
    if runxtcav and 'xtcav' in runs.detnames:
        xtcav=runs.Detector('xtcav')
    if rungmd and 'xgmd' in runs.detnames:
        xgmd=runs.Detector('xgmd')

    print('chans: ',chans)
    def show(eventnum, img):
        print('working event %i,\tnedges = %s'%(eventnum,
                    [port[k].getnedges() for k in chans.keys()] ))
        return eventnum, img

    data = Data(chans, port)

    run = process(chans,port,hsd,vls,ebeam,xtcav,xgmd) \
            >> apply(show) \
            >> apply(data.append) \
            >> write_h5(outname, port, chans)

    src = enumerate(runs.events)
    if nshots is not None: # truncate to nshots
        src = src >> take(nshots)
    # save ea. 100 of the first 1000
    src >> item[100:1000:100] >> run
    # save ea. 1000 of the remaining
    src >> item[::1000] >> run

@sink
def write_h5(data, outname, port, chans):
    for (events, spect, ebunch, gmd) in data:
        with h5py.File(outname,'w') as f:
            print('writing to %s'%outname)
            if runhsd:
                Port.update_h5(f,port,events,chans)
            if runvls:
                Vls.update_h5(f,spect,events)
            if ebeam:
                Ebeam.update_h5(f,ebunch,events) # ?ebeam
            if gmd:
                Gmd.update_h5(f,gmd,events)

class Data:
    def __init__(self, chans, port):
        #wv = {}
        #wv_logic = {}
        #v = [] # vls data matrix
        #vc = [] # vls centroids vector
        #vs = [] # vls sum is I think not used, maybe for normalization or used to be for integration and PDF sampling
        #l3 = [] # e-beam l3 (linac 3) in GeV.
        self.chans = chans
        self.port = port

        xtcavImages = []
        xtcavX0s = []
        xtcavY0s = []

        self.events = []

        self.init = True 
        #vsize = 0

    def append(self, eventnum, img, ebunch, spect, gmd):
        if self.init:
            self.init = False
            ebunch.set_initState(False)
            spect.set_initState(False)
            gmd.set_initState(False)
            for key in self.chans.keys():
                self.port[key].set_initState(False)
        if img is not None:
            self.xtcavImages.append(img[0])
            self.xtcavX0s.append(img[1])
            self.xtcavY0s.append(img[2])
        self.events.append(eventnum)

@stream
def process(events, chans,port,hsd,vls,ebeam,xtcav,xgmd):
    for eventnum, evt in events:
        completeEvent = []

        xtcav_image = None
        if xtcav and all(completeEvent):
            try:
                if xtcav.raw.value(evt) is None:
                    print(eventnum,'skip per problem with XTCAV')
                    continue
                else:
                    #completeEvent.append(True)
                    xtcav_image = xtcav.raw.value(evt)
            except Exception as e:
                print(eventnum,'skipping xtcav, skip per err: ', e)
                continue


## test vlswv
        vlswv = None
        if vls and all(completeEvent):
            ''' VLS specific section, do this first to slice only good shots '''
            vlswv = np.squeeze(vlss.raw.value(evt))
            completeEvent.append(spect.test(vlswv))


## test thisl3
        thisl3 = None
        if ebeam and all(completeEvent):
            ''' Ebeam specific section '''
            thisl3 = ebeam.raw.ebeamL3Energy(evt)
            completeEvent.append(ebunch.test(thisl3))


## test thisgmde
        thisgmde = None
        if gmd and all(completeEvent):
            thisgmde = gmd.raw.energy(evt)
            completeEvent.append(gmd.test(thisgmde))


## test hsds
        if hsd and all(completeEvent):
            if hsd is None:
                print(eventnum,'hsd is None')
                completeEvent.append(False)
            completeEvent.append(all(
                    port[key].test(hsd.raw.waveforms(evt)[chan][0])
                    for key, chan in chans.items()
                                ))
                # here key means 'port number'

        if not all(completeEvent):
            continue

## process xtcav
        img = None
        if xtcav_image:
            img = np.copy(xtcav_image).astype(np.int16)
            mf = np.argmax(np.histogram(img,np.arange(2**8))[0])
            img -= mf
            imgcrop,x0,y0 = xtcav_crop(img,win=(512,256))
            img = imgcrop, x0, y0

## process VLS
        if vlsvw:
            spect.process(vlswv)

## process ebeam
        if thisl3:
            ebunch.process(thisl3)

## process gmd
        if thisgdme:
            gmd.process(thisgmde)

## process hsds
        if hsd:
            ''' HSD-Abaco section '''
            for key, chan in chans.items(): # here key means 'port number'
                s = np.array(hsd.raw.waveforms(evt)[chan][0], dtype=np.int16) # presumably 12 bits unsigned input, cast as int16_t since will immediately in-place subtract baseline
                port[key].process(s)

        yield eventnum, img

if __name__ == '__main__':
    typer.run(main)
