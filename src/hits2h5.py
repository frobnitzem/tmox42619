#!/cds/sw/ds/ana/conda2/inst/envs/ps-4.2.5/bin/python3

            #&0,3,TMO:MPOD:01:M2:C0,TMO:MPOD:01:M9:C0,TMO:MPOD:01:M5:C0,TMO:MPOD:01:M0:C0,TMO:MPOD:01:M1:C0,TMO:MPOD:01:M6:C0,TMO:MPOD:01:M7:C0,TMO:MPOD:01:M3:C0,
            #&1,9,TMO:MPOD:01:M2:C1,TMO:MPOD:01:M9:C1,TMO:MPOD:01:M5:C1,TMO:MPOD:01:M0:C1,TMO:MPOD:01:M1:C1,TMO:MPOD:01:M6:C1,TMO:MPOD:01:M7:C1,TMO:MPOD:01:M3:C1,
            #&2,11,TMO:MPOD:01:M2:C2,TMO:MPOD:01:M9:C2,TMO:MPOD:01:M5:C2,NA,NA,NA,NA,NA,
            #&4,10,TMO:MPOD:01:M2:C4,TMO:MPOD:01:M9:C4,TMO:MPOD:01:M5:C4,TMO:MPOD:01:M0:C4,TMO:MPOD:01:M1:C4,TMO:MPOD:01:M6:C4,TMO:MPOD:01:M7:C4,TMO:MPOD:01:M3:C4,
            #&5,12,TMO:MPOD:01:M2:C5,TMO:MPOD:01:M9:C5,TMO:MPOD:01:M5:C5,TMO:MPOD:01:M0:C5,TMO:MPOD:01:M1:C5,TMO:MPOD:01:M6:C5,TMO:MPOD:01:M7:C5,TMO:MPOD:01:M3:C5,
            #&12,5,TMO:MPOD:01:M2:C12,TMO:MPOD:01:M9:C12,TMO:MPOD:01:M5:C12,TMO:MPOD:01:M0:C12,TMO:MPOD:01:M1:C12,TMO:MPOD:01:M6:C12,TMO:MPOD:01:M7:C12,TMO:MPOD:01:M3:C12,
            #&13,6,TMO:MPOD:01:M2:C13,TMO:MPOD:01:M9:C13,TMO:MPOD:01:M5:C13,TMO:MPOD:01:M0:C13,TMO:MPOD:01:M1:C13,TMO:MPOD:01:M6:C13,TMO:MPOD:01:M7:C13,TMO:MPOD:01:M3:C13,
            #&14,8,TMO:MPOD:01:M2:C14,TMO:MPOD:01:M9:C14,TMO:MPOD:01:M5:C14,TMO:MPOD:01:M0:C14,TMO:MPOD:01:M1:C14,TMO:MPOD:01:M6:C14,TMO:MPOD:01:M7:C14,TMO:MPOD:01:M3:C14,
            #&15,2,TMO:MPOD:01:M2:C15,TMO:MPOD:01:M9:C15,TMO:MPOD:01:M5:C15,TMO:MPOD:01:M0:C15,TMO:MPOD:01:M1:C15,TMO:MPOD:01:M6:C15,TMO:MPOD:01:M7:C15,TMO:MPOD:01:M3:C15,
            #&16,13,TMO:MPOD:01:M2:C16,TMO:MPOD:01:M9:C6,TMO:MPOD:01:M5:C6,NA,NA,NA,NA,NA,

import psana
import numpy as np
import sys
import h5py
from scipy.fftpack import dct,dst

from utils import mypoly

def PWRspectrum(wv):
    return np.power(abs(np.fft.fft(wv).real),int(2))

def rollon(vec,n):
    vec[:int(n)] = vec[:int(n)]*np.arange(int(n),dtype=float)/float(n)
    return vec

rng = np.random.default_rng()

###########################################
########### Class definitions #############
###########################################

class Ebeam:
    def __init__(self):
        self.l3 = []
        self.initState = True
        return 
    def process(self,l3in):
        if self.initState:
            self.l3 = [l3in]
        else:
            self.l3 += [np.uint16(l3in)]
        return self
    def set_initState(self,state):
        self.initState = state
        return self


class Vls:
    def __init__(self):
        self.v = []
        self.vsize = int(0)
        self.vc = []
        self.vs = []
        self.initState = True
        return

    def process(self, vlswv):
        #print("processing vls",vlswv.shape[0])
        num = np.sum(np.array([i*vlswv[i] for i in range(len(vlswv))]))
        den = np.sum(vlswv)
        if self.initState:
            self.v = [vlswv.astype(np.int16)]
            self.vsize = len(self.v)
            self.vc = [np.uint16(num/den)]
            self.vs = [np.uint64(den)]
        else:
            self.v += [vlswv.astype(np.int16)]
            self.vc += [np.uint16(num/den)]
            self.vs += [np.uint64(den)]
        return self

    def set_initState(self,state):
        self.initState = state
        return self

    def print_v(self):
        print(self.v[:10])
        return self

    
#def dctLogic_windowed(s,inflate=1,nrolloff=0,winsz=256,stride=128):

def dctLogicInt(s,inflate=1,nrolloff=128):
    result = np.zeros(s.shape,dtype=np.int32)
    ampscale = 2**8
    rolloff_vec = (ampscale*(1.+np.cos(np.arange(nrolloff,dtype=np.int32)*np.pi/float(nrolloff)))).astype(np.int32)
    sz_roll = rolloff_vec.shape[0] 
    sz = s.shape[0]
    sc = np.append(s,np.flip(s,axis=0)).astype(np.int32)
    ss = np.append(s,np.flip(-1*s,axis=0)).astype(np.int32)
    wc = dct(sc,type=2,axis=0).astype(np.int32)
    ws = dst(sc,type=2,axis=0).astype(np.int32)
    wc[-sz_roll:] *= rolloff_vec
    ws[-sz_roll:] *= rolloff_vec
    wc[:-sz_roll] *= ampscale # scaling since we are keeping to int32
    ws[:-sz_roll] *= ampscale
    if inflate>1: # inflating seems to increase the aliasing... so keeping to inflate=1 for the time being.
        wc = np.append(wc,np.zeros((inflate-1)*wc.shape[0],dtype=np.int32)) # adding zeros to the end of the transfored vector
        ws = np.append(ws,np.zeros((inflate-1)*ws.shape[0],dtype=np.int32)) # adding zeros to the end of the transfored vector
    Dwc = np.copy(wc)
    Dws = np.copy(ws)
    Dwc[:s.shape[0]] *= np.arange(s.shape[0],dtype=np.int32) # producing the transform of the derivative
    Dws[:s.shape[0]] *= np.arange(s.shape[0],dtype=np.int32) # producing the transform of the derivative
    dss = (dst(Dwc,type=3)[:inflate*sz]//(4*sz**2)).astype(np.int32)
    dsc = (dct(Dws,type=3)[:inflate*sz]//(4*sz**2)).astype(np.int32)
    dy = (dsc-dss)
    #dy[:-1] /= np.cos(np.pi*np.arange(inflate*sz)/2.)[:-1]
    y = (dct(wc,type=3,axis=0)[:inflate*sz]//(4*sz)).astype(np.int32)
    result = y*dy   # constructing the sig*deriv waveform 
    return result

def dctLogic(s,inflate=1,nrolloff=128):
    result = np.zeros(s.shape,dtype=np.float32)
    if nrolloff>winsz:
        print('rolloff larger than windowed signal vec')
        return result
    if nrolloff!=0:
        print('rolloff is non-zero... dont bother with that')
        return result
    
    rolloff_vec = 0.5*(1.+np.cos(np.arange(nrolloff,dtype=float)*np.pi/float(nrolloff)))
    sz_roll = rolloff_vec.shape[0] 
    sz = s.shape[0]
    Yc = dct(np.append(s,np.flip(s,axis=0)),type=2)
    Ys = dst(np.append(s,np.flip(s,axis=0)),type=2)
    Yc[-sz_roll:] *= rolloff_vec
    Ys[-sz_roll:] *= rolloff_vec
    if inflate>1: # inflating seems to increase the aliasing... so keeping to inflate=1 for the time being.
        Yc = np.append(Yc,np.zeros((inflate-1)*Yc.shape[0])) # adding zeros to the end of the transfored vector
        Ys = np.append(Ys,np.zeros((inflate-1)*Ys.shape[0])) # adding zeros to the end of the transfored vector
    DYc = np.copy(Yc)
    DYs = np.copy(Ys)
    DYc[:s.shape[0]] *= np.arange(s.shape[0],dtype=float)/s.shape[0] # producing the transform of the derivative
    DYs[:s.shape[0]] *= np.arange(s.shape[0],dtype=float)/s.shape[0] # producing the transform of the derivative
    dys = dst(DYc,type=3)[:inflate*sz]/(4*sz**2)
    dyc = dct(DYs,type=3)[:inflate*sz]/(4*sz**2)
    dy = (dyc-dys)
    dy[:-1] /= np.cos(np.pi*np.arange(inflate*sz)/2.)[:-1]
    y = dct(Yc,type=3)[:inflate*sz]
    result = y*dy   # constructing the sig*deriv waveform 
    return result

'''
def dctLogic(s,inflate=1,nrolloff=128):
    rolloff_vec = 0.5*(1.+np.cos(np.arange(nrolloff,dtype=float)*np.pi/float(nrolloff)))
    sz_roll = rolloff_vec.shape[0] 
    sz = s.shape[0]
    wave = np.append(s,np.flip(s,axis=0))
    WAVE = dct(wave,type=2)
    #WAVE = dct(wave)
    #WAVE = rollon(WAVE,4)
    WAVE[-sz_roll:] *= rolloff_vec
    if inflate>1: # inflating seems to increase the aliasing... so keeping to inflate=1 for the time being.
        WAVE = np.append(WAVE,np.zeros((inflate-1)*WAVE.shape[0])) # adding zeros to the end of the transfored vector
    DWAVE = np.copy(WAVE) # preparing to also make a derivative
    DWAVE[:s.shape[0]] *= np.arange(s.shape[0],dtype=float)/s.shape[0] # producing the transform of the derivative
    return dct(WAVE,type=3)[:inflate*sz]*dct(DWAVE,type=4)[:inflate*sz]/(4*sz**2) # constructing the sig*deriv waveform 
    #return dct(WAVE,type=3)[:inflate*sz]*dst(DWAVE,type=3)[:inflate*sz]/(4*sz**2) # constructing the sig*deriv waveform 
'''


def scanedges_simple(d,minthresh,expand=1):
    tofs = []
    slopes = []
    sz = d.shape[0]
    i = 10
    while i < sz-10:
        while d[i] > minthresh:
            i += 1
            if i==sz-10: return tofs,slopes,len(tofs)
        while i<sz-10 and d[i]<0:
            i += 1
        stop = i
        x0 = stop - 1./float(d[stop]-d[stop-1])*d[stop] 
        i += 1
        v = expand*float(x0)
        tofs += [np.int32(v) + np.int32(rng.random()<v%1)] 
        slopes += [d[stop]-d[stop-1]] ## scaling to reign in the obscene derivatives... probably shoul;d be scaling d here instead
    return tofs,slopes,len(tofs)

def scanedges(d,minthresh,expand=4):
    tofs = []
    slopes = []
    sz = d.shape[0]
    newtloops = 6
    order = 3 # this should stay fixed, since the logic zeros crossings really are cubic polys
    i = 10
    while i < sz-10:
        while d[i] > minthresh:
            i += 1
            if i==sz-10: return tofs,slopes,len(tofs)
        while i<sz-10 and d[i]<d[i-1]:
            i += 1
        start = i-1
        i += 1
        while i<sz-10 and d[i]>d[i-1]:
            i += 1
        stop = i
        i += 1
        if (stop-start)<(order+1):
            continue
        x = np.arange(stop-start,dtype=float) # set x to index values
        y = d[start:stop] # set y to vector values
        x0 = float(stop)/2. # set x0 to halfway point
        #y -= (y[0]+y[-1])/2. # subtract average (this gets rid of residual DC offsets)

        theta = np.linalg.pinv( mypoly(np.array(x).astype(float),order=order) ).dot(np.array(y).astype(float)) # fit a polynomial (order 3) to the points
        for j in range(newtloops): # 3 rounds of Newton-Raphson
            X0 = np.array([np.power(x0,int(k)) for k in range(order+1)])
            x0 -= theta.dot(X0)/theta.dot([i*X0[(k+1)%(order+1)] for k in range(order+1)]) # this seems like maybe it should be wrong
        tofs += [float(start + x0)] 
        X0 = np.array([np.power(x0,int(i)) for k in range(order+1)])
        #slopes += [np.int64(theta.dot([i*X0[(i+1)%(order+1)] for i in range(order+1)]))]
        slopes += [float((theta[1]+x0*theta[2])/2**18)] ## scaling to reign in the obscene derivatives... probably shoul;d be scaling d here instead
    return tofs,slopes,len(tofs)

class Port:
    # Note that t0s are aligned with 'prompt' in the digitizer logic signal
    # Don't forget to multiply by inflate, also, these look to jitter by up to 1 ns
    # hard coded the x4 scale-up for the sake of filling int16 dynamic range with the 12bit vls data and finer adjustment with adc offset correction

    def __init__(self,portnum,hsd,t0=0,nadcs=4,baselim=1000,logicthresh=-2400,slopethresh=500,scale=1,inflate=1,expand=1,nrolloff=256): # exand is for sake of Newton-Raphson
        self.portnum = portnum
        self.hsd = hsd
        self.t0 = t0
        self.nadcs = nadcs
        self.baselim = baselim
        self.logicthresh = logicthresh
        self.slopethresh = slopethresh
        self.initState = True
        self.scale = scale
        self.inflate = inflate
        self.expand = expand
        self.nrolloff = nrolloff
        self.sz = 0
        self.tofs = []
        self.slopes = []
        self.addresses = []
        self.nedges = []
        self.waves = {}
        self.shot = int(0)

    def process(self,s):
        if type(s) == type(None):
            e = []
            de = []
            ne = 0
        else:
            for adc in range(self.nadcs):
                b = np.mean(s[adc:self.baselim+adc:self.nadcs])
                s[adc::self.nadcs] = (s[adc::self.nadcs] * self.scale) - int(self.scale*b)
            logic = dctLogicInt(s,inflate=self.inflate,nrolloff=self.nrolloff) #produce the "logic vector"
            if len(self.addresses)%100 == 0:
                self.waves.update( {'shot_%i'%len(self.addresses):np.copy(logic)} )
            e,de,ne = scanedges_simple(logic,self.logicthresh,self.expand) # scan the logic vector for hits
                        # the expand here is how much we subdivide the pixels in the already dct expanded digitizer steps (sake of Newton-Raphson root resolution)
        if self.initState:
            self.sz = s.shape[0]*self.inflate*self.expand
            self.tofs = [0]
            if ne<1:
                self.addresses = [int(0)]
                self.nedges = [int(0)]
                self.tofs += []
                self.slopes += []
            else:
                self.addresses = [int(1)]
                self.nedges = [int(ne)]
                self.tofs += e
                self.slopes += de
        else:
            if ne<1:
                self.addresses += [int(0)]
                self.nedges += [int(0)]
                self.tofs += []
                self.slopes += []
            else:
                self.addresses += [int(len(self.tofs))]
                self.nedges += [int(ne)]
                self.tofs += e
                self.slopes += de
        return self

    def set_initState(self,state=True):
        self.initState = state
        return self

    def print_tofs(self):
        print(self.tofs)
        print(self.slopes)
        return self

def main():
        ############################################
        ###### Change this to your output dir ######
        ############################################
    #scratchdir = '/reg/data/ana16/tmo/tmox42619/scratch/ryan_output/h5files'
    scratchdir = '/reg/data/ana16/tmo/tmox42619/scratch/ryan_output_2022/h5files'
    expname = 'tmox42619'
    runnum = 62 
    nshots = 100
    if len(sys.argv)>2:
        expname = sys.argv[1]
        runnum = int(sys.argv[2])

    if len(sys.argv)>3:
        nshots = int(sys.argv[3])

    print('starting analysis exp %s for run %i'%(expname,int(runnum)))
    nr_expand = 4 
    chans = {0:3,1:9,2:11,4:10,5:12,12:5,13:6,14:8,15:2,16:13} # HSD to port number:hsd
    logicthresh = {0:-800000, 1:-800000, 2:-400000, 4:-800000, 5:-800000, 12:-800000, 13:-800000, 14:-800000, 15:-800000, 16:-400000}
    #slopethresh = {0:500,1:500,2:300,4:150,5:500,12:500,13:500,14:500,15:500,16:300}
    slopethresh = {0:100,1:100,2:60,4:100,5:100,12:100,13:100,14:100,15:100,16:60}
    #t0s = {0:109840,1:100456,2:99924,4:97180,5:99072,12:98580,13:98676,14:100348,15:106968,16:98028}
    #t0s = {0:109830,1:100451,2:99810,4:97180,5:99071,12:98561,13:98657,14:100331,15:106956,16:97330}
    t0s = {0:73227,1:66793,2:60000,4:64796,5:66054,12:65712,13:65777,14:66891,15:71312,16:60000} # final, based on inflate=4 nr_expand=4

    '''
    argon   prompt>300      proposed
    0       109500  109830
    1       100121  100451
    2       99480   99810
    4       96850   97180
    5       98741   99071
    12      98231   98561
    13      98327   98657
    14      100001  100331
    15      106626  106956
    16      97000   97330
    '''


    spect = Vls()
    ebunch = Ebeam()
    port = {} 
    scale = int(1) # to better fill 16 bit int
    inflate = int(4) 
    for key in logicthresh.keys():
        logicthresh[key] *= scale # inflating by factor of 4 since we are also scaling the waveforms by 4 in vertical to fill bit depth.

    for key in chans.keys():
        port[key] = Port(key,chans[key],t0=t0s[key],logicthresh=logicthresh[key],slopethresh=slopethresh[key],inflate=inflate,expand=nr_expand,scale=scale,nrolloff=10000)

    ds = psana.DataSource(exp=expname,run=runnum)

    #for run in ds.runs():
    run = next(ds.runs())
        #np.savetxt('%s/waveforms.%s.%i.%i.dat'%(scratchdir,expname,runnum,key),wv[key],fmt='%i',header=headstring)
    for i in range(1):
        print(run.detnames)
        eventnum = 0
        runhsd=True
        runvls=False
        runebeam=False
        hsd = run.Detector('hsd')
        vls = run.Detector('andor')
        ebeam = run.Detector('ebeam')
        wv = {}
        wv_logic = {}
        v = [] # vls data matrix
        vc = [] # vls centroids vector
        vs = [] # vls sum is I think not used, maybe for normalization or used to be for integration and PDF sampling
        l3 = [] # e-beam l3 (linac 3) in GeV.


        init = True 
        vsize = 0

        print(run.events())
        print('chans: ',chans)
        for evt in run.events():
            if eventnum > nshots:
                break

            if runvls:
                ''' VLS specific section, do this first to slice only good shots '''
                try:
                    vlswv = np.squeeze(vls.raw.value(evt))
                    vlswv = vlswv-int(np.mean(vlswv[1900:])) # this subtracts baseline
                    if np.max(vlswv)<300:  # too little amount of xrays
                        print(eventnum,'skip per weak vls')
                        #eventnum += 1
                        continue
                    spect.process(vlswv)
                    #spect.print_v()

                except:
                    print(eventnum,'skip per vls')
                    continue

            if runebeam:
                ''' Ebeam specific section '''
                try:
                    thisl3 = ebeam.raw.ebeamL3Energy(evt)
                    thisl3 += 0.5
                    ebunch.process(thisl3)
                except:
                    print(eventnum,'skipping ebeam, skip per l3')
                    continue

            if runhsd:
    
                ''' HSD-Abaco section '''
                for key in chans.keys(): # here key means 'port number'
                    try:
                        s = np.array(hsd.raw.waveforms(evt)[ chans[key] ][0] , dtype=np.int16) 
                        port[key].process(s)

                        if init:
                            init = False
                            ebunch.set_initState(False)
                            spect.set_initState(False)
                            for key in chans.keys():
                                port[key].set_initState(False)
                    except:
                        print(eventnum, 'failed hsd for some reason')
                        continue

                if eventnum<100:
                    if eventnum%10<2: 
                        print('working event %i'%eventnum)
                elif eventnum<1000:
                    if eventnum%100<2: 
                        print('working event %i'%eventnum)
                else:
                    if eventnum%1000<2: 
                        print('working event %i'%eventnum)
                eventnum += 1

        f = h5py.File('%s/hits.%s.run%i.h5'%(scratchdir,expname,runnum),'w') 
                # use f.create_group('port_%i'%i,portnum)
        #_ = [print(key,chans[key]) for key in chans.keys()]
        if runhsd:
            for key in chans.keys(): # remember key == port number
                g = f.create_group('port_%i'%(key))
                g.create_dataset('tofs',data=port[key].tofs,dtype=np.int32) 
                g.create_dataset('slopes',data=port[key].slopes,dtype=np.int32) 
                g.create_dataset('addresses',data=port[key].addresses,dtype=np.uint64)
                g.create_dataset('nedges',data=port[key].nedges,dtype=np.uint32)
                wvgrp = g.create_group('waves')
                for k in port[key].waves.keys():
                    wvgrp.create_dataset(k,data=port[key].waves[k],dtype=np.int32)
                g.attrs.create('inflate',data=port[key].inflate,dtype=np.uint8)
                g.attrs.create('expand',data=port[key].expand,dtype=np.uint8)
                g.attrs.create('t0',data=port[key].t0,dtype=float)
                g.attrs.create('slopethresh',data=port[key].slopethresh,dtype=np.uint64)
                g.attrs.create('hsd',data=port[key].hsd,dtype=np.uint8)
                g.attrs.create('size',data=port[key].sz*port[key].inflate,dtype=int) ### need to also multiply by expand #### HERE HERE HERE HERE
        if runvls:
            grpvls = f.create_group('vls')
            grpvls.create_dataset('data',data=spect.v,dtype=np.int16)
            grpvls.create_dataset('centroids',data=spect.vc,dtype=np.int16)
            grpvls.create_dataset('sum',data=spect.vs,dtype=np.uint64)
            grpvls.attrs.create('size',data=spect.vsize,dtype=np.int32)
        if runebeam:
            grpebeam = f.create_group('ebeam')
            grpebeam.create_dataset('l3energy',data=ebunch.l3,dtype=np.uint16)
        f.close()

    print("Hello, I'm done now!")
    return

if __name__ == '__main__':
    main()
