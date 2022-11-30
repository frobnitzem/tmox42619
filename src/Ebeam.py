import numpy as np

class Ebeam:
    def __init__(self):
        self.l3 = [[]]
        self.l3offset=5100
        self.initState = True
        return 
    @classmethod
    def update_h5(cls,f,ebunch,ebeamEvents):
        grpebeam = None
        if 'ebeam' in f.keys():
            grpebeam = f['ebeam']
        else:
            grpebeam = f.create_group('ebeam')
        d = grpebeam.create_dataset('l3energy',data=ebunch.l3,dtype=np.float16)
        d.attrs.create('l3offset',ebunch.l3offset,dtype=np.uint16)
        return

    def process_list(self,l3list,max_len):
        if self.initState:
            self.l3 = [[np.float16((l3-self.l3offset)) for l3 in l3list] + [0 for i in range(max_len-len(l3list))]]
            self.initState = False
        else:
            self.l3 += [[np.float16(l3-self.l3offset) for l3 in l3list] + [0 for i in range(max_len-len(l3list))]]
        return self

    def process(self,l3in):
        if self.initState:
            self.l3 = [np.float16(l3in-self.l3offset)]
        else:
            self.l3 += [np.float16(l3in-self.l3offset)]
        return self
    def set_initState(self,state):
        self.initState = state
        return self
