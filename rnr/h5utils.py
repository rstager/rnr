import h5py
import numpy as np


class h5record(object):
    '''
    record data from dictionary into h5file.
    '''
    dtmap={np.dtype('float32'):'f',np.dtype('float64'):'f',np.dtype('uint8'):'uint8'}
    def __init__(self,h5file,maxidx=1000000,exclude={},group=None,mode='w'):
        if isinstance(h5file,h5py.File):
            self.h5file=h5file
        else:
            self.h5file=h5py.File(h5file,mode)
        self.maxidx=maxidx
        self.idx=0
        self.exclude=exclude
        self.datasets={}
        self.groupname=group
        self.__new_groupname=self.groupname
        if group is None:
            self.group=self.h5file

    def newgrp(self,groupname):
        self.__new_groupname=groupname


    def __call__(self,path,fields=None,batch=False):
        '''
        Record data from dictionary. Each value will be appended to a dataset by the key.
        If the value is a numpy array or a python list, it will be appended as a row in the dataset by the
        appropriate size. If the value is a scaler, then the dataset will have a single column
        :param path:
        :return:
        '''
        # create group in logging thread to avoid race condition
        if self.__new_groupname is not None:
            self.groupname = self.__new_groupname
            self.group=getattr(self.h5file,self.groupname,None)
            if self.group is None:
                self.group=self.h5file.create_group(self.groupname)
            self.datasets={}
            self.idx=0
            self.__new_groupname=None
        l=None
        if fields is None:
            fields=path.keys()
        for name in fields:
            try:
                data = path.get(name,None)
                #print("name {} len {}".format(name,len(data)))
                if name in self.exclude:
                    continue
                if not batch:
                    data=[data]
                if len(data)==0:continue
                if not name in self.datasets:
                    assert self.idx==0,'{}/{} data set must appear in first call to record {} '\
                        .format(self.groupname,name,self.idx)
                    if  name in self.group.keys():
                        del self.group[name]
                    if hasattr(data[0],'dtype'):
                        ftype=h5record.dtmap[data[0].dtype]
                    else:
                        ftype='f'
                    if  hasattr(data[0],'shape'): #numpy array
                        self.datasets[name] = self.group.create_dataset(name, (self.maxidx,) + data[0].shape,ftype,
                                                                     chunks=(10,) + data[0].shape, compression='lzf')
                    elif hasattr(data[0],'__iter__'): # python list
                        #print(type(data),type(data[0]))
                        self.datasets[name] = self.group.create_dataset(name, (self.maxidx, len(data[0])),ftype,
                                                                   chunks=(10, len(data[0])), compression='lzf')
                    else:
                        #print(type(data),type(data[0])) #scalar
                        self.datasets[name] = self.group.create_dataset(name, (self.maxidx,1),
                                                                    ftype,
                                                                    chunks=(10,1), compression='lzf')

                if l is None:
                    l = len(data)
                assert l == len(data),'{}: all path data must have the same length {} != {}'.format(name,l,len(data))
                for idx in range(l):
                        self.datasets[name][self.idx+idx]=data[idx]
            except Exception as e:
                print(name,e)
        self.idx += l
        self.group.attrs['maxidx']=self.idx
        self.h5file.flush()

    def close(self):
        self.h5file.close()
