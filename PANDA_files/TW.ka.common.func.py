
def is_true(v):
    return(isinstance(v, bool) and v==True)

def is_false(v):
    return(isinstance(v, bool) and v==False)

def is_string(v):
    return(isinstance(v, str))

def len0(v):
    if v is None:
        return(0)
    else:
        return(len(v))

import numpy as np
def which(vv):
    return(np.where(vv)[0])

def print_list(vv, max=10):
    nn=len(vv)
    for i in range(min(nn, max)):
        print(vv[i])
    return(nn)

def freq_table(vv):
    aa=np.unique(vv, return_counts=True)
    aa2=[ [str(v) for v in bb ] for bb in aa ]
    return(np.array(aa2).T)

import pandas as pd
def freq_table1(vv):
    aa=np.unique(vv, return_counts=True)
    dd=pd.Series(aa[1], index=aa[0]) 
    return(dd)

def freq_table2(vv1, vv2):
    tbl=pd.crosstab(vv1, vv2)
    return(tbl)

import re
def grep(pat, vv, value=False, ignorecase=False, compile=False):
    vv=np.array(vv)
    if not isinstance(pat, list) and not isinstance(pat, np.ndarray):
        if compile:
            pat=re.compile(pat)
        if ignorecase:
            vv2=[ii for ii in np.arange(len(vv)) if re.search(pat, vv[ii], flags=re.I) ]
        else:
            vv2=np.array([ii for ii in np.arange(len(vv)) if re.search(pat, vv[ii]) ])
        if len(vv2)>0 and value:
            vv2=vv[vv2]
    elif isinstance(pat, list) or (isinstance(pat, np.ndarray) and pat.ndim==1):
        vv2 = [grep(pat1, vv, value=value, ignorecase=ignorecase, compile=compile) for pat1 in pat]
    return(vv2)

import os
def printdir(dir='.'):
    osdir=os.listdir(dir)
    osdir.sort()
    _ = [ print(_) for _ in osdir if _[0] != '.' ]
    print('')

def read_txt(fn, ret='\n'):
    fn=os.path.expanduser(fn)
    with open(fn, 'r') as fh:
        res=[tt.rstrip(ret) for tt in fh]
    return(res)

def write_txt(fn, obj, mode='w', ret='\n'):
    fn=os.path.expanduser(fn)
    with open(fn, mode) as fh:
        for tt in obj:
           _=fh.write(tt+ret)

def file_readable(fn1):
    fn1=os.path.expanduser(fn1)
    return(isinstance(fn1, str) and os.path.exists(fn1) and os.path.isfile(fn1) and os.access(fn1, os.R_OK))

def file_delete(fn1, warning=False):
    fn1=os.path.expanduser(fn1)
    if file_readable(fn1):
        try:
            os.remove(fn1)
        except:
            print("File exists but not deleted")
    elif warning:
        print("File not found")

def dir_exists(dir1):
    dir1=os.path.expanduser(dir1)
    return(os.path.exists(dir1) and os.path.isdir(dir1))

def mkdir(dir1, warning=False):
    dir1=os.path.expanduser(dir1)
    if dir_exists(dir1):
        if warning:
            print("Folder '%s' exists!!" % (dir1))
    else:
        try:
            os.mkdir(dir1)
        except:
            print("Error! Could not create folder '%s'!!" % (dir1))

import shutil
def force_rmdir(dir1, warning=False):
    dir1=os.path.expanduser(dir1)
    if dir_exists(dir1):
        try:
            shutil.rmtree(dir1)
        except:
            print("rmdir failed: '%s'"%dir1)
    elif warning:
        print("Folder does not exist: '%s'"%dir1)

import IPython
def is_ipython():
    st=IPython.get_ipython().__class__.__name__
    if st is None:
        return False
    elif st=='ZMQInteractiveShell':
        return True
    else:
        return False

def image255(img, to=255):
    if img.ndim==3:
        if img.shape[2]>3:
            img=img[:, :, 0:3]
    
    if to==1 and img.max()>1:
            img=img.astype(np.float32)/255
            img[img>1]=1.0
            img[img<0]=0.0
    elif to==255 and img.max()<=1:
            img=img*255
            img[img>255]=255
            img[img<0]=0
    return(img)

import cv2
def saveimg(ofn, img):
    if isinstance(ofn, np.ndarray) and isinstance(img, str):
        temp=img
        img=ofn
        ofn=temp
    ofn=os.path.expanduser(ofn)
    if img.ndim==3:
        plt.imsave(ofn, img)
    elif img.max()>1:
        img[img>255]=255
        img[img<0]=0
        cv2.imwrite(ofn, img)
    else:
        cv2.imwrite(ofn, (255*img).astype(np.uint8))

import PIL
def fig_show(arr, title='', dpi=150):
    plt.rcParams['figure.dpi']=dpi
    if isinstance(arr, PIL.Image.Image):
        _=plt.imshow(arr)
    elif arr.ndim==2:
        _=plt.imshow(arr, cmap=plt.cm.binary)
    else:
        _=plt.imshow(image255(arr))
    if title is not None and title != '':
        plt.title(str(title))
    plt.show()

import matplotlib.pyplot as plt
if is_ipython():
    get_ipython().run_line_magic('matplotlib', 'inline')

def figs_show(arrs, titles=[], ncol=3, dpi=150, wspace=0.5, hspace=0.4):
    plt.rcParams['figure.dpi']=dpi
    nfig=len(arrs)
    if isinstance(titles, str):
        titles=[titles]
    nrow=1+(nfig-1)//ncol
    #f = plt.figure(figsize=(9, 13))    
    f = plt.figure()
    plt.subplots_adjust(hspace=hspace, wspace=wspace)
    ax = []
    for ii in range(nfig):
        ax.append( f.add_subplot(nrow, ncol, ii + 1) )
        arr=arrs[ii]
        if isinstance(arr, PIL.Image.Image):
            _=plt.imshow(arr)
        elif arr.ndim==2:
            _=plt.imshow(arr, cmap=plt.cm.binary)
        else:
            _=plt.imshow(image255(arr))
        if ii<len(titles):
            title1=titles[ii]
            if title1 is not None and title1 != '':
                ax[-1].set_title(str(title1))
    plt.show(block=True)

import h5py
def save_h5(h5fn, hh): ## hh is a dict
    h5fn=os.path.expanduser(h5fn)
    with h5py.File(h5fn, "w") as hf:
        for kk in hh.keys():
            hf.create_dataset(kk, data=hh[kk])

####
