#!/usr/bin/env python

import os
import sys
import re
import numpy as np
import pandas as pd
import subprocess
from glob import glob
from tqdm.notebook import tqdm
import gc
import shutil
import openslide
import albumentations as albu
import argparse


args = sys.argv[1:]
import argparse

ver='1'
parser = argparse.ArgumentParser(
                prog='PRAD.ka.prepare_tiles.v%s.py' % ver, 
                description="PRAD histological imaging and ISUP grading for kaggle", 
                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                add_help=True)

parser.add_argument('--version', action='version', version=ver+'.0', help=argparse.SUPPRESS)

parser.add_argument("--training", dest='training_flag', action='store_true', default=False, help="save tiles for training data")

parser.add_argument("-l", "--lvl", "--work_lvl", dest='level', metavar='None', type=int,
            default=None, help="working level of WSI tiff")

parser.add_argument("--mask_fr_min", dest='mask_fr_min0', metavar='0.25', type=float,
            default=0.25, help="mask_fr_min0")

parser.add_argument("--Nmax", "--nmax", dest='nmax', metavar='100', type=int,
            default=100, help="filter tiles with larger unmapped area")

parser.add_argument("--N0", "--n0", dest='n0', metavar='80', type=int,
            default=80, help="select top image signals as step2 training")

parser.add_argument("--N1", "--n1", dest='n1', metavar='48', type=int,
            default=48, help="select top TN scores")

#parser.add_argument("--N2", "--n2", dest='n2', metavar='32', type=int,
#            default=32, help="select top GL scores")

parser.add_argument("--N3", "--n3", dest='n3', metavar='16', type=int,
            default=16, help="select top GL*TN scores")

parser.add_argument("--N3min0", "--n3min0", dest='n3min0', metavar='3', type=int,
            default=3, help="minimal number of selected images")

parser.add_argument("-m", "--model_dir", dest='model_dir', metavar="model_dir", type=str, 
            default='../input/pretrained-models/prev_model', help="pretrained model folder")

parser.add_argument("-r", "--resource_dir", dest='resource_dir', metavar="resource_dir", type=str, 
            default='../input/tw-resources-modules/resources', help="resource and scripts folder")

parser.add_argument("--seed", dest='seed', metavar="123456", type=int, 
            default=123456, help="seed number for RND")

options = parser.parse_args(args)

SEED=options.seed

work_lvl=None
try:
    work_lvl=options.level
    if work_lvl is not None:
        work_lvl=int(float(work_lvl))
except:
    pass

print("work_lvl =", str(work_lvl))

TRAIN_step2_flag=False
try:
    TRAIN_step2_flag=options.training_flag
except:
    pass

SCRIPTS=options.resource_dir
if SCRIPTS=='':
    SCRIPTS=os.path.dirname(sys.argv[0])
print("resource_dir =", SCRIPTS)

PRETRAINED=options.model_dir
print("previous model_dir =", PRETRAINED)



mask_fr_min0=0.25

Nmax=100  ## filter top area
N0=80  ## select top image signals as step2 training
N1=48  ## filter top TN scores
N2=32  ## filter top GL scores
N3=16  ## filter top GL scores
N3min0=3

mask_fr_min0=options.mask_fr_min0
Nmax=options.nmax 
N0=options.n0 
N1=options.n1 
#N2=options.n2 
N3=options.n3 
N3min0=options.n3min0

print(mask_fr_min0, Nmax, N0, N1, N3, N3min0)


def import_module1(pkgname, return_module=False):
    if os.path.dirname(pkgname)=='':
        pkgpath=os.getcwd()
    else:
        pkgpath = os.path.abspath(os.path.expanduser(os.path.dirname(pkgname)))
        pkgname = os.path.basename(pkgname)
    #path = list(sys.path)
    sys.path.insert(0, pkgpath)
    if pkgname[-3:]=='.py':
        pkgname=pkgname[:-3]
    try:
        modu=__import__(pkgname)
        #sys.path[:] = path
        sys.path = sys.path[1:]
        if isinstance(return_module, bool) and return_module:
            return modu
        elif isinstance(return_module, str):
            globals()[return_module]=modu
        else:
            if len(pkgname.split('.'))==1:
                globals()[pkgname]=modu
            else:
                return(modu)
    except:
        sys.path = sys.path[1:]
        raise

def execfile(_script):
    _script=os.path.expanduser(_script)
    exec(open(_script).read(), globals())
    globals()['.temp']=locals()
    for k in globals()['.temp'].keys():
        if k[0]!="_":
            globals()[k]=globals()['.temp'][k]
    del(globals()['.temp'])

import subprocess
def install_package1(package, force=False, upgrade=True, nodeps=False, verbose=True):
    #import subprocess
    #import sys
    tmplst=[sys.executable, "-m", "pip", "install"]
    if force:
        tmplst.append('-y')
    if upgrade:
        tmplst.append('-U')
    if nodeps:
        tmplst.append('--no-deps')
    tmplst.append(package)
    if verbose:
        print("!"+' '.join(tmplst))
    subprocess.check_call(tmplst)

def run_cmd(cmd, output=False):
    #import subprocess
    p = subprocess.Popen(["bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if not err == '' and not err == b'' :
        err=err.decode("utf-8")
        err=re.sub('\nbash: line [0-9]*', 'ERROR', '\n'+err)
        print(err)
    if out is not None:
        out=out.decode("utf-8")
        if not output:
            print(out)
        else:
            return out.splitlines()

#WORKDIR='/kaggle/working'
WORKDIR=os.getcwd()

WORKDIR=os.path.expanduser(WORKDIR)

print("WORKDIR = %s"%WORKDIR)


execfile(os.path.join(SCRIPTS, 'TW.ka.common.func.py'))

import tensorflow as tf
print('tensorflow', tf.__version__)

try:
    import imagecodecs
except:
    install_package1(os.path.join(SCRIPTS, 'imagecodecs-2020.5.30-cp37-cp37m-manylinux2010_x86_64.whl'), upgrade=False)
    import imagecodecs

print('imagecodecs', imagecodecs.__version__)

try:
    import nimfa
except:
    install_package1(os.path.join(SCRIPTS, 'nimfa-1.4.0-py2.py3-none-any.whl'), upgrade=False)
    import nimfa

print('nimfa', nimfa.__version__)

import_module1(os.path.join(SCRIPTS, 'TW_color_normalization_htk.py'), 'CN')

print(CN.__version__)

try:
    import keras_applications
except:
    install_package1(os.path.join(SCRIPTS, 'Keras_Applications-1.0.8-py3-none-any.whl'), upgrade=False)
    import keras_applications

print("keras_applications", keras_applications.__version__)

try:
    import efficientnet
except:
    #run_cmd('pip install --no-deps %s/efficientnet-1.1.0-py3-none-any.whl'%SCRIPTS)
    install_package1(os.path.join(SCRIPTS, 'efficientnet-1.1.0-py3-none-any.whl'), upgrade=False, nodeps=True)
    import efficientnet

print('efficientnet', efficientnet.__version__)

efficientnet.init_keras_custom_objects()
efficientnet.init_tfkeras_custom_objects()

from efficientnet.model import EfficientNetB4, EfficientNetB3, EfficientNetB1

GL_model_fn=os.path.join(PRETRAINED, 'PRAD_GL_effB4.sz128_Lv1.hdf5')
TN_model_fn=os.path.join(PRETRAINED, 'PRAD_TN_effB4_sigmoid.sz128_Lv1.hdf5')

###

import os
import sys
print(sys.executable)
print(sys.version)

import keras
print('Keras version : {}'.format(keras.__version__))

import tensorflow as tf
print('Tensorflow version : {}'.format(tf.__version__))

try:
    tf.get_logger().setLevel('ERROR')
    print(tf.get_logger())
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    print(tf.compat.v1.logging.get_verbosity())
    tf.autograph.set_verbosity(3)
except:
  try:
    tf.compat.v1.get_logger().setLevel('ERROR')
    print(tf.compat.v1.get_logger())
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    print(tf.compat.v1.logging.get_verbosity())
    tf.autograph.set_verbosity(3)
  except:
    pass

print('GPU available: ', tf.test.is_gpu_available())

###

import keras
from keras.models import Model, load_model, clone_model
from keras.layers import Conv2D, BatchNormalization, multiply, LocallyConnected2D, Lambda, Concatenate
from keras.layers import Input, Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras import regularizers, constraints

###

class InferenceDataGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 X,
                 y=None,
                 batch_size = 8
                 ):
        
        self.batch_size = batch_size
        self.X = np.array(X)
        self.y = y
        self.indices = np.arange(len(X))
        
    def __len__(self):
        nnn = int(1+ ((len(self.X)-1) // self.batch_size))
        return nnn
    
    def on_epoch_start(self):
        pass
    
    def __getitem__(self, step):
        batch_indices = self.indices[step * self.batch_size : (step+1) * self.batch_size]
        if self.y is None:
            batch_labels = None
        else:
            batch_labels = self.y[batch_indices]
        batch_images = self.X[batch_indices]
        return np.stack(batch_images), batch_labels 
        
    def __getimages__(self, image_file):
            img = plt.imread(image_file).astype(np.float32)
            #if self.aug:
            #    img=self.aug(image=img)['image']
            return img / 255.0

def select_large(val, N, pre=None):
    if N==0:
        return(np.inf)
    elif not pre is None:
        if ((np.array(val)>=pre).sum()>=N):
            return(pre)
    ind=np.isnan(val)
    if ind.sum()>0:
        val=np.array(val)[np.logical_not(ind)]
    if len(val)==0:
        thr=None
    else:
        mm=np.min(val)
        if N>=len(val):
            thr=mm
        elif N>(val>mm).sum():
            thr=mm
        else:
            thr=np.sort(val)[::-1][N-1]
    return(thr)

####

def ind2pos(ii, sztile, nX):
    iYX=np.array([ii//nX, ii%nX])
    pYX=iYX*sztile
    return(pYX)

def uuid2LR2(uuid, LRdir, input_dir=None, retrain=False, verbose=False):
    ## generate Linear Regression coeffs for color normalization
    LRnpy=os.path.join(LRdir, uuid+'_LRw2.npy')
    LRw2=None
    if file_readable(LRnpy) and not retrain:
        LRw2=np.load(LRnpy)
        if LRw2.shape!=(3,3):
            LRw2=None
        if verbose:
            print(LRnpy)
    if LRw2 is None and input_dir is not None:
        ifn=os.path.join(input_dir, uuid+".tiff")
        wsi=openslide.OpenSlide(ifn)
        work_lvls=wsi.level_dimensions
        thumb2=np.asarray(wsi.read_region( (0,0), len(work_lvls)-1, work_lvls[-1]))[:,:,:3]
        wsi.close()
        try:
            LRw2=CN.TW_img2LR2(thumb2, mask_excl=True, iter=3)
        except:
            LRw2=np.array([])
        np.save(LRnpy, LRw2)
    return(LRw2)

def img2bestcut(img1, pstep=None, szcut=None, plot=False):
    sz = img1.shape[0]
    if szcut is None:
        szcut = sz//2
    if pstep is None:
        pstep=szcut//2
    imH1=CN.im2stain(img1)[:,:,0]
    imSig1=1-(imH1/255.0)
    imK1=np.logical_and(imSig1>0.35, imSig1<0.65)
    imw=imK1
    dd=(sz-szcut)//pstep
    #print(dd)
    pps1=np.arange(dd+1)*pstep
    #print(pps1)
    PPs=[]
    SigN=[]
    cutsigs=[]
    for py in pps1:
        for px in pps1:
            cutsig1=imw[py:(py+szcut), px:(px+szcut)]
            meansig1=cutsig1.mean()
            SigN.append(meansig1)
            PPs.append((py, px))
            if plot:
                cutsigs.append(cutsig1)
    #print([round(ss,3) for ss in SigN])
    ii=np.argmax(np.array(SigN))
    py, px=PPs[ii]
    if plot:
        figs_show(cutsigs, ncol=dd+1)
    return(ii, py, px)

def uuid2tiles5(uuid, input_dir, output1_dir, output0_dir, LRdir, TNdir, return_img=False,
    ref_csv=None, testing_mode=False, verbose=True, sztile1=128, sztile0=256, save_img=True):
    
    if testing_mode:
        N3min=2
        mask_fr_min=0.15
    else:
        N3min=N3min0
        mask_fr_min=mask_fr_min0
    
    sztile=sztile1
    sztile2=sztile//2
    imask_cut1=sztile*sztile*mask_fr_min
    imask_cut1b=sztile*sztile*0.9
    imask_cut1c=sztile*sztile*0.3
    
    mkdir(LRdir)
    
    if verbose:
        print('\n')
        if ref_csv is not None:
            ii=np.where(np.asarray(ref_csv['image_id'].values)==uuid)[0]
            if len(ii)>0:
                print(ref_csv.iloc[ii])
    
    ## process Lv1 images
    if output1_dir is not None:
        mkdir(output1_dir)
        pat1H=os.path.join(output1_dir, uuid+"_H*")
        UDIR1=glob(pat1H)
        if len(UDIR1)>0:
            UDIR1=UDIR1[0]
            ufns1=glob(os.path.join(UDIR1, uuid+'.*.jpg'))
            if verbose:
                print('UDIR1', UDIR1, len(ufns1))
        else:
            UDIR1=''
            ufns1=[]
        pat1X=os.path.join(output1_dir, uuid+"_X*")
        UDIR1X=glob(pat1X)
        if len(UDIR1X)>0:
            UDIR1X=UDIR1X[0]
            if verbose:
                print('UDIR1X', UDIR1X)
        else:
            UDIR1X=''
    else:
        UDIR1=''
        ufns1=[]
        UDIR1X=''
    
    ## process Lv0 images
    if output0_dir is not None:
        mkdir(output0_dir)
        pat0H=os.path.join(output0_dir, uuid+"_H*")
        UDIR0=glob(pat0H)
        if len(UDIR0)>0:
            UDIR0=UDIR0[0]
            ufns0=glob(os.path.join(UDIR0, uuid+'.*.jpg'))
            if verbose:
                print('UDIR0', UDIR0, len(ufns0))
        else:
            UDIR0=''
            ufns0=[]
        pat0X=os.path.join(output0_dir, uuid+"_X*")
        UDIR0X=glob(pat0X)
        if len(UDIR0X)>0:
            UDIR0X=UDIR0X[0]
            if verbose:
                print('UDIR0X', UDIR0X)
        else:
            UDIR0X=''
    else:
        UDIR0=''
        ufns0=[]
        UDIR0X=''
    
    final=None
    if dir_exists(UDIR1X) or dir_exists(UDIR0X):
        if verbose:
            print("Already skipped due to insufficient tiles")
            print(UDIR1X)
            print(UDIR0X)
    elif (output1_dir is None or len(ufns1)>=N3min) and (output0_dir is None or len(ufns0)>=N3min) and (output1_dir is None or output0_dir is None or len(ufns0)==len(ufns1)):
        if verbose:
            if output1_dir is not None:
                print("Already %g jpg files"%len(ufns1))
            else:
                print("Already %g jpg files"%len(ufns0))
    else:
        ## generate Linear Regression coeffs for color normalization
        LRw2=uuid2LR2(uuid, LRdir, input_dir)
        mkdir(TNdir)
        
        if output1_dir is not None:
            force_rmdir(UDIR1)
        if output0_dir is not None:
            force_rmdir(UDIR0)
        
        ifn=os.path.join(input_dir, uuid+".tiff")
        wsi=openslide.OpenSlide(ifn)
        work_lvls=wsi.level_dimensions
        thumb1=np.asarray(wsi.read_region( (0,0), 1, work_lvls[1]))[:,:,:3]
        if verbose:
            print('Lv 1: Y, X =', thumb1.shape)
        wsi.close()
        
        img=thumb1
        shape=img.shape
        
        ## create tissue mask: imask==inclusion_mask
        imask1a=CN.detect_tissue0(img, sensitivity=1)
        imask1b=np.logical_not(CN.mask_white(img))
        imask1c=np.logical_and(imask1a, imask1b)
        imask1=np.stack([imask1c]*3, axis=2)
        
        padBot, padRt = (sztile - shape[0]%sztile)%sztile, (sztile - shape[1]%sztile)%sztile
        if verbose:
            print('padBot, padRt =', padBot, padRt)
        
        img1 = np.pad(img, [[0, padBot], [0,padRt], [0,0]], mode='constant', constant_values=255)
        
        nY=img1.shape[0]//sztile
        nX=img1.shape[1]//sztile
        img1A = img1.reshape(nY, sztile, nX, sztile,3)
        img1B = img1A.transpose(0,2,1,3,4).reshape(-1,sztile,sztile,3)
        xy1=[ind2pos(ii, sztile=sztile, nX=nX)  for ii in range(len(img1B))]
        
        if verbose:
            print('xy1')
            print(nY, nX)
            print(img1B.shape)
        
        img2=img1[sztile2:(img1.shape[0]-sztile2), sztile2:(img1.shape[1]-sztile2), :]
        
        img2A = img2.reshape(nY-1, sztile, nX-1, sztile, 3)
        img2B = img2A.transpose(0,2,1,3,4).reshape(-1,sztile,sztile,3)
        xy2=[np.array(ind2pos(ii, sztile=sztile, nX=nX-1))+sztile2  for ii in range(len(img2B))]
        if verbose:
            print('xy2')
            print(img2B.shape)
        
        img12B=np.r_[img1B, img2B]
        xy12=np.array(xy1+xy2)
        xy12.shape
        
        imask1d = np.pad(imask1, [[0, padBot], [0,padRt], [0,0]], mode='constant', constant_values=0)
        imask1e = imask1d.reshape(nY, sztile, nX, sztile,3)
        imask1f = imask1e.transpose(0,2,1,3,4).reshape(-1,sztile,sztile,3)
        
        imask2d = imask1d[sztile2:(imask1d.shape[0]-sztile2), sztile2:(imask1d.shape[1]-sztile2), :]
        imask2e = imask2d.reshape(nY-1, sztile, nX-1, sztile,3)
        imask2f = imask2e.transpose(0,2,1,3,4).reshape(-1,sztile,sztile,3)
        
        imask12f = np.r_[imask1f, imask2f]
        imask_area=(imask12f[:,:,:,0]>0).reshape(imask12f.shape[0], -1).sum(-1)
        
        img_inds=which(imask_area>=imask_cut1)
        
        if len(img_inds)<N0 and testing_mode:
            tmp_thr=select_large(imask_area, N0)
            if tmp_thr<imask_cut1c:
                tmp_thr=imask_cut1c
            img_inds=which(imask_area>=tmp_thr)
        elif len(img_inds)>Nmax:
            tmp_thr=select_large(imask_area[img_inds], Nmax, imask_cut1b)
            img_inds=which(imask_area>=tmp_thr)
        
        LTpos1C=xy12[img_inds, :]
        img1C=img12B[img_inds, :, :, :]
        imask1g=imask12f[img_inds, :, :, :]
        
        if verbose:
            print('select 1C by Area')
            print(N0, len(LTpos1C))
        
        img1Cnorm=[]
        for kkk in np.arange(len(img1C)):
            tmp_img=CN.img2normw2(img1C[kkk], LRw2)
            img1Cnorm.append(tmp_img)
        img1Cnorm=np.array(img1Cnorm)
        
        if len(img1C)>N1:
            img1C_Hsig=[]
            for ii in np.arange(len(img1Cnorm)):
                tmpH0=CN.im2stain(img1Cnorm[ii])[:,:,0]
                tmpH=tmpH0.reshape(-1)
                tmpmsk=imask1g[ii, :,:, 0].reshape(-1)
                tmpsig=1-(tmpH[tmpmsk>0]/255).mean()
                img1C_Hsig.append(tmpsig)
        
            img1C_Hsig=np.array(img1C_Hsig)
            cut_Hsig=select_large(img1C_Hsig, N1, 0.6)
            tmpind=which(img1C_Hsig>=cut_Hsig)
            img1D=img1Cnorm[tmpind, :, :, :]
            imask1h=imask1g[tmpind, :, :, :]
            LTpos1D=LTpos1C[tmpind, :]
            img1D_Hsig=img1C_Hsig[tmpind]
        else:
            img1D=img1Cnorm
            imask1h=imask1g
            LTpos1D=LTpos1C
            img1C_Hsig=None
        
        if verbose:
            print('select 1D by Signal')
            print(N1, len(LTpos1D))
        
        nTiles=len(img1D)
        if verbose:
            print('1D with normalized tiles')
            print(img1D.shape)
        
        if len(img1D)>0:
            img1D_datagen=InferenceDataGenerator(X=np.array(img1D)/255.0, batch_size=16)
            TN_scores=TN_model.predict_generator(img1D_datagen)
            #TN_scores=TN_model.predict(img1D/255.0)
            TN_scores=TN_scores.reshape(-1)
            TN_calls=np.round(TN_scores).reshape(-1)
            if verbose:
                print('\nTN')
                print(len(TN_calls))
                print(freq_table1(TN_calls.reshape(-1)))
            gc.collect()
            img1D_datagen=InferenceDataGenerator(X=np.array(img1D)/255.0, batch_size=16)
            GL_scores=GL_model.predict_generator(img1D_datagen)
            #GL_scores=GL_model.predict(img1D/255.0)
            GL_scores=GL_scores.reshape(-1)
            GL_calls=np.round(GL_scores).reshape(-1)
            if verbose:
                print('\nGL')
                print(len(GL_calls))
                print(freq_table1(GL_calls))
                print(freq_table2(TN_calls, GL_calls))
            gc.collect()
            
            #cut_TN=select_large(TN_scores, N2, 0.9)
            #sel_TN=which(TN_scores>=cut_TN)
            #cut_GL=select_large(GL_scores[sel_TN], N3, 4)
            cut_TN=0.4
            cut_GL=2.25
            sel_GL=which(np.logical_or(TN_scores>=cut_TN, GL_scores>=cut_GL))
            TNGL_scores_GL=TN_scores[sel_GL]*GL_scores[sel_GL]
            
            if testing_mode and len(sel_GL)<N3 and len(img1D)<len(img1Cnorm):
                ## Redo model predicting on more tiles
                if verbose:
                    print('Output too few tiles (%g), expand input from 1D (%g) to 1C (%g) : %s'%
                             (len(sel_GL), len(img1D), len(img1Cnorm), uuid))
                img1D=img1Cnorm
                imask1h=imask1g
                LTpos1D=LTpos1C
                TNGL_scores=img1C_Hsig
                
                img1D_datagen=InferenceDataGenerator(X=np.array(img1D)/255.0, batch_size=100)
                TN_scores=TN_model.predict_generator(img1D_datagen)
                #TN_scores=TN_model.predict(img1D/255.0)
                TN_scores=TN_scores.reshape(-1)
                TN_calls=np.round(TN_scores).reshape(-1)
                if verbose:
                    print('\nTN_2')
                    print(len(TN_calls))
                    print(freq_table1(TN_calls.reshape(-1)))
                gc.collect()
                img1D_datagen=InferenceDataGenerator(X=np.array(img1D)/255.0, batch_size=100)
                GL_scores=GL_model.predict_generator(img1D_datagen)
                #GL_scores=GL_model.predict(img1D/255.0)
                GL_scores=GL_scores.reshape(-1)
                GL_calls=np.round(GL_scores).reshape(-1)
                if verbose:
                    print('\nGL_2')
                    print(len(GL_calls))
                    print(freq_table1(GL_calls))
                    print(freq_table2(TN_calls, GL_calls))
                gc.collect()
                sel_GL=which(np.logical_or(TN_scores>=cut_TN, GL_scores>=cut_GL))
                if TNGL_scores is None:
                    TNGL_scores_GL=TN_scores[sel_GL]*GL_scores[sel_GL]
                else:
                    TNGL_scores_GL=TNGL_scores[sel_GL]
                if verbose:
                    print('Now %g candidate tiles.'%len(sel_GL))
            
            if len(sel_GL)>N3:
                cut_TNGL=select_large(TNGL_scores_GL, N3)
                sel_tmp=which(TNGL_scores_GL>=cut_TNGL)
                sel_GL=sel_GL[sel_tmp]
            
            TNcsv=os.path.join(TNdir, uuid+'.TN_GL.tsv')
            LTposYX=[str(yx) for yx in LTpos1D]
            ddf=pd.DataFrame(np.c_[np.round(TN_scores,3), np.round(GL_scores,2)], columns=['TN', 'GL'])
            ddf['(topY, leftX)']=LTposYX
            ddf.to_csv(TNcsv, sep='\t', index=False)
            if verbose:
                print("Save TN/GL scores to %s."%TNcsv)
            
            if verbose and len(sel_GL)>0:
                print(ddf.iloc[sel_GL].head())
                print('')
            
            img1G=img1D[sel_GL]
            LTpos1G_GL=LTpos1D[sel_GL]
            TN_scores_1G=TN_scores[sel_GL]
            GL_scores_1G=GL_scores[sel_GL]
            if verbose:
                print(img1G.shape)
        else:
            img1G=img1D  ## empty
            LTpos1G_GL=[]
            TN_scores_1G=[]
            GL_scores_1G=[]
        
        n1G=len(img1G)
        if verbose:
            print("n1G =", n1G)
        ##szhalf=int(sztile1*4-sztile0)//2
        if save_img:
            if output1_dir is not None:
                if n1G < N3min: #nTiles
                    UDIR1=os.path.join(output1_dir, uuid+"_X%03g"%nTiles)
                else:
                    UDIR1=os.path.join(output1_dir, uuid+"_H%03g"%nTiles)
                mkdir(UDIR1)
            
            if output0_dir is not None:
                if n1G < N3min: #nTiles
                    UDIR0=os.path.join(output0_dir, uuid+"_X%03g"%nTiles)
                else:
                    UDIR0=os.path.join(output0_dir, uuid+"_H%03g"%nTiles)
                mkdir(UDIR0)
        
        img0norm=[]
        if n1G>0 and output0_dir is not None:
            wsi=openslide.OpenSlide(ifn)
            for jj in np.arange(n1G):
                imm1=img1G[jj]
                _, dy, dx=img2bestcut(imm1)
                pY1, pX1=LTpos1G_GL[jj, :]
                pX2=int((pX1+dx)*4)
                pY2=int((pY1+dy)*4)
                TN1=TN_scores_1G[jj]
                GL1=GL_scores_1G[jj]
                if output1_dir is not None:
                    if save_img:
                        img1_fn=os.path.join(UDIR1, "%s.Lv%gSz%03g.i%03g.y%05g_x%05g_TN%5.3f_GL%4.2f.jpg"%(uuid, 1, sztile1, jj, pY1, pX1, TN1, GL1))
                        saveimg(img1_fn, imm1)
                
                if output0_dir is not None:
                    region0=np.asarray(wsi.read_region( [pX2, pY2], 0, [sztile0, sztile0] ))[:,:,:3]
                    region0norm=CN.img2normw2(region0, LRw2)
                    img0norm.append(region0norm)
                    if save_img:
                        img0_fn=os.path.join(UDIR0, "%s.Lv%gSz%03g.i%03g.y%05g_x%05g_TN%5.3f_GL%4.2f.jpg"%(uuid, 0, sztile0, jj, pY2, pX2, TN1, GL1))
                        saveimg(img0_fn, region0norm)
            
            wsi.close()
            if verbose:
                print('')
                print(uuid, len(img0norm))
        elif output1_dir is not None:
            for jj in np.arange(n1G):
                imm1=img1G[jj]
                pY1, pX1=LTpos1G_GL[jj, :]
                TN1=TN_scores_1G[jj]
                GL1=GL_scores_1G[jj]
                img1_fn=os.path.join(UDIR1, "%s.Lv%gSz%03g.i%03g.y%05g_x%05g_TN%5.3f_GL%4.2f.jpg"%(uuid, 1, sztile1, jj, pY1, pX1, TN1, GL1))
                saveimg(img1_fn, imm1)
            if verbose:
                print('')
                print(uuid, n1G)
        
        if return_img:
            final=(uuid, np.array(img1G), np.array(img0norm))
        return final
        ## End

def tr_uuid2tiles5(uuid, verbose=False, image_lvl=None, forced=False, return_img=False, save_img=True):
        gc.collect()
        gc.collect()
        if forced:
            ddd1=glob(os.path.join(TRJPG1B, uuid+'*'))
            ddd0=glob(os.path.join(TRJPG0B, uuid+'*'))
            for ddd in ddd1:
                force_rmdir(ddd)
            for ddd in ddd0:
                force_rmdir(ddd)
        if image_lvl is None or image_lvl==1:
            output1_dir=TRJPG1B
        else:
            output1_dir=None
        if image_lvl is None or image_lvl==0:
            output0_dir=TRJPG0B
        else:
            output0_dir=None
        final=uuid2tiles5(uuid, input_dir=TRAIN_IMAGES, output1_dir=output1_dir, output0_dir=output0_dir, 
                    LRdir=TRLRB, TNdir=TRTNB, ref_csv=train_csv, 
                    testing_mode=True, verbose=verbose, return_img=return_img)
        if image_lvl is None or image_lvl==1:
            print(uuid, len(os.listdir(TRJPG1B)))
        else:
            print(uuid, len(os.listdir(TRJPG0B)))
        return final

def te_uuid2tiles5(uuid, verbose=False, image_lvl=None, forced=False, return_img=False, save_img=True):
        gc.collect()
        gc.collect()
        if forced:
            ddd1=glob(os.path.join(TEJPG1B, uuid+'*'))
            ddd0=glob(os.path.join(TEJPG0B, uuid+'*'))
            for ddd in ddd1:
                force_rmdir(ddd)
            for ddd in ddd0:
                force_rmdir(ddd)
        if image_lvl is None or image_lvl==1:
            output1_dir=TEJPG1B
        else:
            output1_dir=None
        if image_lvl is None or image_lvl==0:
            output0_dir=TEJPG0B
        else:
            output0_dir=None
        final=uuid2tiles5(uuid, input_dir=TEST_IMAGES, output1_dir=output1_dir, output0_dir=output0_dir, 
                    LRdir=TELRB, TNdir=TETNB, ref_csv=test_csv, 
                    testing_mode=True, verbose=verbose, return_img=return_img)
        if image_lvl is None or image_lvl==1:
            print(uuid, len(os.listdir(TEJPG1B)))
        else:
            print(uuid, len(os.listdir(TEJPG0B)))
        return final

###

print('Load previous GL and TN models')
sys.stdout.flush()

GL_model=load_model(GL_model_fn)
TN_model=load_model(TN_model_fn)

print('GL and TN models loaded')
sys.stdout.flush()

###

SEED=123456
import numpy as np
import random
set_seed_all(SEED, verbose=True)

####

INPUT_DIR1='../input'
INPUT_DIR1=os.path.expanduser(INPUT_DIR1)
INPUT_DIR=os.path.join(INPUT_DIR1, 'prostate-cancer-grade-assessment')
print(os.path.expanduser(INPUT_DIR))

##

TRAIN_annot=os.path.join(INPUT_DIR, 'train.csv')
TEST_annot=os.path.join(INPUT_DIR, 'test.csv')
TRAIN_IMAGES=os.path.join(INPUT_DIR, 'train_images')
TRAIN_MASKS=os.path.join(INPUT_DIR, 'train_label_masks')

TEST_IMAGES=os.path.join(INPUT_DIR, 'test_images')

model_tag='kaggle_PANDA'

sztile0=256
sztile1=128

TRJPG1B='train_jpgs_L%g_sz%g_V2'%(1, sztile1)
TEJPG1B='test_jpgs_L%g_sz%g_V2'%(1, sztile1)

TRJPG0B='train_jpgs_L%g_sz%g_V2'%(0, sztile0)
TEJPG0B='test_jpgs_L%g_sz%g_V2'%(0, sztile0)

TRLRB='train_color_LR_V2'
TELRB='test_color_LR_V2'

TRTNB='train_TNGL_V2'
TETNB='test_TNGL_V2'

###

print(TRAIN_IMAGES, dir_exists(TRAIN_IMAGES))
print(TRAIN_MASKS, dir_exists(TRAIN_MASKS))
print(TEST_IMAGES, dir_exists(TEST_IMAGES))

print(TRAIN_annot, file_readable(TRAIN_annot))
print(TEST_annot, file_readable(TEST_annot))

##

train_csv0 = pd.read_csv(TRAIN_annot)
train_uuids0=train_csv0['image_id'].values
print(train_csv0.head(3))

##

test_csv = pd.read_csv(TEST_annot)
test_uuids=test_csv['image_id'].values
print('')
print(test_csv.head())
print(test_uuids)

##

Random_testing=False

if not dir_exists(TEST_IMAGES):
    TEST_IMAGES=os.path.join('.', 'test_images')
    mkdir(TEST_IMAGES)
    print('\n', 'Chanage test_images folder:', TEST_IMAGES, '\n')
    Random_testing=True

from shutil import copyfile

if len(os.listdir(TEST_IMAGES))==0 and Random_testing:
    for i in np.arange(len(test_uuids)):
        uuid=test_uuids[i]
        imgfile=os.path.join(TEST_IMAGES, uuid+'.tiff')
        print(i, uuid, file_readable(imgfile), imgfile)
        if not file_readable(imgfile):
            simulated_file=os.path.join(TRAIN_IMAGES, train_uuids0[i]+'.tiff')
            print('from', simulated_file)
            copyfile(simulated_file, imgfile)
            print(i, uuid, file_readable(imgfile), imgfile)
            print('\n\n')

###

if TRAIN_step2_flag:
    train_images=[os.path.basename(ff) for ff in glob(os.path.join(TRAIN_IMAGES, '*.tiff'))]
    train_masks=[os.path.basename(ff) for ff in glob(os.path.join(TRAIN_MASKS, '*.tiff'))]
    train_images.sort()
    train_masks.sort()
    print(len(train_images), len(train_masks))
    
    train_images_uuids=[re.sub('.tiff$', '', ff) for ff in train_images]
    train_masks_uuids=[re.sub('_mask.tiff$', '', ff) for ff in train_masks]
    
    train_both_uuids=[x for x in train_images_uuids if x in train_masks_uuids]
    print(len(train_both_uuids), train_both_uuids[0])
    
    ###
    
    train_valid_uuids=[x for x in train_both_uuids if x in train_uuids0]
    print(len(train_valid_uuids), train_valid_uuids[0])
    
    ind=[ii for ii in range(len(train_uuids0)) if train_uuids0[ii] in train_valid_uuids]
    train_csv = train_csv0.iloc[ind]
    train_work_uuids=train_csv['image_id'].values
    print(train_csv.shape)
    
    if work_lvl is None or work_lvl==1:
        mkdir(TRJPG1B)
    if work_lvl is None or work_lvl==0:
        mkdir(TRJPG0B)
    mkdir(TRLRB)
    mkdir(TRTNB)
    print(TRAIN_IMAGES)
    
    w_uuids=train_work_uuids
    
    iii=0
    for iii in np.arange(iii, len(w_uuids)):
        uuid=str(w_uuids[iii])
        print(  '%g / %g: %s' % (iii, len(w_uuids), uuid )  )
        try:
            _ = tr_uuid2tiles5(uuid, verbose=False, forced=False, image_lvl=work_lvl)
        except:
            print("iii==%g, TRAIN uuid %s failed."%(iii, uuid))
        gc.collect()
        gc.collect()

###

if TEST_step2_flag:
    test_images=[os.path.basename(ff) for ff in glob(os.path.join(TEST_IMAGES, '*.tiff'))]
    test_images.sort()
    print( len(test_images))
    
    test_images_uuids=[re.sub('.tiff$', '', ff) for ff in test_images]
    
    if work_lvl is None or work_lvl==1:
        mkdir(TEJPG1B)
    if work_lvl is None or work_lvl==0:
        mkdir(TEJPG0B)
    mkdir(TELRB)
    mkdir(TETNB)
    print(TEST_IMAGES)
    
    w_uuids=test_uuids
    
    iii=0
    for iii in np.arange(iii, len(w_uuids)):
        uuid=str(w_uuids[iii])
        print(  '%g / %g: %s' % (iii, len(w_uuids), uuid )  )
        try:
            _ = te_uuid2tiles5(uuid, verbose=False, forced=False, image_lvl=work_lvl)
        except:
            print("iii==%g, TEST uuid %s failed."%(iii, uuid))
        gc.collect()
        gc.collect()

del(TN_model)
del(GL_model)
gc.collect()
gc.collect()

print("prepare tiles done")

