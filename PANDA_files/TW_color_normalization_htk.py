## Most codes from HistomicsTK
## By TW on 2020-06-14

import os
import sys
import collections
import subprocess
import re
import numpy as np
import scipy
import scipy.ndimage.morphology
import cv2
import matplotlib.pyplot as plt
from  sklearn.linear_model import LinearRegression

# import_module1('~/TW_python/TW_color_normalization_htk.py', 'CN')

def import_module1(pkgname, return_module=False):
	#import os
	#import sys
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

import subprocess
import sys
def install_package1(package, force=False, upgrade=True):
    #import subprocess
    #import sys
    tmplst=[sys.executable, "-m", "pip", "install"]
    if force:
        tmplst.append('-y')
    if upgrade:
        tmplst.append('-U')
    tmplst.append(package)
    subprocess.check_call(tmplst)

#######

W_target = np.array([
    [0.5807549,  0.08314027,  0.08213795],
    [0.71681094,  0.90081588,  0.41999816],
    [0.38588316,  0.42616716, -0.90380025]
])

stain_unmixing_routine_params = {
    'stains': ['hematoxylin', 'eosin'],
    'stain_unmixing_method': 'macenko_pca',
}

stain_color_map = {
    'hematoxylin': [0.65, 0.70, 0.29],
    'eosin':       [0.07, 0.99, 0.11],
    'dab':         [0.27, 0.57, 0.78],
    'null':        [0.0, 0.0, 0.0]
}

W_im2stain=np.array([stain_color_map['hematoxylin'],
                     stain_color_map['eosin'],
                     stain_color_map['null']]).T

cnorm = {
    'mu': np.array([8.74108109, -0.12440419,  0.0444982]),
    'sigma': np.array([0.6135447, 0.10989545, 0.0286032]),
}

_rgb2lms = np.array([[0.3811, 0.5783, 0.0402],
                     [0.1967, 0.7244, 0.0782],
                     [0.0241, 0.1288, 0.8444]])

_lms2lab = np.dot(
    np.array([[1 / (3**0.5), 0, 0],
              [0, 1 / (6**0.5), 0],
              [0, 0, 1 / (2**0.5)]]),
    np.array([[1, 1, 1],
              [1, 1, -2],
              [1, -1, 0]])
)

_lms2rgb = np.linalg.inv(_rgb2lms)
_lab2lms = np.linalg.inv(_lms2lab)

def TW_normalize_LR(img_src, mask_excl=None):
    img_reinhard = reinhard(
        img_src, mask_out=mask_excl)
    
    img_reinhard_deconv = deconvolution_based_normalization(
                img_reinhard, mask_out=mask_excl)
    
    LR_RGB=image_normalization_LR(img_src, img_reinhard_deconv, mask_excl=mask_excl)
    
    return LR_RGB

def TW_normalize_excl(img, mask_excl=None):
    imgw=np.float32(image255(img))
    if isinstance(mask_excl, bool) and mask_excl==False:
        mask_incl=None
    else:
        if isinstance(mask_excl, bool) and mask_excl==True:
            mask_excl=None
        if mask_excl is None:
            mask_excl=mask_white(imgw, min_rgb=180, max_diff=70, erosion=3)
        assert np.all(imgw.shape[:2]==mask_excl.shape), "Mask_exclusive has different size"
    
    img_reinhard = reinhard(
        imgw, mask_out=mask_excl)
    
    img_reinhard_deconv = deconvolution_based_normalization(
                img_reinhard, mask_out=mask_excl)
    
    return img_reinhard_deconv

def TW_normalize_excl2(img, mask_excl=None, min_rgb=180, max_diff=70):
    imgw=np.float32(image255(img))
    if isinstance(mask_excl, bool) and mask_excl==False:
        mask_incl=None
    else:
        if isinstance(mask_excl, bool) and mask_excl==True:
            mask_excl=None
        if mask_excl is None:
            mask_excl=mask_white(imgw, min_rgb=min_rgb, max_diff=max_diff, erosion=3)
        assert np.all(imgw.shape[:2]==mask_excl.shape), "Mask_exclusive has different size"
    
    img_reinhard = reinhard(
        imgw, mask_out=mask_excl)
    
    img_reinhard_deconv = deconvolution_based_normalization(
                img_reinhard, mask_out=mask_excl)
    
    if mask_excl is not None:
        img_reinhard_deconv[:,:,0][mask_excl]=255
        img_reinhard_deconv[:,:,1][mask_excl]=255
        img_reinhard_deconv[:,:,2][mask_excl]=255
    return img_reinhard_deconv

def img2norm(img, LR_RGB):
    #LR_RGB=[ [inteR, coefR], [inteG, coefG], [inteB, coefB] ]
    
    imgw=np.float32(image255(img))
    im10norm=imgw*0.0
    im10norm[:,:,0]=imgw[:,:,0]*LR_RGB[0][1] + LR_RGB[0][0]
    im10norm[:,:,1]=imgw[:,:,1]*LR_RGB[1][1] + LR_RGB[1][0]
    im10norm[:,:,2]=imgw[:,:,2]*LR_RGB[2][1] + LR_RGB[2][0]
    im10norm[im10norm<0]=0
    im10norm[im10norm>255]=255
    im10norm=im10norm.astype(np.uint8)
    return(im10norm)

def img2normLR_excl(img, LR_RGB, mask_excl=None):
    imgw=np.float32(image255(img))
    newdim=np.prod(imgw.shape[:2])
    if isinstance(mask_excl, bool) and mask_excl==False:
        mask_incl=np.full((newdim), True)
    else:
        if isinstance(mask_excl, bool) and mask_excl==True:
            mask_excl=None
        if mask_excl is None:
            mask_excl=mask_white(imgw)
        assert np.all(img.shape[:2]==mask_excl.shape), "Mask_exclusive has different size"
        mask_incl=np.logical_not(mask_excl.reshape(newdim))
    
    imgw2=imgw.reshape((newdim,3))
    im10norm=np.full((newdim,3), 255.0)
    im10norm[mask_incl,0]=imgw2[mask_incl,0]*LR_RGB[0][1] + LR_RGB[0][0]
    im10norm[mask_incl,1]=imgw2[mask_incl,1]*LR_RGB[1][1] + LR_RGB[1][0]
    im10norm[mask_incl,2]=imgw2[mask_incl,2]*LR_RGB[2][1] + LR_RGB[2][0]
    im10norm[im10norm<0]=0
    im10norm[im10norm>255]=255
    im10norm=im10norm.astype(np.uint8)
    return(im10norm.reshape(img.shape))

def reinhard(
        im_src, target_mu=cnorm['mu'], target_sigma=cnorm['sigma'], 
        src_mu=None, src_sigma=None, mask_out=None):
    
    # convert input image to LAB color space
    im_lab = rgb_to_lab(im_src)
    
    # mask out irrelevant tissue / whitespace / etc
    if mask_out is not None:
        mask_out = mask_out[..., None]
        im_lab = np.ma.masked_array(
            im_lab, mask=np.tile(mask_out, (1, 1, 3)))
    
    # calculate src_mu and src_sigma if either is not provided
    if (src_mu is None) or (src_sigma is None):
        src_mu = [im_lab[..., i].mean() for i in range(3)]
        src_sigma = [im_lab[..., i].std() for i in range(3)]
    
    # scale to unit variance
    for i in range(3):
        im_lab[:, :, i] = (im_lab[:, :, i] - src_mu[i]) / src_sigma[i]
    
    # rescale and recenter to match target statistics
    for i in range(3):
        im_lab[:, :, i] = im_lab[:, :, i] * target_sigma[i] + target_mu[i]
    
    # convert back to RGB colorspace
    im_normalized = lab_to_rgb(im_lab)
    im_normalized[im_normalized > 255] = 255
    im_normalized[im_normalized < 0] = 0
    
    # return masked values and reconstruct unmasked LAB image
    if mask_out is not None:
        im_normalized = im_normalized.data
        for i in range(3):
            original = im_src[:, :, i].copy()
            new = im_normalized[:, :, i].copy()
            original[np.not_equal(mask_out[:, :, 0], True)] = 0
            new[mask_out[:, :, 0]] = 0
            im_normalized[:, :, i] = new + original
    
    im_normalized = im_normalized.astype(np.uint8)
    
    return im_normalized

def deconvolution_based_normalization(
        im_src, W_source=None, im_target=None,
        stains=['hematoxylin', 'eosin'], 
        W_target=W_target,
        stain_unmixing_routine_params=stain_unmixing_routine_params,
        mask_out=None):
    #Perform color normalization using color deconvolution to transform the.
    
    for k in ['W_source', 'mask_out']:
        assert k not in stain_unmixing_routine_params.keys(), \
            "%s must be provided as a separate parameter." % k
    
    # find stains matrix from source image
    stain_unmixing_routine_params['stains'] = stains
    _, StainsFloat, _ = color_deconvolution_routine(
        im_src, W_source=W_source, mask_out=mask_out,
        **stain_unmixing_routine_params)
    
    # Get W_target
    
    if all(j is None for j in [W_target, im_target]):
        # Normalize to 'ideal' stain matrix if none is provided
        W_target = np.array(
            [stain_color_map[stains[0]], stain_color_map[stains[1]]]).T
        W_target = complement_stain_matrix(W_target)
    
    elif im_target is not None:
        # Get W_target from target image
        W_target = stain_unmixing_routine(
            im_target, **stain_unmixing_routine_params)
    
    # Convolve source image StainsFloat with W_target
    im_src_normalized = color_convolution(StainsFloat, W_target)
    
    # return masked values using unnormalized image
    if mask_out is not None:
        keep_mask = np.not_equal(mask_out, True)
        for i in range(3):
            original = im_src[:, :, i].copy()
            new = im_src_normalized[:, :, i].copy()
            original[keep_mask] = 0
            new[mask_out] = 0
            im_src_normalized[:, :, i] = new + original
    
    return im_src_normalized

def rgb_to_sda(im_rgb, I_0, allow_negatives=False):
    # Transform input RGB image or matrix `im_rgb` into SDA (stain
    # darkness) space for color deconvolution.
    
    is_matrix = im_rgb.ndim == 2
    if is_matrix:
        im_rgb = im_rgb.T
    
    if I_0 is None:  # rgb_to_od compatibility
        im_rgb = im_rgb.astype(float) + 1
        I_0 = 256
    
    if not allow_negatives:
        im_rgb = np.minimum(im_rgb, I_0)
    
    im_sda = -np.log(im_rgb/(1.*I_0)) * 255/np.log(I_0)
    return im_sda.T if is_matrix else im_sda

def sda_to_rgb(im_sda, I_0):
    ## Transform input SDA image or matrix `im_sda` into RGB space.  This
    ## is the inverse of `rgb_to_sda` with respect to the first parameter
    
    is_matrix = im_sda.ndim == 2
    if is_matrix:
        im_sda = im_sda.T
    
    od = I_0 is None
    if od:  # od_to_rgb compatibility
        I_0 = 256
    
    im_rgb = I_0 ** (1 - im_sda / 255.)
    return (im_rgb.T if is_matrix else im_rgb) - od

def lab_to_rgb(im_lab):
    #Transforms an image from LAB to RGB color space
    
    # get input image dimensions
    m = im_lab.shape[0]
    n = im_lab.shape[1]
    
    # calculate im_lms values from LAB
    im_lab = np.reshape(im_lab, (m * n, 3))
    im_lms = np.dot(_lab2lms, np.transpose(im_lab))
    
    # calculate RGB values from im_lms
    im_lms = np.exp(im_lms)
    im_lms[im_lms == np.spacing(1)] = 0
    
    im_rgb = np.dot(_lms2rgb, im_lms)
    
    # reshape to 3-channel image
    im_rgb = np.reshape(im_rgb.transpose(), (m, n, 3))
    
    return im_rgb

def rgb_to_lab(im_rgb):
    #Transforms an image from RGB to LAB color space
    
    # get input image dimensions
    m = im_rgb.shape[0]
    n = im_rgb.shape[1]
    
    # calculate im_lms values from RGB
    im_rgb = np.reshape(im_rgb, (m * n, 3))
    im_lms = np.dot(_rgb2lms, np.transpose(im_rgb))
    im_lms[im_lms == 0] = np.spacing(1)
    
    # calculate LAB values from im_lms
    im_lab = np.dot(_lms2lab, np.log(im_lms))
    
    # reshape to 3-channel image
    im_lab = np.reshape(im_lab.transpose(), (m, n, 3))
    
    return im_lab

def lab_mean_std(im_input, mask_out=None):
    ## Compute the mean and standard deviation of the intensities.
    im_lab = rgb_to_lab(im_input)
    
    # mask out irrelevant tissue / whitespace / etc
    if mask_out is not None:
        mask_out = mask_out[..., None]
        im_lab = np.ma.masked_array(
            im_lab, mask=np.tile(mask_out, (1, 1, 3)))
    
    mean_lab = np.array([im_lab[..., i].mean() for i in range(3)])
    std_lab = np.array([im_lab[..., i].std() for i in range(3)])
    
    return mean_lab, std_lab

def convert_matrix_to_image(m, shape):
    # Convert a column matrix of pixels to a 3D image given by shape.
    
    if len(shape) == 2:
        return m
    
    return m.T.reshape(shape[:-1] + (m.shape[0],))

def convert_image_to_matrix(im):
    # Convert an image (MxNx3 array) to a column matrix of pixels
    # (3x(M*N)).  It will pass through a 2D array unchanged.
    
    if im.ndim == 2:
        return im
    
    return im.reshape((-1, im.shape[-1])).T

def color_convolution(im_stains, w, I_0=None):
    #Perform Color Convolution.
    
    # transform 3D input stain image to 2D stain matrix format
    m = convert_image_to_matrix(im_stains)
    
    # transform input stains to optical density values, convolve and
    # tfm back to stain
    sda_fwd = rgb_to_sda(m, 255 if I_0 is not None else None,
                                          allow_negatives=True)
    sda_conv = np.dot(w, sda_fwd)
    sda_inv = sda_to_rgb(sda_conv, I_0)
    
    # reshape output, transform type
    im_rgb = (convert_matrix_to_image(sda_inv, im_stains.shape)
              .clip(0, 255).astype(np.uint8))
    
    return im_rgb

def color_deconvolution(im_rgb, w, I_0=None):
    ## Perform color deconvolution.
    # complement stain matrix if needed
    if np.linalg.norm(w[:, 2]) <= 1e-16:
        wc = complement_stain_matrix(w)
    else:
        wc = w
    
    # normalize stains to unit-norm
    wc = htk_normalize(wc)
    
    # invert stain matrix
    Q = np.linalg.inv(wc)
    
    # transform 3D input image to 2D RGB matrix format
    m = convert_image_to_matrix(im_rgb)[:3]
    
    # transform input RGB to optical density values and deconvolve,
    # tfm back to RGB
    sda_fwd = rgb_to_sda(m, I_0)
    sda_deconv = np.dot(Q, sda_fwd)
    sda_inv = sda_to_rgb(sda_deconv,
                         255 if I_0 is not None else None)
    
    # reshape output
    StainsFloat = convert_matrix_to_image(sda_inv, im_rgb.shape)
    
    # transform type
    Stains = StainsFloat.clip(0, 255).astype(np.uint8)
    
    # return
    Unmixed = collections.namedtuple('Unmixed',
                                     ['Stains', 'StainsFloat', 'Wc'])
    Output = Unmixed(Stains, StainsFloat, wc)
    
    return Output

def complement_stain_matrix(w):
    #Generates a complemented stain matrix
    
    stain0 = w[:, 0]
    stain1 = w[:, 1]
    stain2 = np.cross(stain0, stain1)
    # Normalize new vector to have unit norm
    return np.array([stain0, stain1, stain2 / np.linalg.norm(stain2)]).T

def get_principal_components(m):
    ## Take a matrix m (probably 3xN) and return the 3x3 matrix of
    ## "principal components".  (Actually computed with SVD)
    
    return np.linalg.svd(m.astype(float), full_matrices=False)[0].astype(m.dtype)

def magnitude(m):
    ## Get the magnitude of each column vector in a matrix
    return np.sqrt((m ** 2).sum(0))

def htk_normalize(m):
    ## Normalize each column vector in a matrix
    return m / magnitude(m)


def find_stain_index(reference, w):
    ## Find the index of the stain column vector in w corresponding to the
    ## reference vector.
    
    dot_products = np.dot(reference, w)
    return np.argmax(np.abs(dot_products))

def rgb_separate_stains_macenko_pca(im_rgb, I_0, *args, **kwargs):
    ## Compute the stain matrix for color deconvolution with the "Macenko"
    ## method from an RGB image or matrix.
    
    im_sda = rgb_to_sda(im_rgb, I_0)
    return separate_stains_macenko_pca(im_sda, *args, **kwargs)

def exclude_nonfinite(m):
    ## Exclude columns from m that have infinities or nans. 
    
    return m[:, np.isfinite(m).all(axis=0)]

import nimfa
def separate_stains_xu_snmf(im_sda, w_init=None, beta=0.2):
    #import nimfa
    ## Compute the stain matrix for color deconvolution with SNMF.
    
    # Image matrix
    m = convert_image_to_matrix(im_sda)
    m = exclude_nonfinite(m)
    factorization = \
        nimfa.Snmf(m, rank=m.shape[0] if w_init is None else w_init.shape[1],
                   W=w_init,
                   H=None if w_init is None else np.linalg.pinv(w_init).dot(m),
                   beta=beta)
    factorization.factorize()
    return htk_normalize(np.array(factorization.W))

def rgb_separate_stains_xu_snmf(im_rgb, I_0, *args, **kwargs):
    ## Compute the stain matrix for color deconvolution with SNMF from an
    ## RGB image or matrix.
    
    im_sda = rgb_to_sda(im_rgb, I_0)
    return separate_stains_xu_snmf(im_sda, *args, **kwargs)

def _reorder_stains(W, stains=['hematoxylin', 'eosin']):
    ### Reorder stains in a stain matrix to a specific order.
    
    assert len(stains) == 2, "Only two-stain matrices are supported for now."
    
    def _get_channel_order(W):
        first = find_stain_index(stain_color_map[stains[0]], W)
        second = 1 - first
        # If 2 stains, third "stain" is cross product of 1st 2 channels
        # calculated using complement_stain_matrix()
        third = 2
        return first, second, third
    
    def _ordered_stack(mat, order):
        return np.stack([mat[..., j] for j in order], -1)
    
    return _ordered_stack(W, _get_channel_order(W))

def stain_unmixing_routine(
        im_rgb, stains=['hematoxylin', 'eosin'],
        stain_unmixing_method='macenko_pca',
        stain_unmixing_params={}, mask_out=None):
    ## Perform stain unmixing using the method of choice (wrapper).
    stain_unmixing_method = stain_unmixing_method.lower()
    
    if stain_unmixing_method == 'macenko_pca':
        stain_deconvolution = rgb_separate_stains_macenko_pca
        stain_unmixing_params['I_0'] = None
        stain_unmixing_params['mask_out'] = mask_out
    
    elif stain_unmixing_method == 'xu_snmf':
        stain_deconvolution = rgb_separate_stains_xu_snmf
        stain_unmixing_params['I_0'] = None
        assert mask_out is None, "Masking is not yet implemented in xu_snmf."
    
    else:
        raise ValueError("Unknown/Unimplemented deconvolution method.")
    
    # get W_source
    W_source = stain_deconvolution(im_rgb, **stain_unmixing_params)
    
    if stain_unmixing_method == 'macenko_pca':
        W_source = _reorder_stains(W_source, stains=stains)
    
    return W_source

def color_deconvolution_routine(
        im_rgb, W_source=None, mask_out=None, **kwargs):
    ## Unmix stains mixing followed by deconvolution (wrapper).
    
    # get W_source if not provided
    if W_source is None:
        W_source = stain_unmixing_routine(im_rgb, mask_out=mask_out, **kwargs)
    
    # deconvolve
    Stains, StainsFloat, wc = color_deconvolution(im_rgb, w=W_source, I_0=None)
    
    # mask out (keep in mind, image is inverted)
    if mask_out is not None:
        for i in range(3):
            Stains[..., i][mask_out] = 255
            StainsFloat[..., i][mask_out] = 255.
    
    return Stains, StainsFloat, wc

def separate_stains_macenko_pca(
        im_sda, minimum_magnitude=16, min_angle_percentile=0.01,
        max_angle_percentile=0.99, mask_out=None):
    ## Compute the stain matrix for color deconvolution with the Macenko method.
    
    # Image matrix
    m = convert_image_to_matrix(im_sda)
    
    # mask out irrelevant values
    if mask_out is not None:
        keep_mask = np.equal(mask_out[..., None], False)
        keep_mask = np.tile(keep_mask, (1, 1, 3))
        keep_mask = convert_image_to_matrix(keep_mask)
        m = m[:, keep_mask.all(axis=0)]
    
    # get rid of NANs and infinities
    m = exclude_nonfinite(m)
    
    # Principal components matrix
    pcs = get_principal_components(m)
    # Input pixels projected into the PCA plane
    proj = pcs.T[:-1].dot(m)
    # Pixels above the magnitude threshold
    filt = proj[:, magnitude(proj) > minimum_magnitude]
    # The "angles"
    angles = _get_angles(filt)
    
    # The stain vectors
    def get_percentile_vector(p):
        return pcs[:, :-1].dot(filt[:, argpercentile(angles, p)])
    
    min_v = get_percentile_vector(min_angle_percentile)
    max_v = get_percentile_vector(max_angle_percentile)
    
    # The stain matrix
    w = complement_stain_matrix(htk_normalize(
        np.array([min_v, max_v]).T))
    return w

def _get_angles(m):
    ## Take a 2xN matrix of vectors and return a length-N array of an.
    
    m = htk_normalize(m)
    # "Angle" towards +x from the +y axis
    return (1 - m[1]) * np.sign(m[0])

def argpercentile(arr, p):
    ## Calculate index in arr of element nearest the pth percentile.
    # Index corresponding to percentile
    i = int(p * arr.size + 0.5)
    return np.argpartition(arr, i)[i]

def image_normalization_LR(img_src, img_targ, mask_excl=None, mask_incl=None):
    assert np.all(img_src.shape[:2]==img_targ.shape[:2]), "Source and normalized imaged have different size"
    if mask_excl is not None:
        assert np.all(img_src.shape[:2]==mask_excl.shape), "Mask_exclude has different size"
    
    if mask_incl is not None:
        assert np.all(img_src.shape[:2]==mask_incl.shape), "Mask_include has different size"
    
    newdim=np.prod(img_src.shape[:2])
    
    thumb_arr3=np.float32(image255(img_src)).reshape((newdim, 3))
    norm2_arr3=np.float32(image255(img_targ)).reshape((newdim, 3))
    
    if mask_incl is None and mask_excl is None:
        mask_incl2=np.ones(newdim)>0
    elif mask_excl is None:
        mask_incl2=(mask_incl>0).reshape(newdim)
    else:
        mask_incl2=np.logical_not(mask_excl).reshape(newdim)
    
    thumb_arr4=thumb_arr3[mask_incl2, :]
    norm2_arr4=norm2_arr3[mask_incl2, :]
    
    regressorR = LinearRegression()  
    regressorR.fit(thumb_arr4[:,0:1], norm2_arr4[:,0:1])
    regressorG = LinearRegression()  
    regressorG.fit(thumb_arr4[:,1:2], norm2_arr4[:,1:2])
    regressorB = LinearRegression()  
    regressorB.fit(thumb_arr4[:,2:3], norm2_arr4[:,2:3])
    
    inteR=regressorR.intercept_[0]
    coefR=regressorR.coef_[0][0]
    inteG=regressorG.intercept_[0]
    coefG=regressorG.coef_[0][0]
    inteB=regressorB.intercept_[0]
    coefB=regressorB.coef_[0][0]
    
    LR_RGB=[ [inteR, coefR], [inteG, coefG], [inteB, coefB] ]
    LR_RGB=np.array(LR_RGB)
    return(LR_RGB)

def im2stain(im0, W=W_im2stain):
    # perform standard color deconvolution
    im_stains = color_deconvolution(im0, W).Stains
    return(im_stains)

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

def mask_white(imgw, min_rgb=200, max_diff=55, erosion=3):
    assert imgw.ndim==3, 'must be a 3-d image array'
    imgw=image255(imgw)
    rgb_min=np.float32(np.min(imgw, axis=2))
    rgb_max=np.float32(np.max(imgw, axis=2))
    msk=np.logical_and(rgb_min>=min_rgb, (rgb_max-rgb_min)<=max_diff)
    if erosion>0:
        msk=scipy.ndimage.morphology.binary_erosion(msk, iterations=erosion)
        msk=scipy.ndimage.morphology.binary_dilation(msk, iterations=erosion)
    return(msk)

import cv2
import matplotlib.pyplot as plt
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

def is_true(v):
    return(isinstance(v, bool) and v==True)

def is_false(v):
    return(isinstance(v, bool) and v==False)

def is_string(v):
    return(isinstance(v, str))

from sklearn.linear_model import LinearRegression
def image_norma_LR2(img_src, img_targ, mask_excl=None, mask_incl=None, 
            min_rgb=200, max_diff=55, erosion=3):
    
    if isinstance(img_src, str):
        img_src=plt.imread(os.path.expanduser(img_src))
    
    if isinstance(img_targ, str):
        img_targ=plt.imread(os.path.expanduser(img_targ))
    
    assert np.all(img_src.shape[:2]==img_targ.shape[:2]), "Source and normalized imaged have different size"
    
    if is_true(mask_incl):
        mask_incl=np.logical_not(mask_white(img_src, min_rgb=min_rgb, max_diff=max_diff, erosion=erosion))
    if is_true(mask_excl):
        mask_excl=mask_white(img_src, min_rgb=min_rgb, max_diff=max_diff, erosion=erosion)
    
    if mask_excl is not None:
        assert np.all(img_src.shape[:2]==mask_excl.shape), "Mask_exclude has different size"
    
    if mask_incl is not None:
        assert np.all(img_src.shape[:2]==mask_incl.shape), "Mask_include has different size"
    
    newdim=np.prod(img_src.shape[:2])
    
    thumb_arr3=np.float32(image255(img_src)).reshape((newdim, 3))
    norm2_arr3=np.float32(image255(img_targ)).reshape((newdim, 3))
    
    if mask_incl is None and mask_excl is None:
        mask_regr=np.full(newdim, True)
    elif mask_excl is None:
        mask_regr=(mask_incl>0).reshape(newdim)
    elif mask_incl is None:
        mask_regr=np.logical_not(mask_excl).reshape(newdim)
    else:
        mask_regr=np.logical_and(mask_incl, np.logical_not(mask_excl)).reshape(newdim)
    
    thumb_arr4=255.0-thumb_arr3[mask_regr, :]
    norm2_arr4=255.0-norm2_arr3[mask_regr, :]
    #thumb_arr5=np.c_[thumb_arr4, np.square(thumb_arr4)]
    regressor2 = LinearRegression(fit_intercept=False)
    regressor2.fit(thumb_arr4, norm2_arr4)
    
    LR_RGBw=regressor2.coef_.T
    return(LR_RGBw)

def img2normw2(img, LR_RGBw):
    imgw=255.0-np.float32(image255(img))
    imgw3=imgw.reshape(-1, 3)
    im10norm=np.matmul(imgw3, LR_RGBw)
    im10norm[im10norm<0]=0
    im10norm[im10norm>255]=255
    im10norm2=(255.0-im10norm).reshape(img.shape).astype(np.uint8)    
    return(im10norm2)
