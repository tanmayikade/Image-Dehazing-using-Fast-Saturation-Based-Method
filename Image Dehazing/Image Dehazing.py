import numpy as np
import cv2
import time
import pandas as pd
from matplotlib import pyplot as plt
from skimage import morphology
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

'''
   pixel value: i2f = 0-255 to 0-1
                f2i = 0-1 to 0-255 
'''


def i2f(i_image):
    f_image = np.float32(i_image) / 255.0
    return f_image


def f2i(f_image):
    i_image = np.uint8(f_image * 255.0)
    return i_image


'''
    Compute 'A' as described by Tang et al. (CVPR 2014)
'''


def Compute_A_Tang(im):
    erosion_window = 5
    n_bins = 200  # for histogram
    R = im[:, :, 2]
    G = im[:, :, 1]
    B = im[:, :, 0]

    '''
    Morphological erosion sets a pixel at (i,j) to the minimum over all pixels in the neighborhood centered at
    (i,j). Erosion shrinks bright regions and enlarges dark regions.
    
    The Numpy histogram function is similar to the hist() function of matplotlib library, 
    the only difference is that the Numpy histogram gives the numerical representation of the dataset
    while the hist() gives graphical representation of the dataset.
    
    Compute the dark channel

    '''
    dark = morphology.erosion(np.min(im, 2), morphology.square(erosion_window))
    [h, edges] = np.histogram(dark, n_bins)
    numpixel = im.shape[0] * im.shape[1]  # calculate number of pixels
    thr_frac = numpixel * 0.99
    csum = np.cumsum(h)  # returns cumulative sum of elements upto that element
    nz_idx = np.nonzero(csum > thr_frac)[0][0]
    dc_thr = edges[nz_idx]
    mask = dark >= dc_thr
    # similar to DCP till this step
    # next, median of these top 0.1% pixels
    # median of the RGB values of the pixels in the mask
    rs = R[mask]
    gs = G[mask]
    bs = B[mask]

    A = np.zeros((1, 3))

    A[0, 2] = np.median(rs)
    A[0, 1] = np.median(gs)
    A[0, 0] = np.median(bs)

    return A


'''
    Compute intensity: GetIntensity, and Saturation: GetSauration
'''


def GetIntensity(fi):
    return cv2.divide(fi[:, :, 0] + fi[:, :, 1] + fi[:, :, 2], 3)


def GetSaturation(fi, intensity):
    """
     This saturation is calculated using formula. Refer documentation.
    """
    min_rgb = cv2.min(cv2.min(fi[:, :, 0], fi[:, :, 1]), fi[:, :, 2])
    me = np.finfo(np.float32).eps
    S = 1.0 - min_rgb / (intensity + me)
    return S


'''
    Estimating saturation of scene radiance or haze-free image output. 
    The paper has mentioned 3 functions for common contrast stretch algorithm. We can use any of the 3
    The algorithm wont affect much to the dehazing result

'''


def EstimateSaturation(h_saturation, p1):
    p2 = 2.0
    k1 = 0.5 * (1.0 - cv2.pow(1.0 - 2.0 * h_saturation, p1))
    k2 = 0.5 + 0.5 * cv2.pow((h_saturation - 0.5) / 0.5, p2)
    j_saturation = np.where(h_saturation <= 0.5, k1, k2)
    j_saturation = np.maximum(j_saturation, h_saturation)
    return j_saturation


def EstimateSaturation_Quadratic(h_saturation):
    return h_saturation * (2.0 - h_saturation)


def EstimateSaturation_Gamma(h_saturation, g):
    j_saturation = (np.power(h_saturation, 1.0 / g) + 1.0 - np.power(1.0 - h_saturation, 1.0 / g)) / 2.0
    j_saturation = np.maximum(j_saturation, h_saturation)
    return j_saturation


'''
    Estimate Transmission Map
'''


def EstimateTransimission(h_intensity, h_saturation, j_saturation):
    Td = h_intensity * (j_saturation - h_saturation)
    Tmn = j_saturation
    Tmap = 1.0 - (Td / Tmn)
    me = np.finfo(np.float32).eps  # adding some noise/haze to avoid division by zero in next step(recovery)
    Tmap = np.clip(Tmap, me, 1.0)
    return Tmap


'''
    Recover dehazed image
'''


def Recover(im, tmap, A):
    res = np.empty(im.shape, im.dtype)
    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / tmap + A[0, ind]  # Equation 22 in DCP Paper
        res[:, :, ind] = np.clip(res[:, :, ind], 0.0, 1.0)
    return res


'''
    Adjust image range
'''


def Adjust(im, perh, perl):
    aim = np.empty(im.shape, im.dtype)
    im_h = np.percentile(im, perh)
    im_l = np.percentile(im, perl)

    for ind in range(0, 3):
        aim[:, :, ind] = (im[:, :, ind] - im_l) / (im_h - im_l)
        aim[:, :, ind] = np.clip(aim[:, :, ind], 0.0, 1.0)

    return aim


'''
    Normalize image 0 between 1
'''


def Normalize(im):
    aim = np.empty(im.shape, im.dtype)

    for ind in range(0, 3):
        im_h = np.max(im[:, :, ind])
        im_l = np.min(im[:, :, ind])
        aim[:, :, ind] = (im[:, :, ind] - im_l) / (im_h - im_l)
        aim[:, :, ind] = np.clip(aim[:, :, ind], 0.0, 1.0)

    return aim


'''
   White balance using grayworld assumption
'''


def gray_world(im):
    aim = np.empty(im.shape, im.dtype)
    mu_r = np.average(im[:, :, 2])
    mu_g = np.average(im[:, :, 1])
    mu_b = np.average(im[:, :, 0])
    aim[:, :, 0] = np.minimum(im[:, :, 0] * (mu_g / mu_b), 1.0)  # formula 22 in paper
    aim[:, :, 2] = np.minimum(im[:, :, 2] * (mu_g / mu_r), 1.0)
    aim[:, :, 1] = im[:, :, 1]

    return aim


'''
  CLAHE
'''


def Clahe(im, clip):
    HSV = cv2.cvtColor(f2i(im), cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    HSV[:, :, 2] = clahe.apply(HSV[:, :, 2])
    result_im = i2f(cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR))

    return result_im


'''
 Main
'''
import os

all_images = os.listdir('D:\\AGV Images\\agv_hazy')  # lists all files in location
scores = []

for img in all_images:
    if img.endswith('.jpg'):
        gt_img = img.replace('hazy', 'GT')
        file_hazy = 'D:/Images/agv_hazy/' + img
        file_GT = 'D:/Images/agv_truth/' + gt_img

        hazy_image = i2f(cv2.imread(file_hazy, cv2.IMREAD_COLOR))  # downscaled image intensity
        gro_truth = cv2.imread(file_GT, cv2.IMREAD_COLOR)

        hazy_gray = cv2.cvtColor(hazy_image, cv2.COLOR_BGR2GRAY)
        truth_gray = cv2.cvtColor(gro_truth, cv2.COLOR_BGR2GRAY)

        start_time = time.time()

        A = Compute_A_Tang(hazy_image)  # computed atmospheric light value
        # print("A = ", A)
        S_A = np.max(A) - np.min(A)

        '''
          Compute white balanced A
        '''

        hazy_imageWB = gray_world(hazy_image)
        A_WB = Compute_A_Tang(hazy_imageWB)  # atmospheric light value for gray image
        S_AWB = np.max(A_WB) - np.min(A_WB)

        '''
          Parameter set
              parameters for Adjust: perh = 99.9, perl = 0.5 
              parameter for selecting two branch: epsilon = 0.00 - 0.1
              parameter for CLAHE: clip size  = 1 (0-2)
        '''
        perh = 99.9
        perl = 0.5
        epsilon = 0.02
        cl = 1

        '''
        if else condition to determine if we have to remove color veil or not
        '''

        if S_A < S_AWB + epsilon:
            print('Method I - Normal (no removing color veil)')
            hazy_imagen = np.empty(hazy_image.shape, hazy_image.dtype)

            for ind in range(0, 3):
                hazy_imagen[:, :, ind] = hazy_image[:, :, ind] / A[0, ind]

            hazy_imagen = Normalize(hazy_imagen)
            hazy_I = GetIntensity(hazy_imagen)
            hazy_S = GetSaturation(hazy_imagen, hazy_I)

            '''
           Stretch function I(gamma): 0.2 (heavy haze) - 0.4 (low haze)
           Stretch function II(quadratic): no parameter
           Stretch function III: 4.0 (heavy haze) - 2.0 (low haze) 
           '''

            est_S = EstimateSaturation_Gamma(hazy_S, 0.2)
            # est_S = EstimateSaturation_Quadratic(hazy_S)
            # est_S = EstimateSaturation(hazy_S, 2.5)
            Transmap = EstimateTransimission(hazy_I, hazy_S, est_S)
            r_image = Recover(hazy_image, Transmap, A)
            r_image = Adjust(r_image, perh, perl)

        else:
            print('Method II- White Balance (removes color veil)')
            hazy_imagen = np.empty(hazy_image.shape, hazy_image.dtype)

            for ind in range(0, 3):
                hazy_imagen[:, :, ind] = hazy_image[:, :, ind] / A_WB[0, ind]

            hazy_imagen = Normalize(hazy_imagen)
            hazy_I = GetIntensity(hazy_imagen)
            hazy_S = GetSaturation(hazy_imagen, hazy_I)

            est_S = EstimateSaturation_Gamma(hazy_S, 0.2)
            # est_S = EstimateSaturation_Quadratic(hazy_S)
            # est_S = EstimateSaturation(hazy_S, 2.5)

            Transmap = EstimateTransimission(hazy_I, hazy_S, est_S)
            r_image = Recover(hazy_image, Transmap, A_WB)
            r_image = Adjust(r_image, perh, perl)
            r_image = gray_world(r_image)

        result_ce = Clahe(r_image, cl)

        end_time = time.time()

        print("--- %s seconds ---" % (end_time - start_time))

        '''
        We resize the image in the below code with max size as 600. 
        Resizing for viewing purposes only. 
        All the dehazing operation has been performed on the original image size
        '''

        (h, w) = hazy_image.shape[:2]
        max_size = 600
        if h >= w:
            if h > max_size:
                ns = h / max_size
                nh = int(h / ns)
                nw = int(w / ns)
            else:
                nh = h
                nw = w

        else:
            if w > max_size:
                ns = w / max_size
                nh = int(h / ns)
                nw = int(w / ns)
            else:
                nh = h
                nw = w

        hazy_r = cv2.resize(hazy_image, (nw, nh))
        # trans_r = cv2.resize(Transmap, (nw, nh))
        r_r = cv2.resize(r_image, (nw, nh))
        ce_r = cv2.resize(result_ce, (nw, nh))

        out1 = f2i(r_r)  # original dehazed image
        out2 = f2i(ce_r)  # clahe image

        '''
        Need to convert dehazed images to grayscale since SSIM accepts only grayscale images
        Also converted all images to same data type (one of the errors I got)
        '''

        dehaze1_gray = cv2.cvtColor(out1, cv2.COLOR_BGR2GRAY)
        dehaze_enhance_gray = cv2.cvtColor(out2, cv2.COLOR_BGR2GRAY)

        input_img_gray = cv2.resize(hazy_gray, (nw, nh))  # resizing to show output
        GT_resize = cv2.resize(gro_truth, (nw, nh))  # resize to show output
        GT_resize_gray = cv2.resize(truth_gray, (nw, nh))

        GT_resize_gray = GT_resize_gray.astype(np.float32)
        dehaze1_gray = dehaze1_gray.astype(np.float32)
        dehaze_enhance_gray = dehaze_enhance_gray.astype(np.float32)

        '''
        Calculated PSNR and SSIM values for: 
        1) Input vs GT
        2) GT vs Dehazed original
        3) GT vs Clahe Output
        
        Note: The final output taken into consideration is of Clahe
        '''

        psnr_valueor = psnr(gro_truth, f2i(hazy_image))
        psnr_value1 = psnr(GT_resize, out1)
        psnr_value2 = psnr(GT_resize, out2)

        ssim_val_in = ssim(input_img_gray, GT_resize_gray, channel_axis=None)
        ssim_val_out1 = ssim(dehaze1_gray, GT_resize_gray, channel_axis=None)
        ssim_val_out2 = ssim(dehaze_enhance_gray, GT_resize_gray, channel_axis=None)

        print("SSIM input = ", ssim_val_in, "  PSNR input = ", psnr_valueor)
        print("SSIM dehaze1 = ", ssim_val_out1, "  PSNR dehaze1 = ", psnr_value1)
        print("SSIM clahe = ", ssim_val_out2, "  PSNR clahe = ", psnr_value2)

        scores.append([img[:2], psnr_value2, ssim_val_out2])

        # cv2.imshow('Input hazy image', f2i(hazy_r))
        # cv2.imshow('Transmission Map ', f2i(trans_r))
        # cv2.imshow('OG dehazed image', out1)
        # cv2.imshow('Clahe dehazed image', out2)
        # cv2.imshow("Ground Truth", GT_resize)

        signal = cv2.imwrite('D:\\Task 4 - Dehazing\\Dehazed Images\\' + img[:2] + '_outdoor_dh.png', out2)
        if signal:
            print('Image-' + img[:2] + ' Saved Successfully!')

        # cv2.waitKey()
        # cv2.destroyAllWindows()

scores_df = pd.DataFrame(scores, columns=['k', 'PSNR', 'SSIM'])
scores_df.to_csv('D:\Task 4 - Dehazing\scores.csv', index=False, header=True)
