import numpy as np
import cv2
from skimage import morphology
import time

'''
   pixel value: i2f: 0-255 to 0-1, f2i: 0-1 to 0-255
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


def Compute_A_Tang(im):  # search common contrast stretch algo
    # Parameters
    erosion_window = 7
    n_bins = 200

    R = im[:, :, 2]
    G = im[:, :, 1]
    B = im[:, :, 0]

    # compute the dark channel
    dark = morphology.erosion(np.min(im, 2), morphology.square(erosion_window))
    [h, edges] = np.histogram(dark, n_bins)
    numpixel = im.shape[0] * im.shape[1]
    thr_frac = numpixel * 0.99
    csum = np.cumsum(h)
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
    min_rgb = cv2.min(cv2.min(fi[:, :, 0], fi[:, :, 1]), fi[:, :, 2])
    me = np.finfo(np.float32).eps
    S = 1.0 - min_rgb / (intensity + me)
    return S


'''
    Estimate saturation of scene radiance: 3 methods
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
    me = np.finfo(np.float32).eps
    Tmap = np.clip(Tmap, me, 1.0)

    return Tmap


'''
    Recover dehazed image
'''


def Recover(im, tmap, A):
    res = np.empty(im.shape, im.dtype)

    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / tmap + A[0, ind]
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


def Normalize(im):
    """
    Normalize image 0 between 1
    """
    aim = np.empty(im.shape, im.dtype)

    for ind in range(0, 3):
        im_h = np.max(im[:, :, ind])
        im_l = np.min(im[:, :, ind])
        aim[:, :, ind] = (im[:, :, ind] - im_l) / (im_h - im_l)
        aim[:, :, ind] = np.clip(aim[:, :, ind], 0.0, 1.0)

    return aim


def gray_world(im):
    """
    Def: White balance using grayworld assumption
    Args: takes an RGB image (np.array) as an input
    Returns: aim
    """
    aim = np.empty(im.shape, im.dtype)

    mu_r = np.average(im[:, :, 2])
    mu_g = np.average(im[:, :, 1])
    mu_b = np.average(im[:, :, 0])
    aim[:, :, 0] = np.minimum(im[:, :, 0] * (mu_g / mu_b), 1.0)
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


if __name__ == "__main__":
    import os

    all_videos_dir = "D:\Images"
    output_videos_dir = "D:\Images"
    all_videos = os.listdir(all_videos_dir)
    for vid in all_videos:
        if vid.endswith('.mp4'):
            cap = cv2.VideoCapture(os.path.join(all_videos_dir, vid))
            # cap.set(cv2.CAP_PROP_FPS, 30)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            result = cv2.VideoWriter('dehazed.mp4', fourcc, 1000.0, (width, height))

            prev_time = 0
            new_fr_time = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Done!")
                    break

                windowSze = 5
                scaled_down_frame = i2f(frame)
                scaled_down_frame_gray = cv2.cvtColor(scaled_down_frame, cv2.COLOR_BGR2GRAY)
                # print(frame)
                A = Compute_A_Tang(scaled_down_frame)
                # print("A = ", A)
                S_A = np.max(A) - np.min(A)

                '''
                Compute white balanced A
                '''
                hazy_imageWB = gray_world(scaled_down_frame)
                # scaled_down_frame = i2f(hazy_imageWB)
                A_WB = Compute_A_Tang(hazy_imageWB)
                S_AWB = np.max(A_WB) - np.min(A_WB)

                '''
                Parameter set
                    parameters for Adjust: perh = 99.9, perl = 0.5
                    (for full-reference image: perh = 100, perl = 0
                    parameter for selecting two branch: epsilon = 0.00 - 0.1
                    parameter for CLAHE: clip size  = 1 (0-2)
                    
                '''
                perh = 99.9
                perl = 0.5
                epsilon = 0.02
                cl = 1

                if S_A < S_AWB + epsilon:
                    # print('Phase I - Normal')
                    hazy_imagen = np.empty(scaled_down_frame.shape, scaled_down_frame.dtype)

                    for ind in range(0, 3):
                        hazy_imagen[:, :, ind] = scaled_down_frame[:, :, ind] / A[0, ind]

                    hazy_imagen = Normalize(hazy_imagen)
                    hazy_I = GetIntensity(hazy_imagen)
                    hazy_S = GetSaturation(hazy_imagen, hazy_I)

                    '''
                    Stretch function I: 0.2 (heavy haze) - 0.4 (low haze)
                    Stretch function II: no parameter
                    Stretch function III: 4.0 (heavy haze) - 2.0 (low haze)
                    '''
                    est_S = EstimateSaturation_Gamma(hazy_S, 0.2)
                    # est_S = EstimateSaturation_Quadratic(hazy_S)
                    # est_S = EstimateSaturation(hazy_S, 2.0)
                    Transmap = EstimateTransimission(hazy_I, hazy_S, est_S)
                    r_image = Recover(scaled_down_frame, Transmap, A)
                    r_image = Adjust(r_image, perh, perl)

                else:
                    # print('Phase II -White Balance')
                    hazy_imagen = np.empty(scaled_down_frame.shape, scaled_down_frame.dtype)

                    for ind in range(0, 3):
                        hazy_imagen[:, :, ind] = scaled_down_frame[:, :, ind] / A_WB[0, ind]

                    hazy_imagen = Normalize(hazy_imagen)
                    hazy_I = GetIntensity(hazy_imagen)
                    hazy_S = GetSaturation(hazy_imagen, hazy_I)

                    est_S = EstimateSaturation_Gamma(hazy_S, 0.2)
                    # est_S = EstimateSaturation_Quadratic(hazy_S)
                    # est_S = EstimateSaturation(hazy_S, 2.0)

                    Transmap = EstimateTransimission(hazy_I, hazy_S, est_S)
                    r_image = Recover(scaled_down_frame, Transmap, A_WB)
                    r_image = Adjust(r_image, perh, perl)
                    r_image = gray_world(r_image)

                result_ce = Clahe(r_image, cl)

                new_fr_time = time.time()

                (h, w) = scaled_down_frame.shape[:2]

                ce_r = cv2.resize(result_ce, (800, 600))  # change to (w,h)
                fin_out_frame = f2i(ce_r)

                fps = 1 / (new_fr_time - prev_time)
                prev_time = new_fr_time

                # print('fps=', fps)
                # fps = int(fps)
                # print('int fps=', fps)
                # fps = str(fps)

                cv2.putText(frame, "HEY there", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 255, (0, 0, 255), 4)

                result.write(fin_out_frame)

                # print('dtype=', fin_out_frame.dtype)
                # print('shape=', fin_out_frame.shape)
                # print('\n')
                cv2.imshow('Enhanced frame', fin_out_frame)

                if cv2.waitKey(25) & 0xFF == ord('d'):
                    print("End")
                    break

            cap.release()
            result.release()
            cv2.destroyAllWindows()
            print('Video was saved!')
