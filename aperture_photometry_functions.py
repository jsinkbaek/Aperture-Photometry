from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from detect_peaks import detect_peaks
import scipy.ndimage as ndimage
import easygui as eg
from matplotlib import gridspec
import time as tm


def get_max(img, sigma, alpha=20, size=10, limit=50):
    i_out = []
    j_out = []
    image_temp = np.copy(img)
    while True:
        k = np.argmax(image_temp)
        j, i = np.unravel_index(k, image_temp.shape)
        if len(i_out) > limit:
            break
        elif image_temp[j, i] >= alpha*sigma:
            i_out.append(i)
            j_out.append(j)
            x = np.arange(i-size, i+size)
            y = np.arange(j-size, j+size)
            xv, yv = np.meshgrid(x, y)
            image_temp[yv.clip(0, image_temp.shape[0]-1), xv.clip(0, image_temp.shape[1]-1)] = 0

            # print(xv)
        else:
            break
    return i_out, j_out


def get_fitsimage(image_file, show_info=False):

    # Open fits file and show info
    hdu_list = fits.open(image_file)
    hdr = hdu_list[0].header
    if show_info is True:
        hdu_list.info()
        print(repr(hdr))

    # Get image data from file
    image_data = hdu_list[0].data
    image_rgain = hdr['GAIN']
    image_rnoise = hdr['RDNOISE']
    image_hjd = hdr['HELJD']

    # Print image dimensions
    if show_info is True:
        print(type(image_data))
        print(image_data.shape)

    # Close fits file
    hdu_list.close()

    return image_data, image_rgain, image_rnoise, image_hjd


def get_oroimage(fits_file, show_info=True):

    # Open fits file and show info
    hdu_list = fits.open(fits_file)
    hdr = hdu_list[0].header
    if show_info is True:
        hdu_list.info()
        print(repr(hdr))

    result = {}

    # Get image data from file
    result['img_data'] = hdu_list[0].data
    if 'OBJECT' in hdr.keys():
        result['img_object'] = hdr['OBJECT']
        result['img_filter'] = hdr['FILTER']
        result['ra'] = hdr['RA']
        result['dec'] = hdr['DEC']
    elif hdr['IMAGETYP'] == 'Flat Field':
        result['img_object'] = None
        result['img_filter'] = hdr['FILTER']
        result['ra'] = None
        result['dec'] = None
    else:
        result['img_object'] = None
        result['img_filter'] = None
        result['ra'] = None
        result['dec'] = None
    result['time_utc'] = hdr['TIME']
    result['exp_time'] = hdr['EXPTIME']
    result['img_type'] = hdr['IMAGETYP']

    # Close fits file
    hdu_list.close()

    return result


def oro_fileloader():
    fileselect = eg.fileopenbox(title='Select images, darks and flats', multiple=True)
    imgs = {'img_filter': [], 'time_utc': [], 'ra': [], 'dec': [], 'exp_time': []}
    darks = {'time_utc': [], 'exp_time': []}
    flats = {'img_filter': [], 'time_utc': [], 'exp_time': []}
    bias = None
    for file in fileselect:
        file_data = get_oroimage(file, show_info=False)
        if file_data['img_type'] == 'Light Frame':
            if 'img_data' in imgs.keys():
                imgs['img_data'] = np.append(imgs['img_data'], file_data['img_data'][:, :, np.newaxis], axis=2)
            else:
                imgs['img_data'] = file_data['img_data']
                imgs['img_data'] = imgs['img_data'].reshape((len(imgs['img_data'][:, 0]),
                                                             len(imgs['img_data'][0, :]), 1))
            imgs['img_filter'].append(file_data['img_filter'])
            imgs['time_utc'].append(file_data['time_utc'])
            imgs['ra'].append(file_data['ra'])
            imgs['dec'].append(file_data['dec'])
            imgs['exp_time'].append(file_data['exp_time'])
        elif file_data['img_type'] == 'Dark Frame':
            if 'img_data' in darks.keys():
                darks['img_data'] = np.append(darks['img_data'], file_data['img_data'][:, :, np.newaxis], axis=2)
            else:
                darks['img_data'] = file_data['img_data']
                darks['img_data'] = darks['img_data'].reshape((len(darks['img_data'][:, 0]),
                                                               len(darks['img_data'][0, :]), 1))
            darks['time_utc'].append(file_data['time_utc'])
            darks['exp_time'].append(file_data['exp_time'])
        elif file_data['img_type'] == 'Flat Field':
            if 'img_data' in flats.keys():
                flats['img_data'] = np.append(flats['img_data'], file_data['img_data'][:, :, np.newaxis], axis=2)
            else:
                flats['img_data'] = file_data['img_data']
                flats['img_data'] = flats['img_data'].reshape((len(flats['img_data'][:, 0]),
                                                               len(flats['img_data'][0, :]), 1))
            flats['img_filter'].append(file_data['img_filter'])
            flats['time_utc'].append(file_data['time_utc'])
            flats['exp_time'].append(file_data['exp_time'])
        else:
            print(file_data['img_type'])
            print('Unknown filetype')
    return imgs, darks, flats, bias


def bias_correction(img_frames, biasframes):
    # Create master bias (assume biasframes is 3dim array with each image in the first 2 dimensions)
    master_bias = np.sum(biasframes, axis=2)/len(biasframes[0, 0, :])

    # Retract master bias from images
    for i in range(0, len(img_frames[0, 0, :])):
        img_frames[:, :, i] = img_frames[:, :, i] - master_bias

    return img_frames


def dark_correction(img_frames, darkframes, biasframes=None):
    # Remove bias from darks, if applicable
    if biasframes is not None:
        darkframes = bias_correction(darkframes, biasframes)

    # Create master dark from darkframes (assume darkframes is 3dim array with each image in the first 2 dimensions)
    master_dark = np.sum(darkframes, axis=2)/len(darkframes[0, 0, :])

    # Retract master dark from images
    for i in range(0, len(img_frames[0, 0, :])):
        img_frames[:, :, i] = img_frames[:, :, i] - master_dark

    return img_frames


def flat_correction(img_frames, flatframes, darkframes, flat_expt, dark_expt, biasframes=None):
    # Remove bias from flats, if applicable
    if biasframes is not None:
        flatframes = bias_correction(flatframes, biasframes)

    # Remove darks from flats (while correcting for different exposure times)
    for k in range(0, len(flatframes[0, 0, :])):
        current_dark = np.copy(darkframes)
        for i in range(0, len(darkframes[0, 0, :])):
            current_dark[:, :, i] = current_dark[:, :, i] * flat_expt[k] / dark_expt[i]
        flatframes[:, :, k] = dark_correction(flatframes[:, :, k][:, :, np.newaxis], current_dark,
                                              biasframes=biasframes)[:, :, 0]

    # Create master flat from flatframes (assume flatframes is 3dim array with each image in the first 2 dimensions)
    master_flat = np.sum(flatframes, axis=2)/len(flatframes[0, 0, :])
    master_flat = master_flat/np.mean(master_flat)

    # Divide imgs with master_flat
    for i in range(0, len(img_frames[0, 0, :])):
        img_frames[:, :, i] = img_frames[:, :, i] / master_flat

    return img_frames


def filter_pipe(imgs, darks, flats, bias=None, filters=('Bessell V', 'Bessell B', 'Bessell R')):
    # Expects all the image objects to be dictionaries with at least 'img_data' and 'img_filter' keys
    img_filtered = {}
    if bias is not None:
        bias_imgdat = bias['img_data']
    else:
        bias_imgdat = None

    dark_expt = darks['exp_time']
    for f in filters:
        # Find indices with current filter
        imgs_indx = [i for i in range(len(imgs['img_filter'])) if imgs['img_filter'][i] == f]
        flats_indx = [i for i in range(len(flats['img_filter'])) if flats['img_filter'][i] == f]
        if bias is not None:
            bias_indx = [i for i in range(len(bias['img_filter'])) if bias['img_filter'][i] == f]
        else:
            bias_indx = None

        # Select frames
        current_imgs = imgs['img_data'][:, :, imgs_indx]
        current_flats = flats['img_data'][:, :, flats_indx]
        flats['exp_time'] = np.asarray(flats['exp_time'])
        flat_expt = flats['exp_time'][flats_indx]

        # Perform corrections using current darks, flats and bias
        if bias is not None:
            current_imgs = bias_correction(current_imgs, biasframes=bias_imgdat)
        current_imgs = dark_correction(current_imgs, darks['img_data'], biasframes=bias_imgdat)
        current_imgs = flat_correction(current_imgs, current_flats, darks['img_data'], flat_expt, dark_expt,
                                       biasframes=bias_imgdat)

        img_filtered[f] = current_imgs
    return img_filtered


def matchmaker(img_frames, catalogue_frame, stddev=1):
    from astropy.convolution import Gaussian2DKernel
    from astropy.convolution import convolve
    # Number of image frames
    n_imgs = len(img_frames[0, 0, :])

    # Create kernel for smoothing
    kernel = Gaussian2DKernel(x_stddev=stddev)

    # Initialize arrays for smoothed images and local maxima coordinates (j, i) = (y, x)
    img_smooth = np.empty(img_frames.shape)
    imax_smooth = [np.empty(1) for k in range(0, n_imgs)]
    jmax_smooth = [np.empty(1) for k in range(0, n_imgs)]

    # Smooth catalogue/reference image and find local maxima
    cat_smooth = convolve(catalogue_frame[:, :, 0], kernel)
    imax_cat, jmax_cat = get_max(cat_smooth, np.std(cat_smooth), alpha=100)
    imax_cat = np.asarray(imax_cat)
    jmax_cat = np.asarray(jmax_cat)

    # Smooth images and find local maxima coordinates for each
    for k in range(0, n_imgs):
        img_smooth[:, :, k] = convolve(img_frames[:, :, k], kernel)
        i_temp, j_temp = get_max(img_smooth[:, :, k], np.std(img_smooth[:, :, k]), alpha=40)
        i_temp = np.asarray(i_temp)
        j_temp = np.asarray(j_temp)

        imax_smooth[k] = i_temp
        jmax_smooth[k] = j_temp

    # Show and select 4 stars in catalogue/reference image to compare images with
    refs_plot = image_plot(cat_smooth, plot_add=[imax_cat, jmax_cat], ginput=True)
    number_of_refs = len(refs_plot)
    i_ref = np.empty((number_of_refs, ))
    j_ref = np.empty((number_of_refs, ))
    for l in range(0, number_of_refs):
        current_ref = refs_plot[l]
        i_ref[l] = imax_cat[int(np.argmin(np.abs(imax_cat-current_ref[0])))]
        j_ref[l] = jmax_cat[int(np.argmin(np.abs(jmax_cat-current_ref[1])))]

    # Create geometric shape from reference points
    from itertools import combinations
    number_of_corners = 4
    number_of_diagonals = 2
    corners = list(range(0, number_of_corners))                     # quadrangle
    shape_combs = [x for x in combinations(corners, 2)]             # possible corner combinations for iteration
    number_of_sides = len(shape_combs)
    shape_ref = np.empty((number_of_sides, ))                       # preload array for storing side length
    for (m, n), k in zip(shape_combs, range(0, number_of_sides)):   # (m, n) elements in combs_ref, k is loop iteration
        curr_i_len = np.abs(i_ref[m] - i_ref[n])
        curr_j_len = np.abs(j_ref[m] - j_ref[n])
        shape_ref[k] = np.sqrt(curr_i_len**2 + curr_j_len**2)       # trigonometric length
    shape_ref_nondiag = np.copy(shape_ref)
    for k in range(0, number_of_diagonals):
        shape_ref = np.delete(shape_ref_nondiag, np.argmax(shape_ref_nondiag))
    print('shape_ref_nondiag shape', shape_ref_nondiag.shape)
    # shape_ref = shape_ref / np.max(shape_ref)                       # make sidelengths proportional to max sidelength
    shape_ref_argsort = np.argsort(shape_ref)
    shape_ref_sort = shape_ref[shape_ref_argsort]
    shape_ref_nd_as = np.argsort(shape_ref_nondiag)
    shape_ref_nd_s = shape_ref_nondiag[shape_ref_nd_as]

    # # Create stellar geometric shapes for each image frame and possible combination
    bestmatch_i = []
    bestmatch_j = []
    print('start loop')
    for k in range(0, n_imgs):
        t1 = tm.time()
        # Create stellar combinations
        stellar_indx = list(range(0, len(imax_smooth[k])))               # A list of indices for every star
        stellar_combs = [x for x in combinations(stellar_indx, number_of_corners)]  # Possible combinations of 4 stars
        stellar_combs_0 = np.copy(stellar_combs)
        print('scombs0 shape', stellar_combs_0.shape)
        n_scombs = len(stellar_combs)
        for l in range(0, n_scombs):
            current_comb = np.array(stellar_combs[l])                    # change to array for multi indexing
            current_comb = [current_comb[list(x)] for x in shape_combs]  # reform to combs of 2 (6 total)
            stellar_combs[l] = np.asarray(current_comb)                  # (0,1,2,3) to [[0,1],[0,2],[0,3],[1,2],..]
        stellar_combs = np.asarray(stellar_combs)

        # Calculate geometry of each combination
        imax_temp = imax_smooth[k]                                  # load current image found stars
        jmax_temp = jmax_smooth[k]

        shape_img = np.empty((n_scombs, number_of_sides))
        i_bmsearch = np.empty(stellar_combs_0.shape)                  # arrays to recover pixel pos of found best match
        j_bmsearch = np.empty(stellar_combs_0.shape)
        for l in range(0, n_scombs):
            current_comb = stellar_combs[l, :, :]
            current_comb_0 = stellar_combs_0[l, :]
            curr_i = imax_temp[current_comb]                        # create arrs with pos same shape as comb arr
            curr_j = jmax_temp[current_comb]
            i_bmsearch[l, :] = imax_temp[current_comb_0]
            j_bmsearch[l, :] = jmax_temp[current_comb_0]
            curr_i_len = np.abs(curr_i[:, 0]-curr_i[:, 1])          # calculate pixel length between stars
            curr_j_len = np.abs(curr_j[:, 0]-curr_j[:, 0])
            shape_img[l, :] = np.sqrt(curr_i_len**2+curr_j_len**2)  # trigonometric length):
            # shape_img[l, :] = shape_img[l, :]/np.max(shape_img[l, :])  # ratio with max length

        # Compare stellar geometric shapes with ref for every combination
        shape_img_nondiag = np.empty((n_scombs, number_of_sides-number_of_diagonals))
        for l in range(0, n_scombs):
            shape_temp = np.copy(shape_img[l, :])
            for g in range(0, number_of_diagonals):
                shape_temp = np.delete(shape_temp, np.argmax(shape_temp))
            shape_img_nondiag[l, :] = shape_temp

        shape_img_argsort = np.argsort(shape_img, axis=1)           # Sort each combination from low to high in sidelen
        shape_img_sort = np.take_along_axis(shape_img, shape_img_argsort, axis=1)

        shape_diff = np.abs(shape_img_sort - shape_ref_sort[np.newaxis, :])
        shape_diff_tot = np.sum(shape_diff, axis=1)                 # Total match difference. The lower the better
        best_comb_match = np.argmin(shape_diff_tot)                 # Minimum variation from reference

        shape_bm = shape_img_sort[best_comb_match, :]
        print('best shape match sorted', shape_bm)
        print('reference shape sorted', shape_ref_sort)
        i_bm = i_bmsearch[best_comb_match, :]
        j_bm = j_bmsearch[best_comb_match, :]
        bestmatch_i.append(i_bm)
        bestmatch_j.append(j_bm)
        bm_argsort = shape_img_argsort[best_comb_match, :]
        t2 = tm.time()
        print('loop time', t2-t1)

    print('frames shape', img_frames.shape)
    imgs_plot = np.append(img_smooth, cat_smooth[:, :, np.newaxis], axis=2)
    print('imgs plot shape', imgs_plot.shape)

    ncols = 4
    nrows = int(np.ceil(len(imgs_plot[0, 0, :])/ncols))
    ax = multiplot(imgs_plot, ncols=ncols, nrows=nrows)
    for k in range(0, n_imgs):
        (crow, ccol) = np.unravel_index(k, (nrows, ncols))
        ax[crow][ccol].plot(bestmatch_i[k][:], bestmatch_j[k][:], 'ro', markersize=5, alpha=0.25)
        del crow, ccol
    (ref_row, ref_col) = np.unravel_index(len(imgs_plot[0, 0, :])-1, (nrows, ncols))
    ax[ref_row][ref_col].plot(i_ref, j_ref, 'bo', markersize=5, alpha=0.25)
    plt.show(block=True)
    return [[bestmatch_i, bestmatch_j], [i_ref, j_ref]]


def oro_pipeline():
    imgs, darks, flats, bias = oro_fileloader()

    imgs_filtered = filter_pipe(imgs, darks, flats, bias, filters=('Bessell V', 'Bessell B', 'Bessell R'))

    # Get reference frame for coordinate matchmaking
    fileselect = eg.fileopenbox(title='Select catalogue/reference image')
    file_data = get_oroimage(fileselect, show_info=False)
    ref_img = {}
    ref_img['img_data'] = file_data['img_data'].reshape((len(file_data['img_data'][:, 0]),
                                                         len(file_data['img_data'][0, :]), 1))
    ref_img['img_filter'] = [file_data['img_filter']]
    ref_img['ra'] = file_data['ra']
    ref_img['dec'] = file_data['dec']
    ref_filtered = filter_pipe(ref_img, darks, flats, filters=(ref_img['img_filter']), bias=None)
    ref_filtered = ref_filtered[ref_img['img_filter'][0]]

    imgs_V = imgs_filtered['Bessell V']
    imgs_B = imgs_filtered['Bessell B']
    imgs_R = imgs_filtered['Bessell R']

    imgs_BVR = np.copy(imgs_B)
    imgs_BVR = np.append(imgs_BVR, imgs_V, axis=2)
    imgs_BVR = np.append(imgs_BVR, imgs_R, axis=2)
    [[best_match_x, best_match_y], [ref_x, ref_y]] = matchmaker(imgs_BVR, ref_filtered)
    return best_match_x, best_match_y, ref_x, ref_y


def image_plot(image_data, plot_add=None, ginput=False, show_info=False, block=True):
    if show_info is True:
        print('Min:', np.min(image_data))
        print('Max:', np.max(image_data))
        print('Mean:', np.mean(image_data))
        print('Stdev:', np.std(image_data))
    fig1, ax1 = plt.subplots(1, 1)

    ax1.imshow(image_data, cmap='gray_r', norm=LogNorm())
    if plot_add is not None:
        ax1.plot(plot_add[0], plot_add[1], 'ro', markersize=5, alpha=0.5)

    if ginput is True:
        st, r1, r2, bckg = plt.ginput(n=4, timeout=0, show_clicks=True)
        plt.close(fig1)
        del fig1
        return [st, r1, r2, bckg]
    else:
        plt.show(block=block)


def xyloader(image_data, **kwargs):
    """
    Short function to load a list of pre-set coordinates for the objects in question, or select with graphical input
    :param image_data: loaded data from a fits file (see get_fitsimage)
    :param kwargs: Optional input to state if a file with pre-set coordinates by name 'coordinates.dat' exists
    :return: objects (x,y coordinates for each object on the image)
    """
    inpt = kwargs.get('inpt', False)

    if inpt is False:
        inpt = input('Do you want to load previously saved data? (y/n) '
                     'If not, you will be prompted to select new coordinates')
    if inpt == 'y':
        # Preload list
        objects = []
        # Open data file in read mode
        f = open('coordinates.dat', 'r')
        #
        for line in f:
            columns = line.split()
            columns[0] = float(columns[0])
            columns[1] = float(columns[1])
            objects.append(columns)
    elif inpt == 'n':
        # Select object coordinates
        objects = image_plot(image_data, ginput=True, show_info=False)
        # Clear data file
        open('coordinates.dat', 'w').close()
        # Write to data file
        for k in range(0, len(objects)):
            content = str(objects[k][0]) + '   ' + str(objects[k][1]) + '\n'
            with open('coordinates.dat', 'a') as f:
                f.write(content)
    else:
        raise ValueError('Unknown keyboard input %s' % inpt)

    return objects


def create_circular_mask(h, w, center=None, radius=None):
    """
    Creates a circular mask to be applied on an image. Can receive a center indicating where the mask should be located,
    and a radius indicating its circular radius size in pixels.
    """
    if center is None:  # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0])**2 + (y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def mask_image(image_data, aperture_pxsize, pixel_coords, coord_offsets):
    """
    Function to create a masked image of the same size as the one given by image_data, but with zeroes everywhere except
    in a circle placed somewhere in the image (given by pixel_coords and coord_offsets) of pixel radius given by
    aperture_pxsize. Calls create_circular_mask in order to create a mask to be applied on the image.
    """
    # Find coordinates with given offset
    aperture_coords = pixel_coords + coord_offsets

    # # Create image aperture

    # Make circular mask
    h, w = image_data.shape[:2]
    mask = create_circular_mask(h, w, center=aperture_coords, radius=aperture_pxsize)

    # Apply circular mask to make aperture image
    masked_img = image_data.copy()
    masked_img[~mask] = 0

    return masked_img


def mask_counting(masked_img):
    # Sum together all pixel values within the masked image
    result = np.sum(masked_img)
    return result


def aperture_counter(coordinates, coord_offset, image_data, aperture_pxsize, show_plot=False, rot_offset=False):
    """
    A function that performs pixel counting with aperture masks in an image (for multiple objects), by calling the
    functions mask_image (to create a masked image) and mask_counting (to count pixels in masked image).
    """
    # coordinates must be a list of x,y values for each object
    # coord_offset must be the x,y offset to be applied to only this image instance

    # Find number of objects
    n = len(coordinates)

    # Create empty array to use for pixel counts
    aperture_counts = np.zeros(n)

    # Create empty list of images
    img = np.zeros(image_data.shape)

    # For loop to create aperture mask and sum pixel counts within
    for k in range(0, n):
        coord = np.asarray(coordinates[k])

        if rot_offset is True:
            offset = coord_offset[k]
        else:
            offset = coord_offset

        masked_image = mask_image(image_data, aperture_pxsize, coord, offset)
        aperture_counts[k] = mask_counting(masked_image)
        img = img + masked_image

    if show_plot is True:
        plt.figure()
        plt.imshow(img, cmap='gray_r', norm=LogNorm())
        plt.show(block=False)

    return aperture_counts


def flux_convergence_test(coordinates, image_data, aperture_steps, rgain, rnoise, coord_offset=np.asarray([0, 0]),
                          show_plot=True, show_aperture_plot=False):
    """
    A long function that calculates optimal aperture size for all selected objects in an image, based on signal/noise
    ratio, and plots it. Has functionality for multiple different uses. Can be used with coordinate measurements for
    a different image, as long as the coordinate offsets for this image is known in comparison to it.
    """
    # Update plot font size for larger text
    plt.rcParams.update({'font.size': 25})

    # Set background index location
    bg_placement = len(coordinates)-1

    # Preload flux array
    flux = np.empty((len(coordinates), len(aperture_steps)))

    # Fill flux array with aperture counts
    for k in range(0, len(aperture_steps)):
        flux[:, k] = aperture_counter(coordinates, coord_offset, image_data, aperture_steps[k], show_plot=False)

    # Calculate S/N ratio (assuming uniform background with low errors)
    sn_ratio = np.empty(flux.shape)
    for k in range(0, len(coordinates)):
        flux_corrected = flux[k, :] - flux[bg_placement, :]
        sn_ratio[k, :] = rgain * flux_corrected / (np.sqrt(rgain * flux_corrected + rgain * flux[bg_placement, :]
                                                           + aperture_steps**2 * np.pi * rnoise**2))

    peaks_indx = []
    peaks_px = []
    for k in range(0, len(coordinates) - 1):
        peaks_indx.append(np.argmax(sn_ratio[k, :]))
        peaks_px.append(aperture_steps[peaks_indx[k]])
    max_peak = max(peaks_indx)
    max_pxsize = int(np.ceil(max(peaks_px)))

    sn_max_pxsize = []
    for k in range(0, len(coordinates) - 1):
        sn_max_pxsize.append(sn_ratio[k, max_peak])

    if show_aperture_plot is True:
        aperture_counter(coordinates, coord_offset, image_data, max_pxsize, show_plot=show_aperture_plot)

    legendtable1 = []
    legendtable2 = []
    if show_plot is True:
        plt.figure()
        for k in range(0, len(coordinates) - 1):
            plt.plot(aperture_steps, 2.5 * np.log10(flux[k, :] - flux[bg_placement, :]), '.', markersize=10)
            legendtable1.append('Object ' + str(k) + ' - Background')
        plt.plot(aperture_steps, 2.5 * np.log10(flux[bg_placement, :]), '.', markersize=10)
        for k in range(0, len(coordinates) - 1):
            plt.plot(peaks_px[k], 2.5 * np.log10(flux[k, peaks_indx[k]] - flux[bg_placement, peaks_indx[k]]), '*',
                     markersize=15)
        legendtable1.append('Background')
        plt.xlabel('Aperture pixel size')
        plt.ylabel('Aperture count magnitude (log10(ADU))')
        plt.legend(legendtable1)
        plt.show(block=False)

        plt.figure()
        for k in range(0, len(coordinates)):
            plt.plot(aperture_steps, sn_ratio[k, :], '.', markersize=10)
            legendtable2.append('Object ' + str(k))
        for k in range(0, len(coordinates) - 1):
            plt.plot(peaks_px[k], sn_ratio[k, peaks_indx[k]], '*', markersize=15)
        plt.xlabel('Aperture pixel size (radius)')
        plt.ylabel('S/N ratio (ADU/e)')
        plt.legend(legendtable2, loc='lower right')
        plt.show(block=True)

    print(max_pxsize)
    return max_pxsize, sn_max_pxsize


def image_looper(coordinates, coord_offset, image_filenames, aperture_pxsize=None, show_plot=True):
    """
    Loops through multiple images, and outputs observed pixel count for each star.
    Assumes that coordinates for first image is known, and that the image offset (in comparison to the first image) is
    known for every single image.
    """
    # Check if length of coord_offset is equal to length of filenames
    if len(coord_offset) != len(image_filenames):
        raise TypeError('Length of lists coord_offset and image_filenames do not match, they are %s'
                        % [len(coord_offset), len(image_filenames)])

    number_of_images = len(image_filenames)
    number_of_objects = len(coordinates)

    # Check if coord_offset is a linear offset (all objects have same offset) or rot+lin offset (different offsets)
    if len(coord_offset[0]) == number_of_objects:
        rot_off = True
    elif isinstance(coord_offset[0], (np.ndarray, np.generic)):
        rot_off = False
    else:
        raise ValueError('Offset dimension or type does not seem to match. If rot+lin offset, type should be list,'
                         'and length should be the same as amount of objects. If linear offset, type should be a numpy'
                         'array, and length should be 2 (x,y).')

    # Preload array of observation values
    measured_counts = np.zeros((number_of_images, number_of_objects))
    hjd = np.empty(number_of_images)

    # Define aperture size steps to be evaluated in flux convergence test
    aperture_pxrange = [3, 12]
    aperture_steps = np.linspace(aperture_pxrange[0], aperture_pxrange[1], 50)

    # Loop over images to fill measured_counts
    for k in range(0, number_of_images):
        image_data, image_gain, image_noise, image_hjd = get_fitsimage(image_filenames[k], show_info=False)
        hjd[k] = image_hjd
        loop_check = False

        if aperture_pxsize is None:
            aperture_pxsize, sn_ratio = flux_convergence_test(coordinates, image_data, aperture_steps, image_gain,
                                                              image_noise, coord_offset=coord_offset[k],
                                                              show_plot=False)
            loop_check = True

        # Perfom aperture creation for each object and get aperture count
        apt_cnt = np.asarray(aperture_counter(coordinates, coord_offset[k], image_data,
                                              aperture_pxsize, show_plot=show_plot, rot_offset=rot_off))

        # Reshape to match a row vector shape
        apt_cnt = np.reshape(apt_cnt, (1, number_of_objects))

        if show_plot is True:
            plt.imshow(image_data, cmap='gray_r', norm=LogNorm())
            plt.show()

        # Input in measured_counts
        measured_counts[k, :] = apt_cnt[:]

        if loop_check is True:
            aperture_pxsize = None

    return measured_counts, hjd


def phaseplot(mag, time, pguess=1, repeat=True):
    """
    Creates a cepheid phase plot. Can be animated/interactive (meaning keyboard interaction changes it, or a simple plot
    that exits after a single loop.
    """
    plt.rcParams.update({'font.size': 30})
    loop = True
    plt.ion()
    period = pguess
    stepsize = 0.25
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot1, = ax.plot(np.mod(time, period)/period, mag, 'b*')
    print('Input a to decrease period, d to increase, q to quit loop, s to halve stepsize, w to double '
          'stepsize, and p to pass (repeat same)')
    while loop is True:
        print('period', period, ', stepsize', stepsize)
        plot1.set_xdata(np.mod(time, period)/period)
        fig.canvas.draw()
        fig.canvas.flush_events()
        fig.legend(['Period '+'%.3f'% period + ' days'])
        ax.set_xlabel('Phase: (time modulus period) / period')
        ax.set_ylabel('Cepheid relative luminosity (V1/S1)')
        inpt = input()
        if inpt == 'q':
            loop = False
        elif inpt == 'a':
            period -= stepsize
        elif inpt == 'd':
            period += stepsize
        elif inpt == 's':
            stepsize = stepsize/2
        elif inpt == 'w':
            stepsize = stepsize*2
        elif inpt == 'p':
            pass
        if repeat is False:
            loop = False


def periodlinreg(mag, calcheck, time):
    """
    Finds maxima and minima of luminosity curve, plots them (for check) and gets two linear regression estimates of
    the cepheid period (one for peaks and one for valleys). Returns a mean of these two values
    """
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                                   AutoMinorLocator)

    # Update plot font size for larger text
    plt.rcParams.update({'font.size': 30})

    # # Plot cepheid luminosity over time

    # Load figure and set axis ticks
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    ax.xaxis.set_minor_locator(MultipleLocator(5))

    # Plot data and set labels
    ax.plot(time, mag, 'k')
    ax.set_xlabel('Time in days')
    ax.set_ylabel('Cepheid magnitude')

    # Find peak and valley indices using outside function "detect_peaks": "Marcos Duarte, https://github.com/demotu/BMC"
    ind_peaks = detect_peaks(mag)
    ind_valleys = detect_peaks(-mag)

    x_peaks = np.asarray(mag[ind_peaks])
    x_valleys = np.asarray(mag[ind_valleys])

    t_peaks = np.asarray(time[ind_peaks])
    t_valleys = np.asarray(time[ind_valleys])

    # Plot peak and valley data
    ax.plot(t_peaks, x_peaks, 'bs', markersize=12)
    ax.plot(t_valleys, x_valleys, 'rp', markersize=12)

    # Plot calibration check
    ax.plot(time, calcheck, '--')

    # Legend
    ax.legend(['V1/S1', 'Peak data', 'Valley data', 'S1/S2'], loc='upper right')

    plt.show(block=False)

    # # Plot linear fit to period from peaks and valleys

    # Create peak and valley numbers
    num_peaks = np.asarray([float(x+1) for x in range(0, len(x_peaks))])
    num_valleys = np.asarray([float(x+1) for x in range(0, len(x_valleys))])

    # Make polynomial 1d fit
    fit_peaks = np.polyfit(num_peaks, t_peaks, 1)
    fit_peaks_fn = np.poly1d(fit_peaks)
    fit_valleys = np.polyfit(num_valleys, t_valleys, 1)
    fit_valleys_fn = np.poly1d(fit_valleys)

    # Load figure and set axis ticks
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    ax.xaxis.set_minor_locator(MultipleLocator(1))

    # Plot peak and valley data, as well as corresponding fits
    ax.plot(num_peaks, t_peaks, 'bs', markersize=12)
    ax.plot(num_peaks, fit_peaks_fn(num_peaks), '--k')
    ax.plot(num_valleys, t_valleys, 'rp', markersize=12)
    ax.plot(num_valleys, fit_valleys_fn(num_valleys), '-.k')
    ax.set_xlabel('Extremum #')
    ax.set_ylabel('Time in days')
    ax.legend(['Peak data', 'Peak fit', 'Valley data', 'Valley fit'])
    plt.show()

    # Print fit slopes
    print('Slopes for each fit   ', fit_peaks[0], '   ', fit_valleys[0])

    # Return mean of both fit slopes
    return np.mean([fit_peaks[0], fit_valleys[0]])


def multiplot(imgs, nrows=5, ncols=5):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                           gridspec_kw={'wspace': 0.0, 'hspace': 0.0, 'width_ratios': [1]*ncols,
                                        'height_ratios': [1]*nrows, 'top': 1-0.5/(nrows+1), 'bottom': 0.5/(nrows+1),
                                        'left': 0.5/(ncols+1), 'right': 1-0.5/(ncols+1)})
    # gs = gridspec.GridSpec(nrows, ncols, width_ratios=[1, 1, 1], wspace=0.05, hspace=0.05)
    for i, axi in enumerate(ax.flat):
        if i < len(imgs[0, 0, :]):
            img = imgs[:, :, i]
            axi.imshow(img, cmap='gray_r', norm=LogNorm())
            axi.set_xticks([])
            axi.set_yticks([])
        else:
            print('out of images')
            delrow, delcol = np.unravel_index(i, (nrows, ncols))
            fig.delaxes(ax[delrow][delcol])
    return ax

