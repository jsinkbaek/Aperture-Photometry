from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from detect_peaks import detect_peaks


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


def image_plot(image_data, ginput=False, show_info=False, block=True):
    if show_info is True:
        print('Min:', np.min(image_data))
        print('Max:', np.max(image_data))
        print('Mean:', np.mean(image_data))
        print('Stdev:', np.std(image_data))
    plt.figure()
    plt.imshow(image_data, cmap='gray_r', norm=LogNorm())

    if ginput is True:
        st, r1, r2, bckg = plt.ginput(n=4, timeout=0, show_clicks=True)
        plt.close()
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
            plt.plot(aperture_steps, 2.5 * np.log10(flux[k, :] - flux[bg_placement, :]), '.')
            legendtable1.append('Object ' + str(k) + ' - Background')
        plt.plot(aperture_steps, 2.5 * np.log10(flux[bg_placement, :]), '.')
        for k in range(0, len(coordinates) - 1):
            plt.plot(peaks_px[k], 2.5 * np.log10(flux[k, peaks_indx[k]] - flux[bg_placement, peaks_indx[k]]), '*')
        legendtable1.append('Background')
        plt.xlabel('Aperture pixel size')
        plt.ylabel('Aperture count magnitude (log10(ADU))')
        plt.legend(legendtable1)
        plt.show(block=False)

        plt.figure()
        for k in range(0, len(coordinates)):
            plt.plot(aperture_steps, sn_ratio[k, :], '.')
            legendtable2.append('Object ' + str(k))
        for k in range(0, len(coordinates) - 1):
            plt.plot(peaks_px[k], sn_ratio[k, peaks_indx[k]], '*')
        plt.xlabel('Aperture pixel size (radius)')
        plt.ylabel('S/N ratio (ADU/e)')
        plt.legend(legendtable2)
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
        ax.set_ylabel('Cepheid Magnitude')
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


def periodlinreg(mag, time):
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
    ax.plot(t_peaks, x_peaks, 'b*')
    ax.plot(t_valleys, x_valleys, 'r*')
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
    ax.plot(num_peaks, t_peaks, 'b*')
    ax.plot(num_peaks, fit_peaks_fn(num_peaks), '--k')
    ax.plot(num_valleys, t_valleys, 'r*')
    ax.plot(num_valleys, fit_valleys_fn(num_valleys), '-.k')
    ax.set_xlabel('Extremum #')
    ax.set_ylabel('Time in days')
    ax.legend(['Peak data', 'Peak fit', 'Valley data', 'Valley fit'])
    plt.show()

    # Print fit slopes
    print('Slopes for each fit   ', fit_peaks[0], '   ', fit_valleys[0])

    # Return mean of both fit slopes
    return np.mean([fit_peaks[0], fit_valleys[0]])
