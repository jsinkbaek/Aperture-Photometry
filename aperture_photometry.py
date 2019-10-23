import numpy as np
import matplotlib.pyplot as plt
import os
import aperture_photometry_functions as apf
"""
Author: Jeppe Sinkbaek Thomsen, https://github.com/jsinkbaek/Aperture-Photometry

This is a runnable script used for aperture photometry. It consists of two primary parts:
1. Loading and aperture measurement. This part is used to either load previously measured aperture photometry
   data or, if none are available, make aperture measurements on a set of images in order to make and save this data.
   This is the most time consuming, but also most extensive part of the script.
   
2. Cepheid data analysis. This short part is used for analysis of the previously mentioned aperture photometry data,
   specifically in the case of the data being from a Cepheid or similarly variable star. It includes a way of estimating
   the cepheid variability period from linear regression of peaks and valleys, as well as an interactive phase-plot
   (expected period can be varied to see how it changes the phase-plot).
   
Most of this script's functionality is found in the function repository "aperture_photometry_functions.py", which is 
referenced throughout the script as apf.

Dependencies:

    - offsets.txt (must be present in root folder and include an x,y offset for every image to be analyzed 
      (when comparing with a reference image)
      
    - o4201193.10.fts (must be present in root folder. Reference image. Can be replaced with another, if name is 
      changed in the script to account for this)
      
    - detect_peaks.py (file with a function used. 
      Author of this file is Marcos Duarte,https://github.com/demotu/BMC)
      
    - The folder cepheid_data including the following files: 
            - filenames.txt (a file with a list of filenames for all the image fits files to be analyzed) 
            - Image files to be analyzed

"""

# # Loading and aperture measurement
data_exists = True

# Check if datafile cepheid.dat exists (if so, load data), or if aperture selection needs to be performed
try:
    f = open(os.path.join('cepheid_data', 'cepheid.dat'), 'r')
    ceph_flux = []
    time = []
    for line in f:
        columns = line.split()
        ceph_flux.append(float(columns[0]))
        time.append(float(columns[1]))
    f.close()

    g = open(os.path.join('cepheid_data', 'calibrationstar.dat'), 'r')
    calibration_check = []
    for line in g:
        columns = line.split()
        calibration_check.append(float(columns[0]))
    g.close()

    ceph_flux = np.asarray(ceph_flux)
    time = np.asarray(time)
    calibration_check = np.asarray(calibration_check)

except FileNotFoundError:
    data_exists = False


# Perform aperture measurement if necessary
if data_exists is False:
    img_file_name = 'o4201193.10.fts'
    img_data, img_rgain, img_rnoise, img_hjd = apf.get_fitsimage(img_file_name, show_info=False)

    object_coordinates = apf.xyloader(img_data)

    # Define aperture size steps to be evaluated in flux convergence test
    aperture_pxrange = [3, 20]
    aperture_steps = np.linspace(aperture_pxrange[0], aperture_pxrange[1], 50)
    apf.flux_convergence_test(object_coordinates, img_data, aperture_steps, img_rgain, img_rnoise)

    filenames = []
    f = open(os.path.join('cepheid_data', 'filenames2.txt'), 'r')
    for line in f:
        filenames.append(os.path.join('cepheid_data', line.split()[0]))
    f.close()

    offsets = []
    f = open('offsets.txt', 'r')
    for line in f:
        columns = line.split()
        columns[0] = float(columns[0])
        columns[1] = float(columns[1])
        offsets.append(np.asarray([columns[1], columns[0]]))
    f.close()

    measured_counts, helJD = apf.image_looper(object_coordinates, offsets, filenames, show_plot=False)

    # Time in days from first observation
    time = np.asarray([(x - helJD[0]) for x in helJD])

    # Subtract background
    measured_counts[:, 0] = measured_counts[:, 0] - measured_counts[:, 3]
    measured_counts[:, 1] = measured_counts[:, 1] - measured_counts[:, 3]
    measured_counts[:, 2] = measured_counts[:, 2] - measured_counts[:, 3]

    # Calibrated flux ratio
    ceph_flux = measured_counts[:, 0] / measured_counts[:, 1]
    calibration_check = measured_counts[:, 1] / measured_counts[:, 2]

    print(ceph_flux)
    print(calibration_check)
    plt.plot(time, ceph_flux)
    plt.plot(time, calibration_check)
    plt.ylabel('ADU ratio for objects')
    plt.xlabel('Time in days after '+str(helJD[0])+'HJD')
    plt.legend(['V1/S1', 'S1/S2'])
    plt.show()

    # Examine and delete images with potentially bad quality
    inpt = input('Do you want to examine images and potentially delete any data points? Input Y to do so')
    if inpt == 'Y':
        loop = True
    else:
        loop = False

    while loop is True:
        # Plot and select
        plt.plot(time, ceph_flux)
        plt.plot(time, calibration_check)
        plt.title('Please select a data point to examine')
        inptime = plt.ginput(n=1, timeout=0, show_clicks=True)[0][0]
        print(inptime)
        plt.close()
        indx = np.argmin(np.abs(time - inptime))

        # Plot and show selected
        plt.plot(time, ceph_flux)
        plt.plot(time, calibration_check)
        plt.plot(time[indx], ceph_flux[indx], 'r*')
        plt.show()

        # Select filename and get fits file and object coordinates
        img_file_name = filenames[indx]
        img_data, img_rgain, img_rnoise, img_hjd = apf.get_fitsimage(img_file_name, show_info=True)
        object_coordinates = apf.xyloader(img_data, inpt='y')

        # Create aperture steps for flux convergence test
        aperture_pxrange = [3, 12]
        aperture_steps = np.linspace(aperture_pxrange[0], aperture_pxrange[1], 50)

        # Plot full image
        apf.image_plot(img_data, show_info=True, block=False)

        # Run flux convergence test and plot results, as well as apertured image
        apf.flux_convergence_test(object_coordinates, img_data, aperture_steps, img_rgain, img_rnoise,
                                  coord_offset=offsets[indx], show_plot=True, show_aperture_plot=True)

        # Prompt to delete currently selected image and data associated
        inpt = input('Do you want to delete data related to the previous images? Input Y to do so')
        if inpt == 'Y':
            time = np.delete(time, indx)
            ceph_flux = np.delete(ceph_flux, indx)
            calibration_check = np.delete(calibration_check, indx)
            del filenames[indx]
            del offsets[indx]

        # Prompt to quit loop
        inpt = input('Do you wish to repeat this selection procedure? Input N to stop the loop')
        if inpt == 'N':
            loop = False

    # Clear data file
    open(os.path.join('cepheid_data', 'cepheid.dat'), 'w').close()

    # Write to data file
    for k in range(0, len(ceph_flux)):
        content = str(ceph_flux[k]) + '   ' + str(time[k]) + '\n'
        with open(os.path.join('cepheid_data', 'cepheid.dat'), 'a') as f:
            f.write(content)

    # Clear data file
    open(os.path.join('cepheid_data', 'calibrationstar.dat'), 'w').close()

    # Write to data file
    for k in range(0, len(calibration_check)):
        content = str(calibration_check[k]) + '   ' + str(time[k]) + '\n'
        with open(os.path.join('cepheid_data', 'calibrationstar.dat'), 'a') as f:
            f.write(content)

    data_exists = True


# # Cepheid data analysis

pguess = apf.periodlinreg(ceph_flux, calibration_check, time)

# # Phaseplot used to find period
apf.phaseplot(ceph_flux, time, pguess=pguess)

