# # Aperture-Photometry
Code and image data used for aperture photometry by the author. Specialized for a specific data set, which is included, but
be compatible with others with small modifications.

Primary files consists of the runnable script aperture_photometry.py, and the file aperture_photometry_functions.py containing
most of the functions used by the script.

Otherwise depends on the following files:
- offsets.txt (must be present in root folder and include an x,y offset for every image to be analyzed (when comparing with
   a reference image)
- o4201193.10.fts (must be present in root folder. Reference image. Can be replaced with another, if name is changed in the
     script to account for this)
- detect_peaks.py (file with a function used. Author of this file is Marcos Duarte,https://github.com/demotu/BMC)
- The folder cepheid_data including the following files:
        - filenames.txt (a file with a list of filenames for all the image fits files to be analyzed)
        - Image files to be analyzed
