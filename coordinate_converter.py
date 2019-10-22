import numpy as np
import matplotlib.pyplot as plt


def localhourangle(local_sidereal_time, right_ascension):
    """
    Calculates local hour angle from LST and RA.

    Parameters
    ----------
    local_sidereal_time : list, tuple, np.ndarray
        Local sidereal time in hours, minutes and seconds

    right_ascension : list, tuple, np.ndarray
        Right ascension in hours, minutes and seconds

    Returns
    -------
    np.ndarray
        Local hour angle in hours
    """

    # Convert variables to arrays and check size
    for item in [local_sidereal_time, right_ascension]:
        if isinstance(item, (list, tuple, np.ndarray)):
            item = np.asarray(item)
        else:
            raise TypeError('Item must be a list, tuple or array, not %s' % type(item))
        if item.shape != (1, 3) or (3, 1):
            raise TypeError('Item has wrong dimensions, should be (1, 3) or (3, 1), not %s' % item.shape)

    # Check if variables are same shape
    if local_sidereal_time.shape != right_ascension.shape:
        print('lst', local_sidereal_time.shape)
        print('ra', right_ascension.shape)
        raise TypeError('The two variables have different dimensions')

    # Do the thing
    lha_hms = local_sidereal_time - right_ascension

    # Convert from h,m,s to hours with decimal
    [hours, minutes, seconds] = [lha_hms[0], lha_hms[1], lha_hms[2]]
    local_hour_angle = hours + minutes/60 + seconds/3600

    return local_hour_angle


def altazim(hourangle, latitude, declination, radian=False, harad=False):
    """
    Convert equatorial coordinates to altitude-azimuthal.

    Parameters
    ----------
    hourangle : list, tuple, np.ndarray
        Local Hour Angle = Local Sidereal Time - Right Ascension. Assumed in hours.

    latitude : list, tuple, np.ndarray
        Latitude angle in reference to Polaris. Assumed in degrees.

    declination : list, tuple, np.ndarray
        Declination on the celestial sphere, equatorial coordinates. Assumed in degrees.

    radian : bool
        Optional parameter. Default is False, meaning that angles are assumed to be in degrees.
        If set to True, latitude and declination angles is assumed to be in radians.

    harad : bool
        Optional parameter. Default is False, meaning that hourangle is assumed to be in hours.
        If set to True, hourangle is assumed to be in radians.

    Returns
    -------
    [np.ndarray, np.ndarray]
        Returns a list [altitude, azimuthal] with the two alt-az coordinates altitude and azimuthal, as numpy arrays.

    """

    # Convert variables to arrays
    for item in [hourangle, latitude, declination]:
        if isinstance(item, (list, tuple, np.ndarray)):
            pass
        else:
            raise TypeError('Item must be a list, tuple or array, not %s' % type(item))
    hourangle = np.asarray(hourangle)
    latitude = np.asarray(latitude)
    declination = np.asarray(declination)

    # Conversion factor from degree to radians
    deg_to_rad = np.pi/180

    # Option stating that angles are in radians and should not be converted
    if radian is False:
        latitude = deg_to_rad * latitude
        declination = deg_to_rad * declination

    # Option stating if hour time angle is already in radians and should not be converted
    if harad is False:
        hourangle = deg_to_rad * 15 * hourangle

    # Calculate sine and cosine values
    [sindec, sinlat, sinha] = [np.sin(declination), np.sin(latitude), np.sin(hourangle)]
    [cosdec, coslat, cosha] = [np.cos(declination), np.cos(latitude), np.cos(hourangle)]

    # Calculate altitude from equatorial coordinates
    sinalt = sindec * sinlat + cosdec * coslat * cosha
    altitude = np.arcsin(sinalt)

    # Calculate azimuthal from equatorial and altitude
    cosalt = np.cos(altitude)
    cosazi = - sinha * sindec / cosalt
    azimuthal = np.arccos(cosazi)

    # Convert to degrees
    altitude = altitude / deg_to_rad
    azimuthal = azimuthal / deg_to_rad

    return [altitude, azimuthal]


# # Plot altitude as a function of local hour angle for both declinations and zenith
lha = [np.linspace(0, 24, 1000), np.linspace(0, 24, 1000)]
dec = [np.linspace(20, 20, 1000), np.linspace(-10, -10, 1000)]
dec_zenith = np.linspace(56, 56, 1000)
lat = [np.linspace(56, 56, 1000), np.linspace(56, 56, 1000)]
lat_zenith = np.linspace(56, 56, 1000)
hline = np.linspace(0, 0, 1000)

[alt, azi] = altazim(lha, lat, dec)

lha = np.asarray(lha)
lha1 = lha[0, :]
lha2 = lha[1, :]
alt1 = alt[0, :]
alt2 = alt[1, :]
[alt_zenith, azi_zenith] = altazim(lha1, lat_zenith, dec_zenith)

plt.plot(lha1, alt1)
plt.plot(lha2, alt2)
plt.plot(lha1, hline, 'k--')
plt.plot(lha1, alt_zenith, '--')
plt.xlabel('Hour angle from 0 to 24 hours')
plt.xlim([0, 24])
plt.ylabel('Altitude in degrees')
plt.legend(['Declination 20 degrees', 'Declination -10 degrees', 'Horizon', 'Zenith declination'])
plt.show()

# # Amount of hours above horizon
hours_per_plotpoint = 24/1000

# Find amount of points with altitude > 0
pos_amnt1 = np.sum(alt1 > 0)
pos_amnt2 = np.sum(alt2 > 0)

# Calculate hours
hours_above_horizon_1 = pos_amnt1 * hours_per_plotpoint
hours_above_horizon_2 = pos_amnt2 * hours_per_plotpoint

# Print
print('Hours above horizon for declination 20 is', hours_above_horizon_1)
print('Hours above horizon declination -10 is', hours_above_horizon_2)
