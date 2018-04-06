import numpy as np
import astropy.units as U
from astropy import wcs

HIfreq = 1.420405751E9 * U.Hz

class DataCube():

    def __init__(
            self, 
            n_px_x = 256, 
            n_px_y = 256, 
            n_channels = 64, 
            px_size = 15. * U.arcsec, 
            channel_width = 4. * U.km * U.s ** -1,
            velocity_centre = 0. * U.km * U.s ** -1,
            ra = 0. * U.deg,
            dec = 0. * U.deg
    ):

        datacube_unit = U.Jy * U.pix ** -2
        self._array = np.zeros((n_px_x, n_px_y, n_channels, 1)) * datacube_unit
        self.n_px_x, self.n_px_y, self.n_channels = n_px_x, n_px_y, n_channels
        self.px_size = px_size
        self.channel_width = channel_width
        self.velocity_centre = velocity_centre
        self.ra = ra
        self.dec = dec
        self.padx = 0
        self.pady = 0
        self.wcs = wcs.WCS(naxis=3)
        self.wcs.wcs.crpix = [
            self.n_px_x / 2. + .5, 
            self.n_px_y / 2. + .5, 
            self.n_channels // 2
        ]
        self.units = [U.deg, U.deg, U.m * U.s ** -1]
        self.wcs.wcs.cunit = [unit.to_string('fits') for unit in self.units]
        self.wcs.wcs.cdelt = [
            -self.px_size.to(self.units[0]).value, 
            self.px_size.to(self.units[1]).value, 
            self.channel_width.to(self.units[2]).value
        ]
        self.wcs.wcs.crval = [self.ra.to(self.units[0]).value, self.dec.to(self.units[1]).value, self.velocity_centre.to(self.units[2]).value]
        self.wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN', 'VELO-OBS']
        self.wcs = wcs.utils.add_stokes_axis_to_wcs(self.wcs, self.wcs.wcs.naxis)
        self._channel_mids()
        self._channel_edges()
        self._freq_channel_mode = False
            
        return

    def _channel_mids(self):
        self.channel_mids = self.wcs.wcs_pix2world(
            np.zeros(self.n_channels), 
            np.zeros(self.n_channels), 
            np.arange(self.n_channels), 
            np.zeros(self.n_channels),
            0
        )[2] * self.units[2]
        return

    def _channel_edges(self):
        self.channel_edges = self.wcs.wcs_pix2world(
            np.zeros(self.n_channels + 1), 
            np.zeros(self.n_channels + 1), 
            np.arange(self.n_channels + 1) - .5, 
            np.zeros(self.n_channels + 1),
            0
        )[2] * self.units[2]
        return

    def spatial_slices(self):
        return iter(self._array[..., 0].transpose((2, 0, 1)))

    def spectra(self):
        return iter(self._array[..., 0].reshape(self.n_px_x * self.n_px_y, self.n_channels))

    def freq_channels(self):
        if self._freq_channel_mode:
            return
        else:
            velocity_widths = np.diff(self.channel_edges)
            convert_to_Hz = lambda q: q.to(U.Hz, equivalencies=U.doppler_radio(HIfreq))
            self.wcs.wcs.cdelt[2] = np.abs(convert_to_Hz(self.wcs.wcs.cdelt[2] * self.units[2]) \
                                           - convert_to_Hz(0. * self.units[2])).value
            self.wcs.wcs.crval[2] = convert_to_Hz(self.wcs.wcs.crval[2] * self.units[2]).value
            self.wcs.wcs.ctype[2] = 'FREQ-OBS'
            self.units[2] = U.Hz
            self.wcs.wcs.cunit[2] = self.units[2].to_string('fits')
            self._channel_mids()
            self._channel_edges()
            self._freq_channel_mode = True
            return

    def velocity_channels(self):
        if not self._freq_channel_mode:
            return
        else:
            frequency_widths = np.diff(self.channel_edges)
            convert_to_ms = lambda q: q.to(U.m * U.s ** -1, equivalencies=U.doppler_radio(HIfreq))
            self.wcs.wcs.cdelt[2] = np.abs(convert_to_ms(self.wcs.wcs.cdelt[2] * self.units[2]) \
                                           - convert_to_ms(0. * self.units[2])).value
            self.wcs.wcs.crval[2] = convert_to_ms(self.wcs.wcs.crval[2] * self.units[2]).value
            self.wcs.wcs.ctype[2] = 'VELO-OBS'
            self.units[2] = U.m * U.s ** -1
            self.wcs.wcs.cunit[2] = self.units[2].to_string('fits')
            self._channel_mids()
            self._channel_edges()
            self._freq_channel_mode = False
            return


    def add_pad(self, pad):
        tmp = self._array
        self._array = np.zeros((self.n_px_x + pad[0] * 2, self.n_px_y + pad[1] * 2, self.n_channels, 1))
        self._array = self._array * tmp.unit
        self._array[pad[0]:-pad[0], pad[1]:-pad[1], ...] = tmp
        self.wcs.wcs.crpix += np.array([pad[0], pad[1], 0, 0])
        self.padx, self.pady = pad
        return
        
    def drop_pad(self):
        if (self.padx == 0) and (self.pady == 0):
            return
        self._array = self._array[self.padx:-self.padx, self.pady:-self.pady, ...]
        self.wcs.wcs.crpix -= np.array([self.padx, self.pady, 0, 0])
        self.padx, self.pady = 0, 0
        return

    def copy(self):
        copy = DataCube(
            self.n_px_x, 
            self.n_px_y, 
            self.n_channels, 
            self.px_size, 
            self.channel_width,
            self.velocity_centre,
            self.ra,
            self.dec
        )
        copy.padx, copy.pady = self.padx, self.pady
        copy.wcs = self.wcs
        copy._array = self._array.copy()
        return copy

    def __repr__(self):
        return self._array.__repr__()
