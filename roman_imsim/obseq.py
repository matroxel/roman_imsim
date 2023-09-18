import galsim.roman as roman
from astropy.time import Time
import galsim
import galsim.config
from galsim.config import RegisterObjectType,RegisterInputType

filter_dither_dict_ = {
    1:'R062',
    2:'Z087',
    3:'Y106',
    4:'J129',
    5:'H158',
    6:'F184',
    7:'K213',
    8:'W146'
}

class ObSeqDataLoader(object):
    """Read the exposure information from the observation sequence.
    """
    def __init__(self, file_name, visit=None, logger=None):
        self.logger = galsim.config.LoggerWrapper(logger)
        self.file_name = file_name
        self.visit = visit

        try:
            self.read_obseq()
        except:
            # Read visit info from the config file.
            self.logger.warning('Reading visit info from config file.')

            req = {'ra': float, 'dec': float, 'pa': float, 'date': None, 'filter': str}
            opt = {'exptime' : float}
            params, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt) 

            self.ob['ra']      = params['ra'] 
            self.ob['dec']     = params['dec']
            self.ob['pa']      = params['pa'] 
            self.ob['date']    = params['date'] 
            self.ob['mjd']     = Time(params['date'], format='datetime').mjd
            self.ob['filter_'] = params['filter']
            if exptime in params:
                self.ob['exptime'] = params['exptime']
            else:
                self.ob['exptime'] = roman.exptime

    def read_obseq(self):
        """Read visit info from the obseq file."""
        if self.file_name is None:
            raise ValueError('No obseq filename provided, trying to build from config information.')
        if self.visit is None:
            raise ValueError('The visit must be set when reading visit info from an obseq file.')

        self.logger.warning('Reading info from obseq file %s for visit %s',
                            self.file_name, self.visit)

        ob = fio.FITS(self.file_name)[-1][self.visit]

        self.ob['ra']      = ob['ra'] 
        self.ob['dec']     = ob['dec']
        self.ob['pa']      = ob['pa'] 
        self.ob['date']    = Time(ob['date'],format='mjd').datetime 
        self.ob['mjd']     = ob['date']
        self.ob['filter_'] = filter_dither_dict_[ob['filter']]
        self.ob['exptime'] = ob['exptime']

    def get_data(self):
        return self.ob

def ObSeqData(config, base, ignore, gsparams, logger):
    """Returns the obseq data for a pointing.
    """
    pointing = galsim.config.GetInputObj('obseq_data', config, base, 'OpSeqDataLoader')
    return pointing.get_data(), False

RegisterInputType('obseq_data', OpSeqDataLoader())
RegisterObjectType('ObSeqData', ObSeqData, input_type='obseq_data')