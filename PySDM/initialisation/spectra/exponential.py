from scipy.stats import expon
from ..impl.spectrum import Spectrum

class Exponential(Spectrum):

    def __init__(self, norm_factor, scale):
        super().__init__(expon, (
            0,     # loc
            scale  # scale = 1/lambda
        ), norm_factor)
