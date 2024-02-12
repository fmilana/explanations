import numpy as np


# https://eli5.readthedocs.io/en/latest/_modules/eli5/formatters/html.html
def weight_color_hsl(weight, weight_range, min_lightness=0.8):
    """ Return HSL color components for given weight,
    where the max absolute weight is given by weight_range.
    """
    hue = 120 if weight > 0 else 0
    saturation = 1.0
    rel_weight = (abs(weight) / weight_range) ** 0.7
    lightness = 1.0 - (1 - min_lightness) * rel_weight
    return hue, saturation, lightness
            

def get_weight_range(weights):
    """ Max absolute value in a list of floats.
    """
    return max_or_0(abs(weight) for weight in weights)


def format_hsl(hsl_color):
    """ Format hsl color as css color string.
    """
    hue, saturation, lightness = hsl_color
    return 'hsl({}, {:.2%}, {:.2%})'.format(hue, saturation, lightness)


def _weight_opacity(weight, weight_range):
    """ Return opacity value for given weight as a string.
    """
    min_opacity = 0.8
    if np.isclose(weight, 0) and np.isclose(weight_range, 0):
        rel_weight = 0.0
    else:
        rel_weight = abs(weight) / weight_range
    return '{:.2f}'.format(min_opacity + (1 - min_opacity) * rel_weight)


def max_or_0(it):
    lst = list(it)
    return max(lst) if lst else 0