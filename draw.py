import seaborn as sns


palette = sns.color_palette('coolwarm', as_cmap=True)
# custom diverging palette from red to blue
# palette = sns.diverging_palette(240, 10, as_cmap=True)


def get_weight_range(weights):
    """ Max absolute value in a list of floats.
    """
    abs_weights = [abs(weight) for weight in weights]
    return max(abs_weights) if abs_weights else 0


# def get_weight_opacity(weight, weight_range):
#     """ Return opacity value for given weight as a string.
#     """
#     min_opacity = 0.8
#     if np.isclose(weight, 0) and np.isclose(weight_range, 0):
#         rel_weight = 0.0
#     else:
#         rel_weight = abs(weight) / weight_range
#     return f"{min_opacity + (1 - min_opacity) * rel_weight:.2f}"


def get_weight_rgb(weight, weight_range):
    # normalize the weight to range [-1, 1]
    normalized_weight = weight / weight_range
    # convert the normalized weight to range [0, 1] for color mapping
    normalized_weight = (normalized_weight + 1) / 2
    # get the rgb color from the palette
    rgb_color = palette(normalized_weight)
    # convert the rgb color to a CSS color string
    css_color = f'rgb({rgb_color[0]*100:.0f}%, {rgb_color[1]*100:.0f}%, {rgb_color[2]*100:.0f}%)'
    return css_color