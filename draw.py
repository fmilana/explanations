import re
import numpy as np
from pathlib import Path
from preprocess import get_stop_words


def create_html(sentence, predicted_category, predicted_category_proba, lime_bias, lime_weights, shap_weights):
    # words = re.sub(r'\W', ' ', sentence).split()
    words = sentence.split()

    html_path = Path("results/html/results.html")
    stop_words = get_stop_words()

    print(f'lime bias: {lime_bias}')
    print(f'{len(lime_weights)} lime weights: {lime_weights}')
    print(f'{len(shap_weights)} shap_weights: {shap_weights}')

    with open(html_path, 'w+') as f:
        f.write(f'<h1>{predicted_category}</h1>\n')
        f.write(f'<h2>{predicted_category_proba}</h2><br><br>\n')
        f.write('<h3>LIME</h3>\n')
        draw_sentence(words, stop_words, lime_weights, f)
        f.write('\n<br><br>\n')
        f.write('<h3>SHAP</h3>\n')
        draw_sentence(words, stop_words, shap_weights, f)
        

def draw_sentence(words, stop_words, weights, f):
    weight_index = 0

    weight_range = get_weight_range(weights)

    for word in words:
        cleaned_word = re.sub(r'\W', '', word)

        if cleaned_word.lower() in stop_words:
            f.write(f'<span>{word}</span> ')
        else:
            weight = weights[weight_index]

            f.write(f'<span '
            'style="background-color: {color}; opacity: {opacity}" '
            'title="{weight:.3f}"'
            '>{token}</span> '.format(
                color=format_hsl(
                    weight_color_hsl(weight, weight_range, min_lightness=0.6)),
                opacity=_weight_opacity(weight, weight_range),
                weight=weights[weight_index],
                token=word))
            
            weight_index += 1


# https://eli5.readthedocs.io/en/latest/_modules/eli5/formatters/html.html
def weight_color_hsl(weight, weight_range, min_lightness=0.8):
    """ Return HSL color components for given weight,
    where the max absolute weight is given by weight_range.
    """
    hue = 120 if weight > 0 else 0
    saturation = 1
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