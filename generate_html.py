import re
import json
from preprocess import get_stop_words
from draw import get_weight_range, get_weight_rgba



def _add_style(html_path, font_family, line_height):
    with open(html_path, 'a+', encoding='utf-8') as f:
        f.write(f'<style>\nbody {{\nfont-family: {font_family};\n}}\nspan {{\nline-height: {line_height};\n}}\n</style>\n')


def _add_title(title,  html_path):
    with open(html_path, 'a+', encoding='utf-8') as f:
        f.write(f'<h1>{title}</h1>\n')


def _add_section(tokens, proba, lime_weights, shap_weights, occlusion_weights, html_path):
    stop_words = get_stop_words()

    with open(html_path, 'a+', encoding='utf-8') as f:
        f.write(f'<h2>score: {proba:.2f}</h2>\n')
        f.write('<h3>LIME</h3>\n')
        f.write(_get_sentence_html(tokens, stop_words, lime_weights))
        f.write('\n<br><br>\n')
        f.write('<h3>SHAP</h3>\n')
        f.write(_get_sentence_html(tokens, stop_words, shap_weights))
        f.write('\n<br><br>\n')
        f.write('<h3>OCCLUSION</h3>\n')
        f.write(_get_sentence_html(tokens, stop_words, occlusion_weights))
        f.write('\n<br><br>\n')
        

def _get_sentence_html(tokens, stop_words, weights):
    weight_index = 0

    weight_range = get_weight_range(weights)

    sentence_html = ''

    for token in tokens:
        cleaned_token = re.sub(r'\W', '', token)

        try:
            weight = weights[weight_index]
        
            if cleaned_token.lower() in stop_words or weight == 0.0:
                # html_span = f'<span style="background-color: #e0e0e0">{cleaned_token}</span>'
                html_span = f'<span>{cleaned_token}</span>'
            else:
                cleaned_token = cleaned_token.replace(' ', '&nbsp;')
                border_bottom_color = get_weight_rgba(weight, weight_range)
                background_color = re.sub(r'1\.0\)$', '0.2)', border_bottom_color)
                # opacity = get_weight_opacity(weight, weight_range)
                # background color
                # html_span = f'<span style="background-color: {background_color_half_opacity}" title="{weight}">{cleaned_token}</span>'
                # border-bottom
                # html_span = f'<span style="border-bottom: 6px solid {background_color}; padding-bottom: 1px;" title="{weight:.2f}">{cleaned_token}</span>'
                # border-bottom and background color
                html_span = f'<span style="border-bottom: 5px solid {border_bottom_color}; background-color: {background_color}; padding-bottom: 1px;">{cleaned_token}</span>'
                # border-bottom and border-top
                # html_span = f'<span style="border-bottom: 6px solid {background_color}; border-top: 5px solid {background_color}; padding-bottom: 1px; padding-top: 1px;" title="{weight:.2f}">{cleaned_token}</span>'
                # border-bottom and border-top and background color
                # html_span = f'<span style="border-bottom: 5px solid {background_color}; border-top: 5px solid {background_color}; background-color: {background_color_half_opacity}; padding-bottom: 1px; padding-top: 1px;" title="{weight:.2f}">{cleaned_token}</span>'

            token = token.replace(cleaned_token, html_span)
            sentence_html += f'{token}'
        except (IndexError, ZeroDivisionError):
            token = token.replace(cleaned_token, f'<span>{cleaned_token}</span>')
            sentence_html += f'{token}'
        
        weight_index += 1

    return sentence_html


def _generate_file(results_json, html_path):
    # clear html
    with open(html_path, 'w', encoding='utf-8') as f:
        pass

    # add style
    _add_style(html_path, font_family='Arial', line_height='3')

    for key, value in results_json.items():
        _add_title(key, html_path)

        proba = value['classification_score']
        parts = value['parts']

        tokens = [part['token'] for part in parts]
        lime_weights = [part['lime_weight'] for part in parts]
        shap_weights = [part['shap_weight'] for part in parts]
        occlusion_weights = [part['occlusion_weight'] for part in parts]

        _add_section(tokens, proba, lime_weights, shap_weights, occlusion_weights, html_path)


if __name__ == '__main__':
    try:
        print('Loading JSON...')
        results_json = json.load(open('results/json/results.json', 'r', encoding='utf-8'))
        print('JSON loaded.')
        print('Generating HTML...')
        _generate_file(results_json, 'results/html/results.html')
        print('HTML generated.')
    except FileNotFoundError:
        print('JSON not found. Please run generate_json.py first.')