import re
import json
from draw import get_weight_range, get_weight_rgba


def _add_style(html_path):
    with open(html_path, 'a+', encoding='utf-8') as f:
        f.write(f'<style> body {{font-family: arial; text-align: center;}}</style>\n')


def _add_title(title,  html_path):
    with open(html_path, 'a+', encoding='utf-8') as f:
        f.write(f'<h1>{title}</h1>\n')


def _add_section(tokens, proba, lime_weights, shap_weights, occlusion_weights, query, html_path):
    with open(html_path, 'a+', encoding='utf-8') as f:
        f.write(f'<h2>score: {proba:.2f}</h2>\n')
        if query:
            f.write(f'<p style="color: black; font-family: Arial; text-align: center; line-height: 2.5;">"{"".join(tokens)}"</p>')
            f.write('\n<br><br>\n')
        else:
            f.write('<h3>ORIGINAL</h3>\n')
            f.write(f'<p style="color: black; font-family: Arial; text-align: center; line-height: 2.5;">"{"".join(tokens)}"</p>')
            f.write('\n<br><br>\n')
            f.write('<h3>LIME</h3>\n')
            f.write(_get_sentence_html(tokens, lime_weights))
            f.write('\n<br><br>\n')
            f.write('<h3>SHAP</h3>\n')
            f.write(_get_sentence_html(tokens, shap_weights))
            f.write('\n<br><br>\n')
            f.write('<h3>OCCLUSION</h3>\n')
            f.write(_get_sentence_html(tokens, occlusion_weights))
            f.write('\n<br><br>\n')
        

def _get_sentence_html(tokens, weights):
    weight_index = 0

    weight_range = get_weight_range(weights)

    sentence_html = '<p style="color: black; font-family: Arial; text-align: center; line-height: 2.5;">"'

    for token in tokens:
        # only clean if token is a word
        if re.search(r'\w', token):
            cleaned_token = re.sub(r'\W', '', token)
        else:
            cleaned_token = token

        try:
            weight = weights[weight_index]

            if weight != 0.0:
                cleaned_token = cleaned_token.replace(' ', '&nbsp;')
                border_bottom_color = get_weight_rgba(weight, weight_range)
                background_color = re.sub(r'1\.0\)$', '0.2)', border_bottom_color)
                html_span = f'<span style="border-bottom: 5px solid {border_bottom_color}; background-color: {background_color}; padding-bottom: 1px;">{cleaned_token}</span>'
                token = token.replace(cleaned_token, html_span)
                sentence_html += f'{token}'
            else:
                sentence_html += f'{token}'
        
        except (IndexError, ZeroDivisionError):
            sentence_html += f'{token}'
        
        weight_index += 1

    sentence_html += '"</p>'

    return sentence_html


def _generate_file(results_json, html_path):
    # clear html
    with open(html_path, 'w', encoding='utf-8') as f:
        pass

    # add style
    _add_style(html_path)

    for key, value in results_json.items():
        _add_title(key, html_path)

        proba = value['classification_score']
        parts = value['parts']

        tokens = [part['token'] for part in parts]
        lime_weights = [part['lime_weight'] for part in parts]
        shap_weights = [part['shap_weight'] for part in parts]
        occlusion_weights = [part['occlusion_weight'] for part in parts]

        query = key.endswith('Query')

        _add_section(tokens, proba, lime_weights, shap_weights, occlusion_weights, query, html_path)


if __name__ == '__main__':
    try:
        results_json = json.load(open('results/json/results.json', 'r', encoding='utf-8'))
        print('Generating HTML...')
        _generate_file(results_json, 'results/html/results.html')
        print('HTML saved in results/html/results.html')
    except FileNotFoundError:
        print('JSON not found. Please run generate_json.py first.')