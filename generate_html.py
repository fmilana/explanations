import re
import json
from draw import get_weight_range, get_weight_rgba


def _add_style(html_paths):
    for html_path in html_paths:
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(f'<style> body {{font-family: arial; text-align: center;}}</style>\n')


def _add_title(title,  html_paths):
    for html_path in html_paths:
        with open(html_path, 'a', encoding='utf-8') as f:
            f.write(f'<h1>{title}</h1>\n')


def _add_section(sentence, tokens, score, lime_weights, shap_weights, occlusion_weights, html_paths):
    with open(html_paths[0], 'a', encoding='utf-8') as f:
        f.write(f'<h2>score: {score:.2f}</h2>\n')
        # Write to original.html
        with open(html_paths[1], 'a', encoding='utf-8') as orig_file:
            orig_file.write('<h3>ORIGINAL</h3>\n')
            orig_file.write(f'<p style="color: black; font-family: Arial; text-align: center; line-height: 2; margin: 0;">"{sentence}"</p>')
            orig_file.write('\n<br><br>\n')

        # Append original sentence to the main HTML file (output.html)
        f.write('<h3>ORIGINAL</h3>\n')
        f.write(f'<p style="color: black; font-family: Arial; text-align: center; line-height: 2; margin: 0;">"{sentence}"</p>')
        f.write('\n<br><br>\n')

        # Write to lime.html
        lime_html = _get_sentence_html(tokens, lime_weights)
        with open(html_paths[2], 'a', encoding='utf-8') as lime_file:
            lime_file.write('<h3>LIME</h3>\n')
            lime_file.write(lime_html)
            lime_file.write('\n<br><br>\n')

        # Append LIME to the main HTML file (output.html)
        f.write('<h3>LIME</h3>\n')
        f.write(lime_html)
        f.write('\n<br><br>\n')

        # Write to shap.html
        shap_html = _get_sentence_html(tokens, shap_weights)
        with open(html_paths[3], 'a', encoding='utf-8') as shap_file:
            shap_file.write('<h3>SHAP</h3>\n')
            shap_file.write(shap_html)
            shap_file.write('\n<br><br>\n')

        # Append SHAP to the main HTML file
        f.write('<h3>SHAP</h3>\n')
        f.write(shap_html)
        f.write('\n<br><br>\n')

        # Write to occlusion.html
        occlusion_html = _get_sentence_html(tokens, occlusion_weights)
        with open(html_paths[4], 'a', encoding='utf-8') as occlusion_file:
            occlusion_file.write('<h3>OCCLUSION</h3>\n')
            occlusion_file.write(occlusion_html)
            occlusion_file.write('\n<br><br>\n')

        # Append Occlusion to the main HTML file (output.html)
        f.write('<h3>OCCLUSION</h3>\n')
        f.write(occlusion_html)
        f.write('\n<br><br>\n')
        

def _get_sentence_html(tokens, weights):
    weight_index = 0

    weight_range = get_weight_range(weights)

    sentence_html = '<p style="color: black; font-family: Arial; text-align: center; line-height: 2; margin: 0;">"'

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


def _generate_files(results_json, html_paths):    
    _add_style(html_paths)

    total_items = len(results_json)
    progress_counter = 0

    for key, value in results_json.items():
        _add_title(key, html_paths)

        sentence = value['sentence']
        score = value['classification_score']
        parts = value['parts']

        tokens = [part['token'] for part in parts]
        lime_weights = [part['lime_weight'] for part in parts]
        shap_weights = [part['shap_weight'] for part in parts]
        occlusion_weights = [part['occlusion_weight'] for part in parts]

        _add_section(sentence, tokens, score, lime_weights, shap_weights, occlusion_weights, html_paths)

        print(f'{progress_counter+1}/{total_items} sentences added.', end='\r')
        progress_counter += 1


if __name__ == '__main__':
    try:
        results_json = json.load(open('output/json/output.json', 'r', encoding='utf-8'))
        html_paths = ['output/html/output.html', 'output/html/original.html', 'output/html/lime.html', 'output/html/shap.html', 'output/html/occlusion.html']
        print('Generating HTML files...')
        _generate_files(results_json, html_paths)
        print('HTML files saved in output/html/')
    except FileNotFoundError:
        print('JSON not found. Please run generate_json.py first.')