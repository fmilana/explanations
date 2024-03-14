import re
import json
from draw import get_weight_range, get_weight_rgba


def _add_style(html_path):
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(f'<style> body {{font-family: arial; text-align: center;}}</style>\n')


def _add_title(title,  html_path):
    with open(html_path, 'a+', encoding='utf-8') as f:
        f.write(f'<h1>{title}</h1>\n')


def _add_section(sentence, words, interpret_tokens, score, lime_weights, shap_weights, occlusion_weights, interpret_weights, is_query, html_path):
    with open(html_path, 'a+', encoding='utf-8') as f:
        f.write(f'<h2>score: {score:.2f}</h2>\n')
        if is_query:
            f.write(f'<p style="color: black; font-family: Arial; text-align: center; line-height: 2; margin: 0;">"{sentence}"</p>')
            f.write('\n<br><br>\n')
        else:
            f.write('<h3>ORIGINAL</h3>\n')
            f.write(f'<p style="color: black; font-family: Arial; text-align: center; line-height: 2; margin: 0;">"{sentence}"</p>')
            f.write('\n<br><br>\n')
            f.write('<h3>LIME</h3>\n')
            f.write(_get_sentence_html(words, lime_weights))
            f.write('\n<br><br>\n')
            f.write('<h3>SHAP</h3>\n')
            f.write(_get_sentence_html(words, shap_weights))
            f.write('\n<br><br>\n')
            f.write('<h3>OCCLUSION</h3>\n')
            f.write(_get_sentence_html(words, occlusion_weights))
            f.write('\n<br><br>\n')
            f.write('<h3>TRANSFORMERS-INTERPRET</h3>\n')
            f.write(_get_sentence_html(interpret_tokens, interpret_weights, is_interpret=True))
        

def _get_html_span(token_or_word, weight, weight_range):
    border_bottom_color = get_weight_rgba(weight, weight_range)
    background_color = re.sub(r'1\.0\)$', '0.2)', border_bottom_color)
    return f'<span style="border-bottom: 5px solid {border_bottom_color}; background-color: {background_color}; padding-bottom: 1px;">{token_or_word}</span>'


def _get_sentence_html(words_or_tokens, weights, is_interpret=False):
    weight_index = 0

    weight_range = get_weight_range(weights)

    sentence_html = '<p style="color: black; font-family: Arial; text-align: center; line-height: 2; margin: 0;">"'

    if is_interpret:
        for i, token in enumerate(words_or_tokens):
            sentence_html += _get_html_span(token, weights[i], weight_range)
    else:
        for word in words_or_tokens:
            # only clean if word is a word
            if re.search(r'\w', word):
                cleaned_word = re.sub(r'\W', '', word)
            else:
                cleaned_word = word

            try:
                weight = weights[weight_index]

                if weight != 0.0:
                    cleaned_word = cleaned_word.replace(' ', '&nbsp;')
                    html_span = _get_html_span(cleaned_word, weight, weight_range)
                    word = word.replace(cleaned_word, html_span)
                    sentence_html += f'{word}'
                else:
                    sentence_html += f'{word}'
            
            except (IndexError, ZeroDivisionError):
                sentence_html += f'{word}'
            
            weight_index += 1

    sentence_html += '"</p>'

    return sentence_html


def _generate_file(results_json, html_path):
    # add style
    _add_style(html_path)

    for key, value in results_json.items():
        _add_title(key, html_path)

        sentence = value['sentence']
        score = value['classification_score']
        parts = value['parts']
        interpret_parts = value['interpret_parts']

        words = [part['word'] for part in parts]
        lime_weights = [part['lime_weight'] for part in parts]
        shap_weights = [part['shap_weight'] for part in parts]
        occlusion_weights = [part['occlusion_weight'] for part in parts]
        interpret_tokens = [part['cleaned_token'] for part in interpret_parts]
        interpret_weights = [part['interpret_weight'] for part in interpret_parts]

        is_query = key.endswith('Query')

        _add_section(sentence, words, interpret_tokens, score, lime_weights, shap_weights, occlusion_weights, interpret_weights, is_query, html_path)


if __name__ == '__main__':
    try:
        results_json = json.load(open('results/json/results.json', 'r', encoding='utf-8'))
        print('Generating HTML...')
        _generate_file(results_json, 'results/html/results.html')
        print('HTML saved in results/html/results.html')
    except FileNotFoundError:
        print('JSON not found. Please run generate_json.py first.')