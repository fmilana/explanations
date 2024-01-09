import re
from pathlib import Path
from preprocess import get_stop_words


def create_html(sentence, predicted_category, predicted_category_proba, lime_bias, lime_values, shap_values):
    words = re.sub(r'\W', ' ', sentence).split()

    html_path = Path("results/html/results.html")
    stop_words = get_stop_words()

    print(f'len(lime_values) = {len(lime_values)}')
    print(f'len(shap_values) = {len(shap_values)}')

    with open(html_path, 'w+') as f:
        f.write(f'<h1>{predicted_category}</h1><br>')
        f.write(f'<h2>{predicted_category_proba}</h2><br><br>')
        draw_sentence(words, stop_words, lime_values, f)
        f.write('<br><br>')
        draw_sentence(words, stop_words, shap_values, f)
        

def draw_sentence(words, stop_words, values, f):
    cleaned_word_index = 0
    for word in words:
        if word.lower() in stop_words:
            f.write(f'<span>{word}</span> ')
        else:
            print(f'indexing values for word {word} with index {cleaned_word_index}')
            f.write(f'<span>{word} ({values[cleaned_word_index]})</span> ')
            cleaned_word_index += 1
            