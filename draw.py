import re
from pathlib import Path
from preprocess import get_stop_words


def create_html(sentence, lime_bias, lime_values, shap_values):
    print(f'sentence = {sentence}')
    html_path = Path("results/html/results.html")
    stopwords = get_stop_words()

    print(f'stopwords = {stopwords}')

    with open(html_path, 'w+') as f:
        for word in re.sub(r'\W', ' ', sentence).split():
            print(f'word = {word}')
            if word.lower() in stopwords:
                f.write(f'<span>{word}</span> ')
            else:
                f.write(f'<span>{word.upper()}</span> ')