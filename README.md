# explanations

![Screenshot of the heatmaps](readme_screenshot.png)

LIME, SHAP and Occlusion heat maps for XGBoost ClassifierChains.

## Setup

Install libraries in a new conda environment. 

<!-- ### On Mac:

```
conda env create -f envs/expenv_mac.yml
```

Install required homebrew packages:
```
brew install pkg-config
brew install mysql
export LDFLAGS="-L/usr/local/opt/openssl/lib"
export CPPFLAGS="-I/usr/local/opt/openssl/include"
``` -->

### On Windows:

```
conda env create -f envs/expenv_bert_win.yml
```
### Fix OmniXAI installation:
Follow [this guide](https://github.com/tongshuangwu/polyjuice/issues/12#issuecomment-1665358584) (envs/expenv/Lib/site-packages/polyjuice/generations/generator_helpers.py).

### Activate conda environment:

```
conda activate expenv-bert
```

### Install PyTorch with CUDA:

https://pytorch.org/get-started/locally/


### Download spaCy models:
```
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
```

## Generate explanations

1. Train the classifier (saved in models/final/):
```
python train.py
```

2. Sample sentences (saved in results/samples.csv):
```
python sample.py
```

3. Generate JSON (saved in results/json/results.json):

```
python generate_json.py
```

4. Generate HTML from JSON (saved in results/html/results.html):
```
python generate_html.py
```