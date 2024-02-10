# explanations

![Screenshot of the heatmaps](readme_lime_heatmaps.png)

Generating explanations for TACA's XGBoost ClassifierChains trained on the restaurant reviews dataset.

## Setup

Install libraries in a new conda environment. 

### On Mac:

```
conda env create -f envs/expenv_mac.yml
```

Install required homebrew packages:
```
brew install pkg-config
brew install mysql
export LDFLAGS="-L/usr/local/opt/openssl/lib"
export CPPFLAGS="-I/usr/local/opt/openssl/include"
```

### On Windows:

```
conda env create -f envs/expenv_win.yml
```
### Fix OmniXAI installation:
Follow [this guide](https://github.com/tongshuangwu/polyjuice/issues/12#issuecomment-1665358584) (envs/expenv/Lib/site-packages/polyjuice/generations/generator_helpers.py).

### Activate conda environment:

```
conda activate expenv
```

### Download spaCy models:
```
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
```

## Generate explanations

Train the model (model.sav saved in the model folder):
```
python train.py
```

Sample sentences and generate heatmaps (html and json saved in the results folder):

```
python generate.py
```