# explanations

![Screenshot of the heatmaps](readme_lime_heatmaps.png)

Generating explanations for TACA's classifier trained on the restaurant reviews dataset.

## Setup

Install libraries in a new conda environment. 

### On Mac:

```
conda env create -f expenv_mac.yml
```

To fix OmniXAI installation:
```
brew install pkg-config
brew install mysql
export LDFLAGS="-L/usr/local/opt/openssl/lib"
export CPPFLAGS="-I/usr/local/opt/openssl/include"
python -m spacy download en_core_web_sm
```
Then follow [this guide](https://github.com/tongshuangwu/polyjuice/issues/12#issuecomment-1665358584).

### On Windows:

```
conda env create -f expenv_win.yml
```

### Activate conda environment:

```
conda activate expenv
```

## Generate explanations

To generate explanations (output stored in the results folder):
```
python run_[explanation_name].py
```

explanation_names:
- lime
- shap
- anchors
- counter