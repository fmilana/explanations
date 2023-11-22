# explanations

![Screenshot of the heatmaps](readme_lime_heatmaps.png)

Generating LIME heatmaps on TACA's classifier and restaurant reviews dataset.

## How to generate heatmaps

1) Install libraries in a new conda environment. On Mac:

```
conda env create -f expenv_mac.yml
```

On Windows:

```
conda env create -f expenv_win.yml
```

2) Activate conda environment:

```
conda activate expenv
```

Run LIME (results stored in lime.txt and lime.html files in results/lime/):

```
python run_lime.py
```

Run SHAP (results stored in shap.html and shap.png in results/shap/):
```
python run_shap.py
```

Run Anchors (results stored in anchors.txt in results/anchors/):
```
python run_anchors.py
```