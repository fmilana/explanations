# explanations

![Screenshot of the heatmaps](lime_heatmaps.png)

Generating LIME heatmaps on TACA's classifier and restaurant reviews dataset.

## How to generate heatmaps

1) Install libraries in a new conda environment. On Mac:

```
conda env create -f limeenv_mac.yml
```

On Windows:

```
conda env create -f limeenv_win.yml
```

2) Activate conda environment:

```
conda activate limeenv
```

3) Run main.py (results stored in .html and .txt files in the results folder):

```
python main.py
```