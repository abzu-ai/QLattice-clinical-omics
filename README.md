# QLattice Clinical Omics paper

Reproducible analysis and data for "Identifying molecular interactions in omics data for clinical biomarker discovery"
paper

## Reproduce results

### Create environment
We recommend using Virtualenv to create an environment.
We use python 3.8.

From a terminal in the root folder:

```
virtualenv venv -p 3.8
source venv/bin/activate
pip install -r requirements.txt
```

### Run notebooks

Under the folder notebooks there is a folder for each one of the four cases discussed in the paper.
All models, performance metrics, and figures can be reproduced by running the notebooks.


## QLattice tutorial

Below is a basic tutorial on how to use the QLattice to find models that relate the input variables of a dataset to its output variable. Its Jupyter notebook can be found in the `notebooks` folder of this repository. Other tutorials can be found in the [Feyn+QLattice documentation page](https://docs.abzu.ai/docs/tutorials/python/covid_mrna.html).

---
Finding AD biomarkers in proteomics data
---
<div style='visibility:hidden' id='notebook-meta' data-filetype='.csv' data-notebook='.ipynb'></div>

by: [Samuel Demharter and Meera Machado](https://www.abzu.ai/team)

Feyn version: 2.1+

Last updated: 23/09/2021

Can the QLattice deal with omics data that is noisy and contains thousands of features? It certainly can!

Omics data typically contains hundreds to thousands of features (proteins, transcripts, methylated DNA etc.) that are measured in samples derived from sources such as blood, tissue or cell culture. These types of approaches are often used for exploratory analysis e.g. in **biomarker discovery** or understanding the **mechanism of action** of a certain drug. It often resembles a bit of a "fishing exercise".

Thus, there is a need to quickly and reliably identify the most important **features** and their **interactions** that contribute to a certain signal (e.g. disease state, cell-type identity, cancer detection).

In this tutorial we present a brief workflow for building simple and interpretable models for proteomics data. This specific example is taken from a study by [Bader & Geyer et al. 2020](https://www.embopress.org/doi/full/10.15252/msb.20199356) (Mann group) and contains samples taken from the cerebrospinal fluid of Alzheimer Disease (AD) patients and non-AD patients. We will show you how to build `QLattice` model that can classify people into AD and non-AD according to their proteomic profiles.

The dataset contains over a thousand features (features in this example describe the intensity of different proteins measured by mass spectrometry).


```python
import numpy as np
import pandas as pd
import feyn

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
```

## Load the data
Note, the data has been preprocessed and missing values have been imputed.
It contains **1166 proteins** and **88 non-AD** and **49 AD** subjects.


```python
data = pd.read_csv("../data/ad_omics.csv")

# Let's record the categorical data types in our dataset (note features will be treated as numerical by default).
stypes = {}
for f in data.columns:
    if data[f].dtype =='object':
        stypes[f] = 'c'
```

## Split dataset into train and test set


```python
# Set random seed for reproducibility
random_seed = 42

# Define the target variable
target = "_clinical AD diagnosis"

# Split
train, test = train_test_split(data, test_size=0.33, stratify=data[target], random_state=random_seed)
```

## Train the QLattice

### Sample and fit models

This occurs in the following steps:

1. Sample models from the QLattice;
2. Fit the models by minimizing [BIC (Bayesian Information Criterion)](https://en.wikipedia.org/wiki/Bayesian_information_criterion);
3. Update the QLattice with the best models' structures;
4. Repeat the process;

This is all captured within the `auto_run` function


```python
# Connecting
ql = feyn.connect_qlattice()

# Reset and set random seed
ql.reset(random_seed=random_seed)

# Sample and fit models
models = ql.auto_run(
    data=train,
    output_name=target,
    kind='classification',
    stypes=stypes,
    n_epochs=20
    )
```


<?xml version="1.0" encoding="utf-8" ?>
<svg baseProfile="full" height="160.0" preserveAspectRatio="none" version="1.1" viewBox="0 0 471 160.0" width="471" xmlns="http://www.w3.org/2000/svg" xmlns:ev="http://www.w3.org/2001/xml-events" xmlns:xlink="http://www.w3.org/1999/xlink"><defs /><svg class="graph" height="155.0" width="471" x="0" y="0"><defs /><text fill="#1E1E1E" font-family="monospace" font-size="12" text-anchor="middle" x="406.0" y="99.0">Loss: 1.82E-01</text><text fill="#1E1E1E" font-family="monospace" font-size="12" text-anchor="start" x="0" y="145.0">Epoch no. 20/20 - Tried 23076 models - Completed in 1m 6s.</text><rect class="node" fill="#00F082" height="35" stroke="#1E1E1E" stroke-width="1" width="90" x="361" y="50.0"><title>_clinical AD diagnosis
logistic:
w=-4.8638
bias=-1.1859</title></rect><text fill="#1E1E1E" font-family="monospace" font-size="12" style="pointer-events:none" text-anchor="middle" x="406.0" y="71.5">_clinica..</text><text fill="#1E1E1E" font-family="monospace" font-size="10" style="pointer-events:none" text-anchor="end" x="449" y="59.0">0</text><text fill="#1E1E1E" font-family="monospace" font-size="10" style="pointer-events:none" text-anchor="start" x="363" y="59.0">out</text><line stroke="#1E1E1E" x1="331" x2="361" y1="67.5" y2="67.5" /><rect class="node" fill="#FAFAFA" height="35" stroke="#FF1EC8" stroke-width="1" width="90" x="241" y="50.0"><title>add</title></rect><text fill="#1E1E1E" font-family="monospace" font-size="12" style="pointer-events:none" text-anchor="middle" x="286.0" y="71.5">add</text><text fill="#1E1E1E" font-family="monospace" font-size="10" style="pointer-events:none" text-anchor="end" x="329" y="59.0">1</text><line stroke="#1E1E1E" x1="211" x2="241" y1="37.5" y2="58.5" /><line stroke="#1E1E1E" x1="211" x2="241" y1="97.5" y2="76.5" /><rect class="node" fill="#FAFAFA" height="35" stroke="#FF1EC8" stroke-width="1" width="90" x="121" y="20.0"><title>add</title></rect><text fill="#1E1E1E" font-family="monospace" font-size="12" style="pointer-events:none" text-anchor="middle" x="166.0" y="41.5">add</text><text fill="#1E1E1E" font-family="monospace" font-size="10" style="pointer-events:none" text-anchor="end" x="209" y="29.0">2</text><line stroke="#1E1E1E" x1="91" x2="121" y1="37.5" y2="28.5" /><line stroke="#1E1E1E" x1="91" x2="121" y1="97.5" y2="46.5" /><rect class="node" fill="#00F082" height="35" stroke="#1E1E1E" stroke-width="1" width="90" x="1" y="20.0"><title>MAPT
linear:
scale=0.000038
scale offset=24475.827038
w=-2.125441
bias=0.6458</title></rect><text fill="#1E1E1E" font-family="monospace" font-size="12" style="pointer-events:none" text-anchor="middle" x="46.0" y="41.5">MAPT</text><text fill="#1E1E1E" font-family="monospace" font-size="10" style="pointer-events:none" text-anchor="end" x="89" y="29.0">3</text><text fill="#1E1E1E" font-family="monospace" font-size="10" style="pointer-events:none" text-anchor="start" x="3" y="29.0">num</text><rect class="node" fill="#00F082" height="35" stroke="#1E1E1E" stroke-width="1" width="90" x="1" y="80.0"><title>GPC1
linear:
scale=0.000013
scale offset=121701.256437
w=-0.724680
bias=0.7295</title></rect><text fill="#1E1E1E" font-family="monospace" font-size="12" style="pointer-events:none" text-anchor="middle" x="46.0" y="101.5">GPC1</text><text fill="#1E1E1E" font-family="monospace" font-size="10" style="pointer-events:none" text-anchor="end" x="89" y="89.0">4</text><text fill="#1E1E1E" font-family="monospace" font-size="10" style="pointer-events:none" text-anchor="start" x="3" y="89.0">num</text><rect class="node" fill="#00F082" height="35" stroke="#1E1E1E" stroke-width="1" width="90" x="121" y="80.0"><title>NID2
linear:
scale=0.000013
scale offset=141003.958448
w=0.792789
bias=-1.3965</title></rect><text fill="#1E1E1E" font-family="monospace" font-size="12" style="pointer-events:none" text-anchor="middle" x="166.0" y="101.5">NID2</text><text fill="#1E1E1E" font-family="monospace" font-size="10" style="pointer-events:none" text-anchor="end" x="209" y="89.0">5</text><text fill="#1E1E1E" font-family="monospace" font-size="10" style="pointer-events:none" text-anchor="start" x="123" y="89.0">num</text></svg></svg>


## Inspect the top model


```python
best = models[0]
best.plot(train, test)
```

![html](QLattice_tutorial_files/model_plot.html)


With the plot below, we inspect the *Pearson correlation* between the values at each node and the true output:


```python
best.plot_signal(train)
```



<?xml version="1.0" encoding="utf-8" ?>
<svg baseProfile="full" height="200.0" preserveAspectRatio="none" version="1.1" viewBox="0 0 471 200.0" width="471" xmlns="http://www.w3.org/2000/svg" xmlns:ev="http://www.w3.org/2001/xml-events" xmlns:xlink="http://www.w3.org/1999/xlink"><defs /><svg class="graph" height="195.0" width="471" x="0" y="0"><defs /><rect class="node" fill="#1df190" height="35" stroke="#1E1E1E" stroke-width="1" width="90" x="361" y="50.0"><title>_clinical AD diagnosis
logistic:
w=-4.8638
bias=-1.1859</title></rect><text fill="#1E1E1E" font-family="monospace" font-size="12" style="pointer-events:none" text-anchor="middle" x="406.0" y="71.5">_clinica..</text><text fill="#1E1E1E" font-family="monospace" font-size="10" style="pointer-events:none" text-anchor="end" x="449" y="59.0">0</text><text fill="#1E1E1E" font-family="monospace" font-size="10" style="pointer-events:none" text-anchor="start" x="363" y="59.0">out</text><line stroke="#1E1E1E" x1="331" x2="361" y1="67.5" y2="67.5" /><rect class="node" fill="#fe4bd2" height="35" stroke="#1E1E1E" stroke-width="1" width="90" x="241" y="50.0"><title>add</title></rect><text fill="#1E1E1E" font-family="monospace" font-size="12" style="pointer-events:none" text-anchor="middle" x="286.0" y="71.5">add</text><text fill="#1E1E1E" font-family="monospace" font-size="10" style="pointer-events:none" text-anchor="end" x="329" y="59.0">1</text><line stroke="#1E1E1E" x1="211" x2="241" y1="37.5" y2="58.5" /><line stroke="#1E1E1E" x1="211" x2="241" y1="97.5" y2="76.5" /><rect class="node" fill="#fe4ed3" height="35" stroke="#1E1E1E" stroke-width="1" width="90" x="121" y="20.0"><title>add</title></rect><text fill="#1E1E1E" font-family="monospace" font-size="12" style="pointer-events:none" text-anchor="middle" x="166.0" y="41.5">add</text><text fill="#1E1E1E" font-family="monospace" font-size="10" style="pointer-events:none" text-anchor="end" x="209" y="29.0">2</text><line stroke="#1E1E1E" x1="91" x2="121" y1="37.5" y2="28.5" /><line stroke="#1E1E1E" x1="91" x2="121" y1="97.5" y2="46.5" /><rect class="node" fill="#fe52d4" height="35" stroke="#1E1E1E" stroke-width="1" width="90" x="1" y="20.0"><title>MAPT
linear:
scale=0.000038
scale offset=24475.827038
w=-2.125441
bias=0.6458</title></rect><text fill="#1E1E1E" font-family="monospace" font-size="12" style="pointer-events:none" text-anchor="middle" x="46.0" y="41.5">MAPT</text><text fill="#1E1E1E" font-family="monospace" font-size="10" style="pointer-events:none" text-anchor="end" x="89" y="29.0">3</text><text fill="#1E1E1E" font-family="monospace" font-size="10" style="pointer-events:none" text-anchor="start" x="3" y="29.0">num</text><rect class="node" fill="#fcb1e9" height="35" stroke="#1E1E1E" stroke-width="1" width="90" x="1" y="80.0"><title>GPC1
linear:
scale=0.000013
scale offset=121701.256437
w=-0.724680
bias=0.7295</title></rect><text fill="#1E1E1E" font-family="monospace" font-size="12" style="pointer-events:none" text-anchor="middle" x="46.0" y="101.5">GPC1</text><text fill="#1E1E1E" font-family="monospace" font-size="10" style="pointer-events:none" text-anchor="end" x="89" y="89.0">4</text><text fill="#1E1E1E" font-family="monospace" font-size="10" style="pointer-events:none" text-anchor="start" x="3" y="89.0">num</text><rect class="node" fill="#fbccf0" height="35" stroke="#1E1E1E" stroke-width="1" width="90" x="121" y="80.0"><title>NID2
linear:
scale=0.000013
scale offset=141003.958448
w=0.792789
bias=-1.3965</title></rect><text fill="#1E1E1E" font-family="monospace" font-size="12" style="pointer-events:none" text-anchor="middle" x="166.0" y="101.5">NID2</text><text fill="#1E1E1E" font-family="monospace" font-size="10" style="pointer-events:none" text-anchor="end" x="209" y="89.0">5</text><text fill="#1E1E1E" font-family="monospace" font-size="10" style="pointer-events:none" text-anchor="start" x="123" y="89.0">num</text><text fill="#1E1E1E" font-family="monospace" font-size="10" text-anchor="middle" x="406.0" y="45.0">0.88</text><text fill="#1E1E1E" font-family="monospace" font-size="10" text-anchor="middle" x="286.0" y="45.0">-0.8</text><text fill="#1E1E1E" font-family="monospace" font-size="10" text-anchor="middle" x="166.0" y="15.0">-0.78</text><text fill="#1E1E1E" font-family="monospace" font-size="10" text-anchor="middle" x="46.0" y="15.0">-0.76</text><text fill="#1E1E1E" font-family="monospace" font-size="10" text-anchor="middle" x="46.0" y="75.0">-0.33</text><text fill="#1E1E1E" font-family="monospace" font-size="10" text-anchor="middle" x="166.0" y="75.0">-0.21</text><rect fill="#ff1ec8" height="20" stroke-width="1" width="50" x="160.5" y="175.0" /><text fill="#1E1E1E" font-family="monospace" font-size="12" text-anchor="middle" x="185.5" y="188.0">-1</text><rect fill="#f9fafa" height="20" stroke-width="1" width="50" x="210.5" y="175.0" /><text fill="#1E1E1E" font-family="monospace" font-size="12" text-anchor="middle" x="235.5" y="188.0">0</text><rect fill="#00f082" height="20" stroke-width="1" width="50" x="260.5" y="175.0" /><text fill="#1E1E1E" font-family="monospace" font-size="12" text-anchor="middle" x="285.5" y="188.0">+1</text><text fill="#1E1E1E" font-family="monospace" font-size="12" text-anchor="middle" x="235.5" y="170.0">Pearson correlation</text></svg></svg>



As expected, `MAPT` (i.e. Tau) seems to be driving most of the signal here. Let's investigate further.

## Explore features

Let's look at how the different features play together.


```python
show_quantiles = 'NID2'
fixed = {}
fixed[show_quantiles] = [
    train[show_quantiles].quantile(q=0.25),
    train[show_quantiles].quantile(q=0.5),
    train[show_quantiles].quantile(q=0.75)
]

best.plot_response_1d(train, by = "MAPT", input_constraints=fixed)
```



![png](QLattice_tutorial_files/QLattice_tutorial_17_0.png)



This response plot shows you how higher `NID2` levels shift the `MAPT` curve to the left. I.e. the higher your `NID2` levels, the lower your `MAPT` levels have to be for a positive AD prediction.
