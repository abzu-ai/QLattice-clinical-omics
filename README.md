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

