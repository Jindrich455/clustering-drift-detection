# TU Delft CSE3000 Research Project Repository
This is a repository for the CSE Bachelor end project at Delft University of Technology.
## Research question: "How well do clustering similarities-based concept drift detectors identify concept drift in case of synthetic/real-world data?"
This repository was used to answer this research question. We implemented two existing clustering similarities-based drift detection algorithms, MSSW[1] and UCDD[2]. The algorithms and common functions are organised in .py files, and the evaluation was made through separate jupyter notebooks.
```
[1] D. Shang, G. Zhang, and J. Lu, “Fast concept drift detection using unlabeled data,” in Developments of Artificial Intelligence Technologies in Computation and Robotics, Cologne, Germany, Oct. 2020, pp. 133–140. doi: 10.1142/9789811223334_0017.
[2] Y. Yuan, Z. Wang, and W. Wang, “Unsupervised concept drift detection based on multi-scale slide windows,” Ad Hoc Networks, vol. 111, p. 102325, Feb. 2021, doi: 10.1016/j.adhoc.2020.102325.
```

## Technology used
* OS: ```Windows 11 Home, 64-bit, x64-based processor```
* Python: ```Python 3.9.0```
* Pip: ```pip 21.1.1```
* Conda: ```conda 22.11.1```

## Locations of functions of interest - drift detection functions
* mssw > core > mssw.py > all_drifting_batches(...)
* ucdd_improved > core > ucdd.py > all_drifting_batches(...)

## To run the .py files
### High-level instructions
1. Navigate to the root folder ```clustering-drift-detection``` (root folder of this repository) in your OS's command line
2. Create a new virtual environment
3. Activate this environment
4. Run ```pip install -r requirements.txt```
5. Wait for the installation to complete
7. Now you should have all packages necessary to run the .py files from your command line

### Windows-specific instructions
1. Open your cmd
2. Navigate to ```clustering-drift-detection``` (root folder of this repository)
3. Create a new Python virtual environment with ```python3 -m venv venv```: this will create a new folder ```venv```
4. Activate this environment with ```venv\Scripts\activate```
5. Run ```pip install -r requirements.txt```
6. Wait for the installation to complete
7. Now you should have all packages necessary to run the .py files from your command line

## To run the jupyter notebooks
### High-level instructions
1. Activate the environment created for .py files
2. Open and run any jupyter notebook in the mssw or ucdd_improved packages with this environment

### Windows-specific instructions
1. Open an Anaconda Prompt
2. Navigate to ```clustering-drift-detection``` (root folder of this repository)
3. Activate the environment created for .py files through ```venv\Scripts\activate```
4. Run ```jupyter notebook``` to open any of the notebooks in the ```mssw``` or ```ucdd_improved``` packages
