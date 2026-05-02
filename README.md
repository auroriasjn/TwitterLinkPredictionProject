# Twitter Link Prediction Final Project
This is a final project for CPSC 483 - Deep Learning on Graph-Structured Data, which fuses elements of CPSC 471 - Trustworthy Deep Learning as an inspiration. The dataset provided here is the Higgs Twitter Dataset provided by SNAP.

The entire package is pip installable. Assuming you have all listed requirements, a simple `pip install .` in the directory you `pull`-ed suffices to get the package up and running, in addition to installing from `requirements.txt`. However, an installation script `./setup.sh` can also be run after downloading this repository.

The four notebooks, `follow_training.ipynb`, `mention_training.ipynb`, `reply_training.ipynb`, and `retweet_training.ipynb`, can all be run out of the box. This entire package is a Lightning wrapper to create GNNs which are trained for the link prediction task.
