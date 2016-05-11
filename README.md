# Machine Learning for music playlists
### Building music playlists with Scikit-Learn tools
— March 2016 —

This is a series of posts devoted to analysis of iTunes music library using Scikit-Learn tools and the Echo Nest API. I examine a variety of track attributes (e.g. tempo, time signature, energy) to build themed playlists.

To solve that problem I apply supervised machine learning classification algorithms: k-Neighbors, Random Forest, and SVM classifiers. I use Scikit-Learn, a popular Python package designed to give access to well-known ML algorithms within Python code. 

The hero and the foundation of my analysis is the Echo Nest API, which provides broad and deep data on millions of artists and songs.

#### Contents
The series of posts includes the following notebooks:
* [00_Overview](https://github.com/Tykovka/itunes-music-analysis/blob/master/00_Overview.ipynb) — Overview of the analysis, its goals and methods, installation notes.
* [01_Data_preparation](https://github.com/Tykovka/itunes-music-analysis/blob/master/01_Data_preparation.ipynb) — Data gathering and cleaning.
* [02_Data_visualisation](https://github.com/Tykovka/itunes-music-analysis/blob/master/02_Data_Visualisation.ipynb) — Visualisation and overview of data.
* [03_Preprocessing](https://github.com/Tykovka/itunes-music-analysis/blob/master/03_Preprocessing.ipynb) — Data preprocessing to use it as input for Scikit-learn machine learning algorithms.
* [04_Novelty_detection](https://github.com/Tykovka/itunes-music-analysis/blob/master/04_Novelty_detection.ipynb) — One-Class SVM algorithm to identify matching tracks in the non-labelled dataset.
* [05_kNN_classifier](https://github.com/Tykovka/itunes-music-analysis/blob/master/05_kNN_classifier.ipynb) — k-Nearest Neighbors classifier.
* [06_Random_forest_classifier](https://github.com/Tykovka/itunes-music-analysis/blob/master/06_Random_forest_classifier.ipynb) — Random Forest classifier.
* [07_SVC](https://github.com/Tykovka/itunes-music-analysis/blob/master/07_SVC.ipynb) — Support Vector Machine classifier.
* 08_Summary (WIP) — Summary of the analysis where I compare the performance of all applied classification techniques.

You can view materials using the nbviewer service.
Note, however, that you cannot modify or run the contents within nbviewer. To modify them, first download the repository, change to the notebooks directory, and run ipython notebook. For more information on the IPython notebook, see http://ipython.org/notebook.html
