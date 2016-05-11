# Machine Learning for music playlists
### Building music playlists with Scikit-Learn tools
— March 2016 —

This is a series of posts devoted to analysis of iTunes music library using Scikit-Learn tools and the Echo Nest API.

In this analysis, I detect tracks in my iTunes music library that would suit my fitness practices. I'm interested in three classes of music ("cycling", "ballet", "yoga").

To solve that problem I use supervised machine learning classification algorithms: k-Neighbors, Random Forest, and SVM classifiers.

One of the main goals of this analysis is to explore the basics of Scikit-Learn tools. Scikit-Learn is a popular Python package designed to give access to well-known ML algorithms within Python code. 

The hero and the foundation of my analysis is the Echo Nest API, which provides broad and deep data on millions of artists and songs.

#### Contents\n",
"The series of posts includes the following notebooks:  \n",
"* [00_Overview](https://github.com/Tykovka/itunes-music-analysis/blob/master/00_Overview.ipynb) — Overview of the analysis, its goals and methods, installation notes.  \n",
"* [01_Data_preparation](https://github.com/Tykovka/itunes-music-analysis/blob/master/01_Data_preparation.ipynb) — Data gathering and cleaning.  \n",
"* [02_Data_visualisation](https://github.com/Tykovka/itunes-music-analysis/blob/master/02_Data_Visualisation.ipynb) — Visualisation and overview of data.  \n",
"* [03_Preprocessing](https://github.com/Tykovka/itunes-music-analysis/blob/master/03_Preprocessing.ipynb) — Data preprocessing to use it as input for Scikit-learn machine learning algorithms.  \n",
"* [04_Novelty_detection](https://github.com/Tykovka/itunes-music-analysis/blob/master/04_Novelty_detection.ipynb) — One-Class SVM algorithm to identify matching tracks in the non-labelled dataset.  \n",
"* [05_kNN_classifier](https://github.com/Tykovka/itunes-music-analysis/blob/master/05_kNN_classifier.ipynb) — k-Nearest Neighbors classifier.\n",
"* [06_Random_forest_classifier](https://github.com/Tykovka/itunes-music-analysis/blob/master/06_Random_forest_classifier.ipynb) — Random Forest classifier.\n",
"* [07_SVC](https://github.com/Tykovka/itunes-music-analysis/blob/master/07_SVC.ipynb) — Support Vector Machine classifier.\n",
"* 08_Summary(WIP) — Summary of the analysis where I compare the performance of all applied classification techniques.\n",
"  \n",

