# the future division statement
from __future__ import division
import numpy as np
import pandas as pd

%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

# set seaborn plot defaults
import seaborn as sns;
sns.set(palette="husl")
sns.set_context("notebook")
sns.set_style("whitegrid")

def nparray_to_df(data, store):
    """Transform dataset in numpy array format to a df.
    Return the df.
    """
    # make a list of column names
    li = (['index'] +
          store['train_std'].columns.tolist() +
          ['match', 'label', 'prob'])

    # transform set into a DF
    df = pd.DataFrame(data, columns=li)

    # change column type to integer
    df[['index', 'match', 'label']] = df[['index', 'match', 'label']].astype(int)

    # change index to the original index
    df = df.set_index(['index'])

    return df

def class_size(model_pred):
    """Count number of items
    in every class.
    """
    unique, counts = np.unique(model_pred[:, 0], return_counts=True)
    return dict(zip(unique, counts))

def class_mean(model_pred, lb_ind):
    """Return mean probability
    for the class
    """
    return np.mean(model_pred[model_pred[:, 0] == lb_ind][:, 1])

def model_summary(model_pred, model,
                  model_name, labels, cvs):
    """Print model summary
    """
    print ("Summary of the {} classifier "
           "performance \n\nModel: "
           .format(model_name))
    print model
    print ("\nModel accuracy score: {:.3f}"
           "\nNumber of tracks in the test set: {}. \n"
           .format(cvs, len(model_pred)))

    for lb in labels:
        # label index
        lb_ind = labels.index(lb)
        # size of the class
        lb_size = class_size(model_pred)[lb_ind]
        print ("Number of tracks assigned the \"{}\" class: "
               "{}, or {:.1%} of all tracks."
               .format(lb, lb_size,
                       lb_size / len(model_pred)))
        print ("Mean probability for the class: {:.3f}. \n"
               .format(class_mean(model_pred, lb_ind)))

def prob_with_threshold(model_pred, threshold):
    """Return an array with class probability
    higher than a threshold.
    """
    prob_thres = model_pred[model_pred[:, 1] >= threshold]
    return prob_thres

def plot_prob_thres(model_pred, labels):
    # threshold values
    xs = list(a / 10.0 for a in range(11))

    # all classes
    total = [len(prob_with_threshold(model_pred, i)) for i in xs]
    line_total = plt.plot(xs, total, linewidth=2, label='total')

    for lb in labels:
        # label index
        lb_ind = labels.index(lb)
        # size of the class
        lb_size = [class_size(prob_with_threshold(model_pred,
                                                  i))[lb_ind] for i in xs]
        line_lb = plt.plot(xs, lb_size, linewidth=2, label=lb)

    plt.ylabel('Number of tracks')
    plt.xlabel('Probability')
    plt.legend()
    plt.show()

def summary_of_prob_with_thres(model_pred, threshold, labels):
    """Print summary with a threshold applied to
    the model prediction.
    """
    model_thres = prob_with_threshold(model_pred, threshold)
    print ("Total number of tracks to classify: {}."
           "\nNumber of tracks with {} probability "
           "of class membership: {}, "
           "or {:.1%} of the initial assignment."
           .format(len(model_pred),
                   threshold,
                   len(model_thres),
                  (len(model_thres)/len(model_pred))))
    print "\nAmong which..."
    for lb in labels:
        # label index
        lb_ind = labels.index(lb)
        # size of the class
        lb_size = class_size(model_pred)[lb_ind]
        # size of the class with a threshold
        lb_thres_size = class_size(model_thres)[lb_ind]
        print ("...assigned the \"{}\" class: {} "
               "({:.1%} of the initial assignment.)"
               "\nMean probability for the class: {:.3f}. \n"
               .format(lb, lb_thres_size,
                       (lb_thres_size / lb_size),
                       class_mean(model_thres, lb_ind)))

