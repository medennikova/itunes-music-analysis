# the future division statement
from __future__ import division
import numpy as np
import pandas as pd

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

def class_prob(model, test_data):
    """Calculate probability for
    every class and return an array
    with highest probability for each
    track.
    """
    probability = model.predict_proba(test_data)

    # make a list of highest probabilities
    max_proba = np.asarray([x.max() for x in probability])
    return max_proba[:, None]

def model_summary(df, model, model_name, labels, cvs):
    print ("Summary of the {} classifier "
           "performance\n".format(model_name))
    print "Model: "
    print model
    print ("\nModel accuracy score: {:.3f}"
           .format(cvs))
    # drop 'match' column as it's not relevant here
    df = df.drop('match', 1)
    print ("\nNumber of tracks in the test set: {}."
           .format(len(df)))
    for lb in labels:
        print ("Number of tracks assigned to the \"{}\" class: {}, or {:.2%} of all tracks."
               .format(lb, len(df[df['label'] == labels.index(lb)]),
                      (len(df[df['label'] == labels.index(lb)])/len(df))))
    print "\nMean probability for each category"
    grouped = df.groupby(['label'])
    print grouped['prob'].mean()

def prob_with_threshold(df, threshold):
    """Returns a df with class probability
    higher than a threshold.
    """
    # drop 'match' column as it's not relevant here
    df = df.drop('match', 1)

    prob_thres = df[df['prob'] >= threshold]
    return prob_thres

def sum_category(df, index):
    """Returns number of tracks in a category
    """
    category_size = len(df[df['label'] == index])
    return category_size

def summary_of_prob_with_thres(df, threshold, labels):
    df_prob = prob_with_threshold(df, threshold)
    print ("Total number of tracks to classify: {}."
           .format(len(df)))
    print ("Number of tracks with {} probability "
          "of class membership: {}."
          .format(threshold, len(df_prob)))
    print "Among which..."
    for lb in labels:
        print ("...assigned the \"{}\" class: {} ({} without the threshold)."
               .format(lb, len(df_prob[df_prob['label'] == labels.index(lb)]),
                      len(df[df['label'] == labels.index(lb)])))
    print "\nMean probability for each class"
    grouped = df_prob.groupby(['label'])
    print grouped['prob'].mean()

def plot_prob_thres(df, labels):
    # threshold values
    xs = list(a / 10.0 for a in range(11))

    # all classes
    total = [len(prob_with_threshold(df, i)) for i in xs]
    line_total = plt.plot(xs, total, linewidth=2, label='total')

    for lb in labels:
        lb_sum = [sum_category(prob_with_threshold(df, i),
                                labels.index(lb)) for i in xs]
        line_lb = plt.plot(xs, lb_sum, linewidth=2, label=lb)

    plt.ylabel('Number of tracks')
    plt.xlabel('Probability')
    plt.legend()
    sns.despine(top=True, right=False)