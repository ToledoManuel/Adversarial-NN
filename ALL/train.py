
# Basic imports
import glob
import pickle
import logging as log
import itertools

# Scientific imports
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import tkinter
from sklearn.metrics import roc_curve, roc_auc_score
from scipy import stats, interpolate
from keras.layers import InputLayer

# Project imports
from adver.utilidad import *
from adver.profile import *
from adver.constants import *
from RUN.common import *
from RUN.comis import *



# Main function definition
@profile
def main (args):

    # Initialisation
    # --------------------------------------------------------------------------
    with Profile("Initialisation"):

        # Initialising
        # ----------------------------------------------------------------------
        args, cfg = initialise(args) # Call user inputs and config dictionary
        
        # Validate train/optimise flags
        if args.optimise_classifier:

            # Stand-alone classifier optimisation
            args.train_classifier  = True
            args.train_adversarial = False
            args.train = False
            cfg['classifier']['fit']['verbose'] = 2

        elif args.optimise_adversarial:

            # Adversarial network optimisation
            args.train_classifier  = False
            args.train_adversarial = True
            args.train = False
            cfg['combined']['fit']['verbose'] = 2

            pass

        cfg['classifier']['fit']['verbose'] = 2 
        cfg['combined']  ['fit']['verbose'] = 2  

        # Initialise Keras backend
        initialise_backend(args)
        
        # Imports for project
        import keras
        import keras.backend as K
        from keras.models import load_model
        from keras.callbacks import TensorBoard
        from keras.utils.vis_utils import plot_model

        # Neural network-specific initialisation of the configuration dict
        initialise_config(args, cfg)

        # Print the current environment setup
        print_env(args, cfg)
        pass


    # Loading data
    # --------------------------------------------------------------------------
    data, features, features_decorrelation = load_data(args.input + 'DataF.h5', train=True)
    num_features = len(features)

    # Regulsarisation parameter
    lambda_reg = cfg['combined']['model']['lambda_reg']  # Use same `lambda` as the adversary
    digits = int(np.ceil(max(-np.log10(lambda_reg), 0)))
    lambda_str = '{l:.{d:d}f}'.format(d=digits,l=lambda_reg).replace('.', 'p')

    # Get standard-formatted decorrelation inputs
    decorrelation = get_decorrelation_variables(data)
    aux_vars = ['logpTJ1', 'logpTJ2']
    data['logpTJ1'] = pd.Series(np.log(data['pTJ1'].values), index=data.index)
    data['logpTJ2'] = pd.Series(np.log(data['pTJ2'].values), index=data.index)
    
    # Specify common weights
    data['weight_clf'] = 1
    data['weight_adv'] = 1
    data['weight_test'] = 1
    
    # -- Classifier
    data['weight_clf'] = pd.Series(data['weight_adv'].values, index=data.index)

    # -- Adversary
    data['weight_adv'] = pd.Series(np.multiply(data['weight_adv'].values,  1. - data['signal'].values), index=data.index)

    from adver.models import classifier_model, adversary_model, combined_model
    
    sess = tf.compat.v1.InteractiveSession()
    # Classifier-only fit, full
    # --------------------------------------------------------------------------
    with Profile("Classifier-only fit, full"):

        # Define variable(s)
        name    = 'classifier'
        basedir = 'models/adversarial/classifier/full/'

        if args.train or args.train_classifier:
            log.info("Training full classifier")

            # Get classifier
            classifier = classifier_model(num_features, **cfg['classifier']['model'])

            # Save classifier model diagram to file
            plot_model(classifier, to_file=args.output + 'model_{}.png'.format(name), show_shapes=True)

            # Parallelise on GPUs
            parallelised = parallelise_model(classifier, args)

            # Compile model (necessary to save properly)
            parallelised.compile(**cfg['classifier']['compile'])

            # Create callbacks
            callbacks = []

            # Prepare arrays
            X = data[features].values
            Y = data['signal'].values
            W = data['weight_clf'].values

            # Fit classifier model
            ret = parallelised.fit(X, Y, sample_weight=W, callbacks=callbacks, **cfg['classifier']['fit'])

            # Save classifier model and training history to file, both in unique
            # output directory and in the directory for pre-trained classifiers.
            save([args.output, basedir], name, classifier, ret.history)


        else:

            # Load pre-trained classifier
            log.info("Loading full classifier from file")
            classifier, history = load(basedir, name)
            pass # end: train/load
   

    # Combined adversarial fit, full
    # --------------------------------------------------------------------------
    with Profile("Combined adversarial fit, full"):

        # Define variables
        name    = 'combined_lambda{}'.format(lambda_str)
        basedir = 'models/adversarial/combined/full/'

        # Load pre-trained classifier
        classifier, _ = load('models/adversarial/classifier/full/', 'classifier')

        # Set up adversary
        adversary = adversary_model(gmm_dimensions=len(DECORRELATION_VARIABLES),
                                    **cfg['adversary']['model'])
        
        # Save adversarial model diagram
        plot_model(adversary, to_file=args.output + 'model_adversary.png', show_shapes=True)

        # Create callback array
        callbacks = list()

        # Set up combined, adversarial model
        combined = combined_model(classifier, adversary, **cfg['combined']['model'])

        # Save combined model diagram
        plot_model(combined, to_file=args.output + 'model_{}.png'.format(name), show_shapes=True)

        if args.train or args.train_adversarial:
            log.info("Training full, combined model")

            # Parallelise on GPUs
            parallelised = parallelise_model(combined, args)

            # Compile model (necessary to save properly)
            parallelised.compile(**cfg['combined']['compile'])

            # Prepare arrays
            X = [data[features]    .values] + [data[aux_vars].values, decorrelation]
            Y = [data['signal']    .values] + [np.ones_like(data['signal'].values)]
            W = [data['weight_clf'].values] + [data['weight_adv'].values]

            # Compile model for pre-training
            classifier.trainable = False
            parallelised.compile(**cfg['combined']['compile'])

            # Pre-training adversary
            log.info("Pre-training")
            pretrain_fit_opts = dict(**cfg['combined']['fit'])
            pretrain_fit_opts['epochs'] = cfg['combined']['pretrain']
            ret_pretrain = parallelised.fit(X, Y, sample_weight=W, **pretrain_fit_opts)

            # Re-compile combined model for full training
            classifier.trainable = True
            parallelised.compile(**cfg['combined']['compile'])

            # Fit classifier model
            log.info("Actual training")
            ret = parallelised.fit(X, Y, sample_weight=W, callbacks=callbacks, **cfg['combined']['fit'])

            # Prepend initial losses
            for metric in parallelised.metrics_names:
                ret.history[metric]          = ret_pretrain.history[metric]          + ret.history[metric]
                pass

            # Save combined model and training history to file, both in unique
            # output directory and in the directory for pre-trained classifiers.
            adv = lambda s: s.replace('combined', 'adversary')
            save([args.output,     basedir],      name,  combined, ret.history)
            save([args.output, adv(basedir)], adv(name), adversary)


        else:

            # Load pre-trained combined _weights_ from file, in order to
            # simultaneously load the embedded classifier so as to not have to
            # extract it manually afterwards.
            log.info("Loading full, combined model from file")
            combined, history = load(basedir, name, model=combined)
            pass # end: train/load

        pass

    return data, args, features




# ---------------------- Prediction -------------------

# Parse command-line arguments
args = parse_args(adversarial=True)

# Call main function
data, args, features = main(args)


 # Define variable(s)
name    = 'classifier'
basedir = 'models/adversarial/classifier/full/'

ann_rocs = pd.DataFrame(columns=['fpr', 'tpr'])


data, features = load_data(args.input + 'DataF.h5', test=True)

# Prepare arrays
X = data[features].values
Y = data['signal'].values

log.info("ENTERING PREDICT: Loading full classifier from file")
classifier, history = load(basedir, name)

# Predict
scores_ann = classifier.predict(X, batch_size = 2048)


fpr_ann, tpr_ann, _ = roc_curve(Y, scores_ann)





ann_rocs = ann_rocs.append({'fpr': fpr_ann, 'tpr': tpr_ann}, ignore_index=True)


tpr_pts = np.linspace(0, 1, 6648)

# optimal CWoLa
fpr_interp = [None] * len(ann_rocs)
    
for i, row in ann_rocs.iterrows():
    fpr, tpr = row
    interp = interpolate.interp1d(tpr, fpr, fill_value=float('nan'), bounds_error=False, assume_sorted=True)

    fpr_pts = interp(tpr_pts)
    fpr_interp[i] = fpr_pts

fpr_interp = np.ma.masked_invalid(1./np.array(fpr_interp))


ann_mean = np.nanmean(fpr_interp, axis=0).data
ann_std = np.nanstd(fpr_interp, axis=0).data
    
sig_interp = np.ma.masked_invalid(np.sqrt(fpr_interp))
    
ocwola_smean = np.nanmean(sig_interp, axis=0).data
ocwola_sstd = np.nanstd(sig_interp, axis=0).data


plt.plot(tpr_pts, ann_mean, label = 'ANN', color= 'green')
plt.show()

