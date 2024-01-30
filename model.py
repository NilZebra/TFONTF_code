## Author: Nils Brandenstein
## Date: 28.04.2023 

import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import scipy
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, matthews_corrcoef, ConfusionMatrixDisplay, classification_report, brier_score_loss
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
seed = 42

## Get Vanilla model with preloaded weights

# Z sampling layer
# Note that in this case, the z sampling layer is deterministic (seeded) to enable reproducibility
def sampling_model(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=K.shape(z_mean), mean=0., stddev=1., seed=seed)
    return z_mean + K.exp(z_log_sigma/2) * epsilon

# Encoder only 
def vanilla_encoder(x_train, t):
    # Specify dimensions for input/output and latent space layers
    original_dim = x_train.shape[1] # number of features/columns
    latent_dim = 2 # latent space dimension
    init = tf.keras.initializers.GlorotNormal(seed=seed) # weight initialiser seed

    # ********** Create Encoder **********

    #--- Input Layer 
    visible = Input(shape=(original_dim,), name='Encoder-Input-Layer')

    #--- Hidden Layer
    h_enc1 = Dense(units=512, activation='relu', kernel_initializer = init, name='Encoder-Hidden-Layer-1')(visible)
    h_enc2 = Dense(units=256, activation='relu', kernel_initializer = init, name='Encoder-Hidden-Layer-2')(h_enc1)
    h_enc3 = Dense(units=128, activation='relu', kernel_initializer = init, name='Encoder-Hidden-Layer-3')(h_enc2)

    #--- Custom Latent Space Layer
    z_mean = Dense(units=latent_dim, kernel_initializer = init, name='Z-Mean')(h_enc3) # Mean component
    z_log_sigma = Dense(units=latent_dim, kernel_initializer = init, name='Z-Log-Sigma')(h_enc3) # Standard deviation component
    z = Lambda(sampling_model, name='Z-Sampling-Layer')([z_mean, z_log_sigma]) # Z sampling layer

    #--- Create Encoder model
    encoder_t = Model(visible, [z], name='Encoder-Model') # Define encoder model
    encoder_t.set_weights(t) # set weights from loaded encoder
    encoder_t.trainable = False # freeze encoder weights
    return encoder_t

# Full model
def vanilla_fullmodel(x_train, weights):
    # Specify dimensions for input/output and latent space layers
    original_dim = x_train.shape[1] # number of features/columns
    latent_dim = 2 # latent space dimension
    init = tf.keras.initializers.GlorotNormal(seed=seed) # weight initialiser seed

    # ********** Create Encoder **********

    #--- Input Layer 
    visible = Input(shape=(original_dim,), name='Encoder-Input-Layer')

    #--- Hidden Layer
    h_enc1 = Dense(units=512, activation='relu', kernel_initializer = init, name='Encoder-Hidden-Layer-1')(visible)
    h_enc2 = Dense(units=256, activation='relu', kernel_initializer = init, name='Encoder-Hidden-Layer-2')(h_enc1)
    h_enc3 = Dense(units=128, activation='relu', kernel_initializer = init, name='Encoder-Hidden-Layer-3')(h_enc2)

    #--- Custom Latent Space Layer
    z_mean = Dense(units=latent_dim, kernel_initializer = init, name='Z-Mean')(h_enc3) # Mean component
    z_log_sigma = Dense(units=latent_dim, kernel_initializer = init, name='Z-Log-Sigma')(h_enc3) # Standard deviation component
    z = Lambda(sampling_model, name='Z-Sampling-Layer')([z_mean, z_log_sigma]) # Z sampling layer

    #--- Create Encoder model
    encoder_t = Model(visible, [z], name='Encoder-Model') # Define encoder model

    # ********** Create Decoder **********

    #--- Input Layer (from the latent space)
    latent_inputs = Input(shape=(latent_dim,), name='Input-Z-Sampling') # Input layer 

    #--- Hidden Layer
    h_dec = Dense(units=128, activation='relu', kernel_initializer = init, name='Decoder-Hidden-Layer-1')(latent_inputs)
    h_dec2 = Dense(units=256, activation='relu', kernel_initializer = init, name='Decoder-Hidden-Layer-2')(h_dec)
    h_dec3 = Dense(units=512, activation='relu', kernel_initializer = init, name='Decoder-Hidden-Layer-3')(h_dec2)

    #--- Output Layer
    outputs = Dense(original_dim, activation='sigmoid', kernel_initializer = init, name='Decoder-Output-Layer')(h_dec3) # Output layer

    #--- Create Decoder model
    decoder = Model(latent_inputs, outputs, name='Decoder-Model') # Define decoder model

    # Define outputs from a VAE model by specifying how the encoder-decoder models are linked
    outpt = decoder(encoder_t(visible)) # note, outputs available from encoder model are z_mean, z_log_sigma and z
    # Instantiate a VAE model
    vae = Model(inputs=visible, outputs=outpt, name='VAE-Model')
    vae.set_weights(weights) # set weights from loaded model
    vae.trainable = False # freeze VAE weights

    return vae

### Script to import metric calculation
## Function to calculate ML metrics ##
def mlmetrics(y, yhat):
    y = np.round(y, 0).ravel() # ravel & round
    yhat = np.round(yhat, 0).ravel() # ravel & round
    prop = (y== yhat).sum()/float(y.size) # prop correctly predicted

    print('Brier Score: ', brier_score_loss(y, yhat))
    print('Balanced accuracy score: ', np.round(balanced_accuracy_score(y, yhat), 2))
    print('MCC: ', np.round(matthews_corrcoef(y, yhat), 2))
    print('Proportion correctly predicted', prop)
    print('Classification report: \n', classification_report(y, yhat))

    cm = confusion_matrix(y, yhat)

    ax = plt.subplot()
    sns.heatmap(cm, annot=True,cmap='Blues',fmt="d",ax=ax)
    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
    ax.set_ylim(2.0, 0)
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['No Follow','Follow'])
    ax.yaxis.set_ticklabels(['No Follow','Follow'])