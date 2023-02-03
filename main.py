# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 21:12:17 2023

@author: denho
"""
# Run commented code below if necessary
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import json
import numpy as np
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Input,Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.activations import sigmoid
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.close('all')
import random
import myplots as mp

import time
starttime = time.time()

''' Parameters'''
# Training hyper parameters
epochs = 2048
batch_size = 16
activation = 'tanh'

# CV parameters
fraction_test = .3

# Number of samples used. Set to 0 if all must be used.
num_of_samples = 0

# Initial NN parameters
units_ini = 128
num_of_layers_ini = 4
num_of_features_ini = 2

# Patience for optimization and training
patience_opt = 3                # Number of loops of no improvements before stopping the optimization loop.
patience_train = 100            # Number of loops of no improvements before stopping the training loop.

relative_step_size = 2      # Relative stepsize used for sweeping number of units and features.
convergence_criteria = 1e-2 # Minimum accepted relative improvement per step in optimization.


#%%
""" Load data """
print("Loading data")
X = np.loadtxt("../X_n512.csv")
Y = np.loadtxt("../Y_n512.csv")#.astype(bool)

# Use randomly chosen samples
if num_of_samples > 0:
    indeces = list(range(len(X)))
    random.shuffle(indeces)
    indeces = indeces[:num_of_samples]
    X = X[indeces]
    Y = Y[indeces]

num_of_examples = X.shape[0]
# Y = Y[:,0]∟

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=fraction_test)


#%%
''' Prepare for optimization '''
is_converged = False

tried_parameters = set()

# Dictionary to save preprocessed data for specific number of features. 
# Reuse data for n-number of features already preprocessed.
processed_data_dict = {}

# function to generate a randomized list of possible actions from parameters x.
def generate_possible_actions(x)->list:
    units,num_of_layers,num_of_features = x
    
    # Generate list of possible actions. Prioritize model expanding actions at the top and shuffle.
    l1 = [(units*relative_step_size,num_of_layers,num_of_features),
         (units,num_of_layers+1,num_of_features),
         (units,num_of_layers,num_of_features*relative_step_size),
         ]
    l2 = [(units/relative_step_size,num_of_layers,num_of_features),
          (units,num_of_layers-1,num_of_features),
          (units,num_of_layers,num_of_features/relative_step_size),
          ]
    random.shuffle(l1)
    random.shuffle(l2)
    l = l1 + l2
    
    # Filter out invalid actions and convert to int
    l = [tuple([int(value) for value in values]) for values in l if all([value >= 1 for value in values])]
    
    # Filter out already used parameters
    global tried_parameters
    l = [values for values in l if not (values in tried_parameters)]
    
    return l

# Function for easy plot of fit
# def plotsample(ax,X_example,Y_example,Y_example_predict=None):
#     Y_example = Y_example > .5
#     x_plot = np.array(range(len(X_example)))
    
#     ax.plot(x_plot,X_example,'or')
#     ax.plot(x_plot[Y_example],X_example[Y_example],'og')
    
#     if not Y_example_predict is None:
#         ax.plot(x_plot[Y_example_predict],X_example[Y_example_predict],'.k')
def plotsample(ax,X,Y,Y_predict=None):
    X_plot = X.reshape(-1,)
    Y_plot = Y.reshape(-1,)
    
    # mask0 = X_plot != ZERO_PAD_VALUE
    # X_plot = X_plot[mask0]
    # Y_plot = Y_plot[mask0]
    
    x_plot = np.array(range(len(X_plot)))
    ax.plot(x_plot,X_plot,'or')
    mask1 = Y_plot.reshape((-1,)) > .5
    ax.plot(x_plot[mask1],X_plot[mask1],'og')
    
    if not Y_predict is None:
        Y_predict_plot = Y_predict.reshape(-1,)
        # Y_predict_plot = Y_predict_plot[mask0]
        mask2 = Y_predict_plot.reshape((-1,)) > .5
        ax.plot(x_plot[mask2],X_plot[mask2],'.k')
        
# Function for assembling and fitting model
n_features = X.shape[1]
def assemble_and_fit(x):
    units,num_of_layers,num_of_features = x
    print(f"\n====== Starting training model with: units={units}, layers={num_of_layers}, features={num_of_features} ======")
    
    # Mark parameters as used
    global tried_parameters
    tried_parameters.add(x)
    
    ''' Preprocess data '''
    # Reuse already preprocessed data if available
    global processed_data_dict
    if num_of_features in processed_data_dict:
        X_train, X_test, Y_train, Y_test = processed_data_dict[num_of_features]
        
    else:
        # Data not available for given number of features. Start preprocessing raw data.
        X2 = np.reshape(X,(X.shape[0],-1,num_of_features))
        Y2 = np.reshape(Y,(X.shape[0],-1,num_of_features))
        X_train, X_test, Y_train, Y_test = train_test_split(X2, Y2, test_size=fraction_test)
        
        # Save for later reuse
        processed_data_dict[num_of_features] = (X_train, X_test, Y_train, Y_test)
        
    num_of_timesteps = X_train.shape[1]
    
    ''' Assemble model and fit '''
    # Assemble model
    model = Sequential()
    # model.add(layers.Input(shape=(num_of_timesteps,num_of_features),dtype=float))
    input_shape = (num_of_timesteps,num_of_features)
    # input_test = X2
    # print(input_test.shape)
    for i_layer in range(num_of_layers):
        kwargs = {}
        if i_layer == 0:
            kwargs['input_shape'] = input_shape
        kwargs['return_sequences'] = True
        # print(kwargs)
        layer = layers.Bidirectional(layers.LSTM(units,activation=activation,**kwargs))
        # input_test = layer(input_test)
        # print(input_test.shape)
        model.add(layer)
    layer = layers.Dense(num_of_features)
    # input_test = layer(input_test)
    # print(input_test.shape)
    model.add(layer)
    
    
    # Compile model
    loss = BinaryCrossentropy(from_logits=True)#from_logits=True)
    model.compile(loss=loss,
                  optimizer=Adam(1e-3))
    
    # Early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience_train)
    
    # Save best model
    modelfilename = f"sweepmodel_units={units}_layers={num_of_layers}_features={num_of_features}.h5"
    mc = ModelCheckpoint(modelfilename, monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    # Train model
    print("Training model")
    # model.build(input_shape)
    # model.summary()
    # print(X_train.shape)
    # print(Y_train.shape)
    # print(model.predict(X_train).shape)    
    history = model.fit(X_train, Y_train, 
                        validation_data=(X_test, Y_test), 
                        epochs=epochs, 
                        batch_size=batch_size,
                        callbacks=[es, mc])
    
    # Plot history
    figtitle = f"Loss history, units={units}, layers={num_of_layers}, features={num_of_features}"
    fig,ax = plt.subplots()
    ax.plot(history.history['loss'],label='Train')
    ax.plot(history.history['val_loss'],label='Test')
    ax.legend()
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    ax.set_title(figtitle)
    ax.set_yscale('log')
    plt.show()
    fig.savefig(figtitle + ".png")  
    plt.close(fig)
    
    # Load best results and plot results
    model = keras.models.load_model(modelfilename)
    test_loss = loss(Y_test,model.predict(X_test))
    
    figtitle = f"Fit result, units={units}, layers={num_of_layers}, features={num_of_features}, testloss={test_loss:.3g}"

    nrows = 5
    ncols = 7
    fig,ax = plt.subplots(nrows,ncols)
    ax = ax.reshape(-1)
    
    X2 = np.reshape(X,(X.shape[0],-1,num_of_features))
    Y2 = np.reshape(Y,(X.shape[0],-1,num_of_features))
    
    list_of_sample_ID = list(range(len(X2)))
    random.shuffle(list_of_sample_ID)
    i_plot_shuffle = 0
    # Y_predict = model.predict(X).numpy() >= .5
    Y_predict = sigmoid(model.predict(X2)).numpy() >= .5
    for i_ax in range(nrows*ncols):
        if i_ax >= len(X2):
            break
        # Select random sample and plot
        i_plot = list_of_sample_ID[i_plot_shuffle]
        plotsample(ax[i_ax],X2[i_plot],Y2[i_plot],Y_predict[i_plot])     
        
        i_plot_shuffle += 1
        
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    
    mp.figxlabel(figtitle,fig=fig)
    
    plt.show()
    fig.savefig(figtitle + ".png")  
    plt.close(fig)
    
    return model,test_loss

# Evaluate at starting point
x_old = (units_ini,num_of_layers_ini,num_of_features_ini)
best_model,loss_old = assemble_and_fit(x_old)
patience_opt_incrementor = 0
    
''' Optimization loop '''
print("====== Starting optimization ======")
while not is_converged:
    # Try next step
    possible_actions = generate_possible_actions(x_old)
    loss_best_in_step = np.inf
    x_best_in_step = None
    found_next_step = False
    for i_action,x in enumerate(possible_actions):        
        ''' Assemble and fit '''
        model,loss_train = assemble_and_fit(x)

        ''' Determine if improvement is good enough '''
        relative_change = (loss_train - loss_old)/loss_old
        print(f"\n\nRelative change = {relative_change:.5g}")
        if (-relative_change) > convergence_criteria:
            # Good enough. Keep current parameters as the new step.
            x_old = x
            loss_old = loss_train
            patience_opt_incrementor = 0
            found_next_step = True
            best_model = model
            print("\n\nGood enough step. Iterating...")
            break
        elif loss_train < loss_best_in_step:
            # Best next step so far
            x_best_in_step = x
            loss_best_in_step = loss_train
            print("\nStep not good enough. Trying another variant if possible.")
        
    
    ''' Global stopping criteria and iteration '''
    # Choose next step of none was found
    if not found_next_step:
        # No step good enough. Increment the patience number.
        patience_opt_incrementor += 1
        print(f'\n\nNo step good enough. Patience incrementor increased to {patience_opt_incrementor}.')
        x_old = x_best_in_step
        # Only overwrite old loss value if the next one is better
        if loss_best_in_step < loss_old:
            loss_old = loss_best_in_step
            best_model = model
            
    # Check if max patience reached
    if patience_opt_incrementor >= patience_opt:
        is_converged = True

print("\n\n --- Optimizer has converged ---")

# best_model.save("Winning model")

# Print execution time
endtime = time.time()
totaltime = endtime - starttime
print("\nTotal execution time: "+ str(totaltime/60) + " minutes")


