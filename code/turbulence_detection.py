from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input

from  sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut

import xlrd
import pickle
import numpy as np
from utils import *
import time

# Created by emma Cotter 2021
# NOTE on a single GPU, cannot have VGG19 in memory and also calculate reconstruction on GPU
# recommend pre-processing all holograms to determine if they have turbulence, then performing reconstruction

# load VGG19 model existing SVM model
featuremodel = VGG19(weights='imagenet',include_top='False')
turbdetector = pickle.load(open(r'trainedmodels\turbsvm.p','rb'))

def flag_turbulence(proc_holo):
    # predict wehther a frame is affectd by turbulence using stored model (retrain function to generate)
    # load image in small frame
    # could save time by figuring out how to resize already loaded hologram, maybe

    # this takes about .074 seconds/hologram on emma's desktop

    path = holopath(proc_holo)
    img = image.load_img(path,target_size=(224,224))

    # preprocess for VGG19
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)

    # calculate features
    f = featuremodel.predict(x)

    # predict with SVM
    turbflag = int(turbdetector.predict(f))

    return turbflag

def retrain(hololist=r'trainedmodels/turbulencetraining.xlsx',validate=True, save = True):
    # retrain SVM model with files in hololist spreadsheet
    # first column has paths of holograms with  "no turbulence" and
    # second column has "turbulence"

    # read training data spreadsheet
    book = xlrd.open_workbook(hololist)
    sheet = book.sheet_by_index(0)
    noturb = sheet.col_values(1)[1:]
    Cnoturb = [0]*len(noturb)
    turb = sheet.col_values(2)[1:]
    Cturb = [1]*len(turb)
    files = noturb  + turb
    classes = Cnoturb+Cturb

    # extract features from each hologram  with VGG19
    features = np.zeros((len(files),1000))
    for i, file in enumerate(files):
        imgpath = holopath(file) # generate path of hologram image file
        img = image.load_img(imgpath,target_size = (224,224)) # load hologram image file
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis=0)
        x = preprocess_input(x)
        t = time.time()
        features[i,:] = featuremodel.predict(x)

    # perform validation
    if validate:
        X = features.copy()
        y = np.array(classes)
        loo = LeaveOneOut() # create leave one out validadtion
        n = 0
        results = np.zeros_like(y)
        yp = np.zeros_like(y)
        # iterate for every hologram in training data
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            pline = make_pipeline(StandardScaler(),SVC(gamma='auto'))
            pline.fit(X_train,y_train)
            t = time.time()
            y_testp = pline.predict(X_test)
            yp[n] = y_testp
            if (y_testp == y_test):
                results[n] = 1
            n+=1

        # calculate and display performance metrics
        precision = sum(yp[y==1])/sum(yp==1)
        recall = sum(yp[y==1])/sum(y==1)
        accuracy = sum(y == yp)/len(y)
        print('Precision: ' + str(precision))
        print('Recall: ' + str(recall))
        print('Accuracy: ' + str(accuracy))

    # fit SVM with all data
    pline = make_pipeline(StandardScaler(),SVC(gamma='auto'))
    pline.fit(X,y)

    # save SVM for later use
    if save:
        s = pickle.dump(pline,open(r'trainedmodels\turbsvm.p','wb'))
        turbdetector = pline
