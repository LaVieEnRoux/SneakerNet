from __future__ import print_function

import os
from random import shuffle

import cv2
import tensorflow as tf


def create_image_lists(dataDir, validationSplit, prediction_type="brands"):
    ''' Build the list of images and classes 
    
    Main issue here is that it creates a different validation
    set every time it runs!'''

    output = {}

    brandDirs = os.listdir(dataDir)
    
    for brand in brandDirs:

        if prediction_type == "brands":
            label = brand
        
        print("Processing brand {}...".format(brand))
        
        if prediction_type == "brands":
            trainingImages = []
            validationImages = []

        currentDir = os.path.join(dataDir, brand)
        allSneakers = os.listdir(currentDir)

        for sneakerName in allSneakers:

            if prediction_type == "sneakers":
                trainingImages = []
                validationImages = []
                label = "{} {}".format(brand, sneakerName)

            if prediction_type == "both":
                trainingImages = []
                validationImages = []
                label = "{}--{}".format(brand, sneakerName)

            print("Processing sneaker {}...".format(sneakerName))

            sneakerDir = os.path.join(currentDir, sneakerName)
            allSneakers = [os.path.join(sneakerDir, v) 
                           for v in os.listdir(sneakerDir)]

            goodSneakers = []

            for sneaker in allSneakers:

                # first check if image is valid
                test = cv2.imread(sneaker)
                if test is not None:
                    goodSneakers.append(sneaker)

            # shuffle to ensure an even split
            shuffle(goodSneakers)
            numImages = len(goodSneakers)
            splitIndex = int(numImages * (1 - validationSplit))
            trainSneakers = goodSneakers[:splitIndex]
            valSneakers = goodSneakers[splitIndex:]

            trainingImages.extend(trainSneakers)
            validationImages.extend(valSneakers)

            if prediction_type == "sneakers" or prediction_type == "both":
                output[label] = {
                    'training': trainingImages,
                    'validation': validationImages
                }

        print("{} has {} sneakers in Training".format(label,
                                                      len(trainingImages)))
        print("{} has {} sneakers in Val".format(label, len(validationImages)))

        if prediction_type == "brands":
            output[label] = {
                'training': trainingImages,
                'validation': validationImages
            }

    return output
