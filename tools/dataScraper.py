from __future__ import print_function

import pickle
import argparse
import glob
import os
import sys

from SneakerNet.tools.google_images_download.google_images_download \
    import google_image_scrape


extra_keywords = ["red", "blue", "black", "white", "on foot"]


def scrape_from_keywords(keywords, saveDir, pictureNum):
    '''
    Given the keywords for a sneaker, download from google images
    the first [pictureNum] pics from a keyword
    '''

    for keyword in keywords:
        google_image_scrape(keyword, saveDir, pictureNum)

    return 0


def image_scrape(name, sneakerDir, pictureNum):
    ''' 
    A wrapper for the google_image_scrape function,
    but it adds extra keywords onto the main one to 
    augment the dataset
    '''

    google_image_scrape(name, sneakerDir, pictureNum)

    for extraKey in extra_keywords:
        
        google_image_scrape(" ".join([name, extraKey]), sneakerDir,
                            pictureNum, savePrefix=extraKey)


def scrape_from_file(sneakerFile, dataDir, pictureNum=100):
    ''' 
    Assume each line of the file is set up like this:
    brand,sneaker name\n
    '''

    try:
        f = open(sneakerFile, 'r')
    except IOError:
        print("Couldn't open file {}".format(sneakerFile))
        sys.exit(1)

    for line in f.readlines():

        fields = line.strip().split(',')
        brandName = fields[0]
        sneakerName = fields[1]
        
        # check if it's already been scraped
        brandDir = os.path.join(dataDir, brandName)
        if not os.path.isdir(brandDir):
            os.makedirs(brandDir)
        sneakerDir = os.path.join(brandDir, sneakerName)

        if not os.path.isdir(sneakerDir):
            os.makedirs(sneakerDir)
            image_scrape("{} {}".format(brandName, sneakerName),
                         sneakerDir, pictureNum)
        else:
            print("{}: {} -- already scraped!".format(brandName, sneakerName))

    f.close()


if __name__ == "__main__":

    picNum = 80
    dataDir = "/home/jonsmith/Projects/SneakerNet/data/images"
    sneakerFile = "/home/jonsmith/Projects/SneakerNet/data/sneakers.txt"
    scrape_from_file(sneakerFile, dataDir, picNum)
