import time
import numpy as np
import csv
from sklearn.metrics import precision_recall_fscore_support,classification_report

from classifier import Classifier
from eval import eval_list, load_label_output

def set_reproductible():
    import random as rn
    import os
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(17)
    rn.seed(12345)


if __name__ == "__main__":
    set_reproductible()
    datadir = "../data/"
    trainfile =  datadir + "restaurant_train.csv"
    devfile =  datadir + "data-2_test.csv"
    testfile =  None                        #File for prediction
    start_time = time.perf_counter()
    classifier = Classifier()
    print("\n")
    print("1. Training the classifier...\n")
    classifier.train(trainfile)
    print("\n2. Evaluation on the dev dataset...\n")
    slabels = classifier.predict(devfile)
    glabel_ids,glabels = load_label_output(devfile)
    n = min(len(slabels), len(glabels))
    gab = list(zip(glabel_ids,slabels))
    with open('restau.csv', 'w') as csv_file:
        csvwriter = csv.writer(csv_file, delimiter=';')
        for row in gab:
            csvwriter.writerow(row)
    if testfile is None:
        prf = classification_report(slabels,glabels,labels=[1,-1,0])
        print(prf)
        eval_list(glabels, slabels)
        if testfile is not None:
            # Evaluation on the test data
            print("\n3. Evaluation on the test dataset...\n")
            slabels = classifier.predict(testfile)
            glabels = load_label_output(testfile)
            eval_list(glabels, slabels)
    print("\nExec time: %.2f s." % (time.perf_counter()-start_time))




