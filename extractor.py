import numpy as np
import re

def extract(filename1,filename2):
    input_file = open(filename1)
    output_file = open(filename2)
    data = input_file.readlines()
    out_data = output_file.readlines()
    
    """
    Number of training data records
    """
    ntrain = int(data[0].split()[0])
    traindata = data[1:ntrain+1]
    """
    Number of test data records
    """
    ntest = int(data[int(data[0].split()[0])+1])
    testdata = data[ntrain+2:ntrain+ntest+2]
    features = []
    labels = []
    test_labels = []
    test_features = []
    for line in traindata:
        formatted_line = line.strip("\n")
        labels.append(formatted_line.split(" ")[1])
        features.append(re.sub(r"(\d+):", "", formatted_line).split(" ")[2:])
    
    features = np.array(features).astype(np.float)
    labels = np.array(labels).astype(np.int)
    
    for lines in testdata:
        formatted_lines = lines.strip("\n")
        test_features.append(re.sub(r"(\d+):", "", formatted_lines).split(" ")[1:])
      
    for lines in out_data:
        test_labels.append(lines.split()[1])
    
    test_features = np.array(test_features).astype(np.float)
    test_labels = np.array(test_labels).astype(np.int)
    
    return features,labels,test_features,test_labels
