import csv
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.metrics import plot_confusion_matrix


def read_file(file_name) :
    result = []
    with open(file_name, newline="") as f:
        file = csv.reader(f)
        i = 0
        for row in file :
            if len(row) == 1 :
                result.append(int(row[0]))  # Output result type int
            else :
                result.append([])
                for entry in row :
                    result[i].append(float(entry))
            i += 1
    return result

#initialize list as empty list
def get_avg_cloud(data_in, data_out) :
    vector_size = 256
    results = [list() for i in range (10)]
    for i in range (len(train_in)) :
        cloud = data_in[i]
        index = data_out[i]
        results[index].append(cloud)

    avg = []
    #Find averages
    for num in range(len(results)) :
        total = 0
        length = len(results[num]) # Number of instances of num
        total = [0 for k in range(vector_size)]
        for i in range (len(results[num])) : #Iterate over instances
            for j in range(len(results[num][i])) : #Iterate over 256 entries of vector
                total[j] += results[num][i][j]
        avg.append([x / length for x in total])

    return avg

def get_radius(avg, data_in, data_out) :
    avg = np.array(avg)
    radius = [0.0 for i in range (10)]

    for i in range(len(train_out)) : #Iterate over instances
        #Compute distance
        cloud = data_in[i]
        number = data_out[i]
        distance = np.linalg.norm(avg[number] - cloud)
        if radius[number] < distance :
            radius[number] = distance
    return radius

def get_distances(avg) :
    avg = np.array(avg)
    dist = []
    for i in range(len(avg)) :
        dist.append([])
        for j in range(len(avg)) :
            dist[i].append(np.linalg.norm(avg[i] - avg[j]))
    return dist

def get_min_distance_num(avg, cloud) : 
    #Compute distance between cloud and all averages
    #Return index of element with smallest distance
    min_dist = np.inf
    index = 0
    for i in range (len(avg)) :
        cur = np.linalg.norm(avg[i] - cloud)
        #print(cur)
        if cur < min_dist :
            min_dist = cur
            index = i
    return index

#Task 2

#Classify trainset
def classify_set(data_in, data_out, avg) :
    avg = np.array(avg)
    conf_matrix = [[0 for i in range(11)] for j in range(10)]
    correct = 0
    total = len(data_in)
    for i in range (len(data_in)) :
        pred = get_min_distance_num(avg, data_in[i])
        actual = data_out[i]
        conf_matrix[actual][pred] += 1
        if pred == actual :
            correct += 1
    return conf_matrix, correct / total

def plot_conf_matrix(matrix) :
    disp = plot_confusion_matrix(matrix)
    plot.show()
    
train_in = read_file("./data/train_in.csv")
train_out = read_file("./data/train_out.csv")
test_in = read_file("./data/test_in.csv")
test_out = read_file("./data/test_out.csv")    
    
avg = get_avg_cloud(train_in, train_out)
radius = get_radius(avg, train_in, train_out)
print(radius)
dist = get_distances(avg)
print(dist)

#Classify train set
conf, perc = classify_set(train_in, train_out, avg)
print(conf, perc)
#Classify test set
conf, perc = classify_set(test_in, test_out, avg)
print(conf, perc)


