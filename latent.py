'''Code written for AML project paper "Latent features for Hyperparameter
   configuration selection"
   Author: Rachel de Jong (s1872508)
'''

import pandas
import numpy as np
from scipy.linalg import svd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

#Returns a sample of data for n_configurations different configuartions
def preprocess_data(data_file, hyper_parameter_columns):
    data = pandas.read_csv(data_file)
    data.drop(data.columns[hyper_parameter_columns], axis=1,inplace=True)
    data = data.reset_index().values
    data = np.delete(np.array(data), np.s_[:1], axis=1)
    return data

#Get a random sample of data 
def random_sample_data(data, nr_samples):
    index = np.random.randint(len(data), size = nr_samples)
    return data[index]

#From first AML assignment
def greedy_defaults(meta_data: pandas.DataFrame):
    row_it = 0               #Row (alg) iterator
    column_it = 0            #Column (dataset) iterator
    max_res = 0              #Maximal sum in current iteration     
    best_confs = []          #Set of selected algorithms
    best_conf = 0            #Index fo best configuration to add in current iteration
    index = list(meta_data.index.values)  #List with index of algorithms
    best_acc = []            #Accuracies of configurations chosen
    current_acc = []         #Accuracies chosen in current iteration
    cur_conf_acc = []        #Accuracies of current algorithm
    dif = 1                  #1 if difference is measured otherwise 0

    for i in range (meta_data.shape[1]):
        best_acc.append(-1)
       
    while dif > 0:
        row_it = 0
        max_res = np.sum(best_acc)
        for row in range (0, meta_data.shape[0]): #Configurations
            column_it = 0
            cur_conf_acc = meta_data.loc[index[row_it], :] #Current accuracy
            current_acc = []
            for column in range (0, meta_data.shape[1]): #Datasets
                #Check which one is better: store best in cur_conf_acc
                if cur_conf_acc[column_it] < best_acc[column_it]:
                    current_acc.append(best_acc[column_it])
                else:
                    current_acc.append(cur_conf_acc[column_it])
                column_it += 1
            #Check if adding current configuration results in a better result than found so far
            if np.sum(current_acc) > max_res:
                max_res = np.sum(current_acc)
                best_conf = row_it  #Store index of configuration
            row_it += 1    
        if max_res > np.sum(best_acc):
            #Add new configuration to set of configurations
            best_confs.append(index[best_conf]) 
            #Update best_acc list
            cur_conf_acc = meta_data.loc[index[best_conf], :]
            for i in range (0, meta_data.shape[1]):
                if cur_conf_acc[i] > best_acc[i]:
                    best_acc[i] = cur_conf_acc[i] 
        else:
            dif = 0
        column_it = 0
    return best_confs

#Return average runtime of best single solver given the performances of
#configurations on problems in dataset 'data'
def best_single_solver(data):
    averages = [np.mean(data[i]) for i in range(len(data))]
    return np.min(averages)

#Return average runtime of virtual best solver given the performances of
#configurations on problems in dataset 'data'
def virtual_best_solver(data):
    data = data.T
    avg = [np.min(data[i]) for i in range(len(data))]
    return np.mean(avg)

def average_solver(data):
    averages = [np.mean(data[i]) for i in range(len(data))]
    return np.mean(averages)

#Helpfunction for generating matrix of data required for training and testing
def data_train(data, train_index):
    train = []
    for i in range(len(data)):
        train.append([])
        for j in range (len(train_index)):
            train[i].append(data[i][j])
    return train 

#Return latent features for train set: this is equal to U Transpose
def x_train(data, train_index):
    train = data_train(data, train_index)
    U_train, s_train, VT_train = svd(train)
    U_train = U_train[:,:len(train_index)]
    return U_train.T

#Return lantent features for test set: PVS-1
def x_test(data, test_index):
    test = data_train(data, test_index)
    U, s, VT = svd(test)
    #s diagonal matrix
    s = np.diag(s)
    s_inverse = np.linalg.inv(s)
    V = VT.T
    U_test = np.dot(np.dot(test,V), s_inverse)
    return U_test.T
   
#Train a randomforest regressor to try predict runtimes of an configuration on
#a given instance. A model is trained for each configuration.
#The performance is calculated by choosing the configuration with the lowest 
#runtime on each problem and computing the average
def experiment(data):
    
    kf = KFold(n_splits = 5, shuffle = True)
    kf.get_n_splits(data[0])
    predictions = np.array([])
    dataT = np.array(data).T
    values = []

    #Split data for current configuration
    for train_index, test_index in kf.split(data[0]):
        predictions = []
        for solver in range(len(data)):
            y_train, y_test = data[solver][train_index], data[solver][test_index]
            X_train = x_train(data, train_index)
            X_test = x_test(data, test_index)
        
            regr = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
            regr.fit(X_train, y_train)
            predictions.append(regr.predict(X_test))
            
        predictions = np.array(predictions).T
        #Contains indices of best configurations
        index = [np.argmin(predictions[i]) for i in range(len(predictions))]
        #Contains performances of best configurations
        values.append(np.mean([data[index[i]][test_index[i]] for i in range(len(test_index))]))
        
    return np.mean(np.array(values))

#Perform experiment for random selection of portfolio
def random_selection():
    print('BSS, VBS, AVG, LAT')
    data = preprocess_data('data.csv', [0,1,2,3,4,5])
    #Iterate over configuration set sizes
    for i in range(0, 12):
        bss = []
        vbs = []
        lat = []
        avg = []
        nr_configurations = 40 + 20 * i
        #Repeat experiment 10 times for each configuration size
        for j in range(0, 10):     
            sample = random_sample_data(data, nr_configurations)
            bss.append(best_single_solver(sample))
            vbs.append(virtual_best_solver(sample))
            avg.append(average_solver(sample))
            lat.append(experiment(sample))
    
        print('configs: ' + str(nr_configurations))
        print('AVG: ' + str(np.mean(bss)) + ', ' + str(np.mean(vbs)) + ', ' + str(np.mean(avg)) + ', ' + str(np.mean(lat)))
        print('bss' + str(bss))
        print('vbs' + str(vbs))
        print('lat' + str(lat))
        print('avg' + str(np.mean(avg)))
    
    bss = []
    vbs = []
    avg = []
    lat = []
    nr_configurations = 288
    for j in range(0, 10):     
        sample = random_sample_data(data, nr_configurations)
        bss.append(best_single_solver(sample))
        vbs.append(virtual_best_solver(sample))
        avg.append(average_solver(sample))
        lat.append(experiment(sample))
    
    print('configs: ' + str(nr_configurations))
    print('AVG: ' + str(np.mean(bss)) + ', ' + str(np.mean(vbs)) + ', ' + str(np.mean(avg)) + ', ' + str(np.mean(lat)))
    print('bss' + str(bss))
    print('vbs' + str(vbs))
    print('lat' + str(lat))
    print('avg' + str(np.mean(avg)))

#Perform experiment for greedy selection of portfolio
def experiment_greedy():
    print('BSS, VBS, AVG, LAT')
    data = preprocess_data('data.csv', [0,1,2,3,4,5])
    df = pandas.DataFrame(data)
    index = greedy_defaults(df)
    nr_configurations = 40
    sample = data[index[:nr_configurations] + index[:10]]
    bss = best_single_solver(sample)
    vbs = virtual_best_solver(sample)
    avg = average_solver(sample)
    lat = experiment(sample)
    
    print('configs: ' + str(nr_configurations))
    print('AVG: ' + str(np.mean(bss)) + ', ' + str(np.mean(vbs)) + ', ' + str(np.mean(avg)) + ', ' + str(np.mean(lat)))
    print('bss' + str(bss))
    print('vbs' + str(vbs))
    print('lat' + str(lat))
    print('avg' + str(np.mean(avg)))
