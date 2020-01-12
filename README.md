# AML-Project-2020
File "latent.py" Contains code used for AML project paper "Latent Features for Hyperparameter Selection"
Experiments are automatically run when using the following functions:
-greedy_defaults()
-random_selection()

For the paper tests were run on the dataset svm-ongrid.arff which can be found here:
https://github.com/janvanrijn/openml-multitask/blob/master/data/svm-ongrid.arff
(Author: janvanrijn)

Results folder contains the following results:
-avg_rand.txt: contains average results found using random sampling of portfolio of configurations
  Avg, nr configurations, BSS, VBS, AVG, LAT
-results_greedy.txt: contains restults using greedy default sampling
