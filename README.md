# FirnLearn

# Author

**Ayobami Ogunmolasuyi**

ayobami.o.ogunmolasuyi.th@dartmouth.edu

Thayer School of Engineering at Dartmouth College

# Overview
FirnLearn is an Antarctic steady-state firn densification model based on data science.Firn density models are simulated using a deep artificial neural network (i.e. deep learning). Other models have also been built using Lasso (a result of the best performing elastic net regularized multiple linear regression), random forest (i.e. ensemble decision trees regression), and gradient boosting (i.e. random forest with boosting). FirnLearn has been trained on a dataset of firn density observations (montgomery et al. 2018) and RACMO (Van Wessem et al., 2014; Noel et al., 2018).

FirnLearn is built upon widely used Python libraries (Keras and Scikit-learn).

# FirnLearn Models
Three main models developed for FirnLearn are described below/ Profile predictions with these models can be conducted using the Profile_predictions.py script. The models have been uploaded to the models folder and can be used to predict density profiles as described in the Profile_predictions.py script. 

**Artificial Neural Network**: Artificial Neural Networks (ANNs) are nonlinear statistical models that recognize relationships and patterns between the input and output variables of structured data in a manner that models the biological neurons of the human brain (Hatie and others, 2009; O’Shea and Nash, 2015). The FirnLearn ANN model is trained with the ann_training.py script. 

**Random Forest**: The random forest is an ensemble method that builds multiple decision trees during training and averages their result to get a more accurate and stable output (Breiman, 2001). The Random Forest model builds these different trees independently and in parallel. 

**Gradient Boosting**: Like the Random Forest model, the Gradient Boosting model is an ensemble method that combines the results of different trees. However, the trees in Gradient Boosting are built sequentially, with each new tree being trained to correct the errors made by the preceding tree.

The rf_gb.py script contains cross validation and training of both the random forest and gradient boosting models.

# Data

The data used to train and run the FirnLearn models are in the Data folder of this repository. Scripts to import, process and use the SUMup dataset can be found at https://github.com/MeganTM/SUMMEDup

