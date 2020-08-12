import numpy as np
from knn import KNN
import math

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    tp = 0
    fp = 0
    fn = 0
    for i in range(0,len(real_labels)):
        if real_labels[i] == 1:
            if predicted_labels[i]==1:
                tp = tp + 1
            elif predicted_labels[i]==0:
                fn = fn + 1
        elif real_labels[i] == 0:
            if predicted_labels[i]==1:
                fp = fp + 1
    if (tp+fp) != 0:
        precision = tp/(tp+fp)
    else:
        precision = 0
    
    if (tp+fn) != 0:
        recall = tp/(tp+fn)
    else:
        recall = 0
    
    if (precision + recall) == 0:
        return 0
    
    f1 = (2*precision*recall)/(precision + recall)
    
    return f1
    
    
    raise NotImplementedError


class Distances:
    @staticmethod
    # TODO
    def canberra_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        d = 0
        for i in range(0,len(point1)):
            if abs(point1[i]) + abs(point2[i]) != 0:
                d = d + abs(point1[i] - point2[i])/(abs(point1[i]) + abs(point2[i]))
            else:
                d = d + abs(point1[i] - point2[i])
        return d
        raise NotImplementedError

    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        d = 0
        p = 3
        for i in range(0,len(point1)):
            d = d + pow(abs(point1[i] - point2[i]) , p)
        return pow(d , 1/p)
        
        raise NotImplementedError

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        d = 0
        p = 2
        for i in range(0,len(point1)):
            d = d + pow(abs(point1[i] - point2[i]) , p)
        return pow(d , 1/p)
        raise NotImplementedError

    @staticmethod
    # TODO
    def inner_product_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        d = 0
        for i in range(0,len(point1)):
            d = d + point1[i]*point2[i]
        return d
        raise NotImplementedError

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        d = 0
        c = 0
        den_point1 = 0
        den_point2 = 0
        for i in range(0,len(point1)):
            c = c + point1[i]*point2[i]
            
            den_point1 = den_point1 + pow(point1[i],2)
            den_point2 = den_point2 + pow(point2[i],2)
            
        den = pow(den_point1*den_point2, 0.5)
            
        d = 1 - c/den
        return d
        raise NotImplementedError

    @staticmethod
    # TODO
    def gaussian_kernel_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        d = 0
        g = 0
        for i in range(0,len(point1)):
            g = g + (point1[i]-point2[i])**2
        d = - math.exp(-0.5*g)
        return d
       
        raise NotImplementedError


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you should try different distance function you implemented in part 1.1, and find the best k.
        Use k range from 1 to 30 and increment by 2. Use f1-score to compare different models.

        :param distance_funcs: dictionary of distance functions you must use to calculate the distance.
            Make sure you loop over all distance functions for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val:  List[List[int]] Validation data set will be used on your KNN predict function to produce
            predicted labels and tune k and distance function.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function and self.best_model respectively.
        NOTE: self.best_scaler will be None

        NOTE: When there is a tie, choose model based on the following priorities:
        Then check distance function  [canberra > minkowski > euclidean > gaussian > inner_prod > cosine_dist]
        If they have same distance fuction, choose model which has a less k.
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_model = None
        self.best_scaler = None
        
        dist_dict = {'canberra' : 1, 'minkowski' : 2, 'euclidean' : 3, 
                     'gaussian' : 4, 'inner_prod' : 5, 'cosine_dist' : 6, None : 7}
        
        best_score = 0 
        
        for k in range(1,30,2):
            for name, fn in distance_funcs.items():
                model = KNN(k, fn)
                model.train(x_train, y_train)
                pred = model.predict(x_val)
                
                score = f1_score(y_val, pred)
                
                if score  > best_score:
                    self.best_k = k
                    self.best_distance_function = name
                    self.best_model = model
                    best_score = score
                elif score == best_score:
                    if dist_dict[self.best_distance_function] > dist_dict[name]:
                        self.best_k = k
                        self.best_distance_function = name
                        self.best_model = model
                        best_score = score
                    elif dist_dict[self.best_distance_function] == dist_dict[name]:
                        if self.best_k > k:
                            self.best_k = k
                            self.best_distance_function = name
                            self.best_model = model
                            best_score = score
                                           
        return
        raise NotImplementedError

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is similar to Part 1.3 except that before passing your training and validation data to KNN model to
        tune k and disrance function, you need to create the normalized data using these two scalers to transform your
        data, both training and validation. Again, we will use f1-score to compare different models.
        Here we have 3 hyperparameters i.e. k, distance_function and scaler.

        :param distance_funcs: dictionary of distance funtions you use to calculate the distance. Make sure you
            loop over all distance function for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param scaling_classes: dictionary of scalers you will use to normalized your data.
        Refer to test.py file to check the format.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data set you will use on your KNN predict function to produce predicted
            labels and tune your k, distance function and scaler.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function, self.best_scaler and self.best_model respectively

        NOTE: When there is a tie, choose model based on the following priorities:
        For normalization, [min_max_scale > normalize];
        Then check distance function  [canberra > minkowski > euclidean > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a less k.
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None
        
        dist_dict = {'canberra' : 1, 'minkowski' : 2, 'euclidean' : 3, 
                     'gaussian' : 4, 'inner_prod' : 5, 'cosine_dist' : 6, None : 7}
        
        scaling_dict = { 'min_max_scale': 1, 'normalize': 2, None: 3}
        
        best_score = 0 
        for scaler_name, scaler_class in scaling_classes.items():
            scaler = scaler_class()
            
            new_x_train = scaler(x_train)
            
            new_x_val = scaler(x_val)
            for name, fn in distance_funcs.items():
                for k in range(1,30,2):
                        model = KNN(k, fn)
                        model.train(new_x_train, y_train)
                        pred = model.predict(new_x_val)

                        score = f1_score(y_val, pred)
                        
                        if score  > best_score:
                            self.best_k = k
                            self.best_distance_function = name
                            self.best_scaler = scaler_name
                            self.best_model = model
                            best_score = score
                        elif score == best_score:
                            if scaling_dict[self.best_scaler] > scaling_dict[scaler_name]:
                                self.best_k = k
                                self.best_distance_function = name
                                self.best_scaler = scaler_name
                                self.best_model = model
                                best_score = score
                            elif scaling_dict[self.best_scaler] == scaling_dict[scaler_name]:
                                if dist_dict[self.best_distance_function] > dist_dict[name]:
                                    self.best_k = k
                                    self.best_distance_function = name
                                    self.best_scaler = scaler_name
                                    self.best_model = model
                                    best_score = score
                                elif dist_dict[self.best_distance_function] == dist_dict[name]:
                                    if self.best_k > k:
                                        self.best_k = k
                                        self.best_distance_function = name
                                        self.best_scaler = scaler_name
                                        self.best_model = model
                                        best_score = score

        return
        raise NotImplementedError


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        features_norm = [[0]*len(features[0]) for i in range(len(features))]
        for i in range(0,len(features)):
            s = 0            
            for j in range(0,len(features[0])):
                s += features[i][j]**2 
            n = pow(s,0.5)
            if n == 0:
                n = 1
            for j in range(0,len(features[0])):
                features_norm[i][j] = features[i][j] / n
                           
        
        return features_norm
        raise NotImplementedError


class MinMaxScaler:
    """
    Please follow this link to know more about min max scaling
    https://en.wikipedia.org/wiki/Feature_scaling
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
    will be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler1 = MinMaxScale()
        train_features_scaled = scaler1(train_features)
        # train_features_scaled should be equal to [[0, 1], [1, 0]]

        test_features_scaled = scaler1(test_features)
        # test_features_scaled should be equal to [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        self.mins = []
        self.maxs = []
        #pass

    def __call__(self, features):
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        cols = []
        features_scaled = [[0]*len(features[0]) for i in range(len(features))] 
        if len(self.mins) == 0 and len(self.maxs) == 0:
            for j in range(0,len(features[0])):
                cols = [el[j] for el in features]
                self.mins.append(min(cols))
                self.maxs.append(max(cols))
                 
        for i in range(0,len(features)):    
            for j in range(0,len(features[0])):
                if (self.maxs[j] - self.mins[j]) != 0:
                    features_scaled[i][j] = (features[i][j] - self.mins[j])/(self.maxs[j] - self.mins[j])
                else:
                    features_scaled[i][j] = (features[i][j] - self.mins[j])
                
        
        return features_scaled
        raise NotImplementedError
