import numpy as np
import math
import random
import statistics
from scipy import stats
class knn_classifier:
    x_train = None
    y_train = None
    k=None
    
    def __init__(self,k):
        self.k=k
  
    def fit(self,x,y):
        self.x_train=x
        self.y_train=y
        return self
    
    def predict(self,x):
        y_pred= []
        
        #for i, x1 in x: # would throw an error - so using new for loop below
        for i in range(len(x)): # for each point (feature vector) in the test set (to save as x1)
            x1 = x.iloc[i]
            dist_list = []
            #distance_index_eu=[]
            #for j, x2 in self.x_train: # would throw an error, so using new for loop below
            for j in range(len(self.x_train)): # for each point (feature vector) in the training set (to save as x2)
                x2 = self.x_train.iloc[j]
                #dist_man = self.manhattan_distance(x1,x2)
                dist_eu = self.get_eu_dist(x1,x2)
    
                dist_list.append((dist_eu,j))
    
            distance_list_sorted=sorted(dist_list)

            top_k_neighbours = distance_list_sorted[0:self.k]
            labels = []

            for touple in top_k_neighbours:
                index_of_training_sample = touple[1]
                label_of_training_inst = self.y_train[index_of_training_sample]
                labels.append(label_of_training_inst)
    
            label = None
            
            try:
                label = statistics.mode(labels)
            except:
                label = random.choice(labels)
             
                
            y_pred.append(label)
        
        return y_pred
    
    """
    def get_eu_dist_old(xi, yi): # issue here - can't do a distance between a feature vector and a class label vector
        
        val = 0
        total = 0
        for i, val in xi:
            
            difference = val - yi[i]
            total += math.pow(difference,2)
        
        return math.sqrt(total)
    """
    
    def get_eu_dist(self,x1, x2):
        """
        Calculates the Eculidean distance between two points (feature vectors)
        """
        total = 0
        for i in range(len(x1)):
            difference = x1[i] - x2[i]
            total += math.pow(difference,2)
        
        return math.sqrt(total)

    def manhattan_distance(x1,x2,n):
        sum=0
        #for i,val in x1:
        for i in range(len(x1)):
            sum+=abs(x1.iloc[i]-x2.iloc[i])
            #sum+=abs(xi[i]-xi[j]) + abs(yi[i]-yi[j])                    
        return sum
    
"""if __name__=='__main__':
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
#import knn_classifier()
#from Classifier_knn.ipynb import knn_classifier
    
    X = [[1,2],[13,14],[3,4],[15,16]] 
    y = [0,5,6,1]
    
    print(y_test)
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.2 , random_state = None)
    model = knn_classifier(k)
    model = model.fit(x_train, y_train)
    print(x_test)
    y_predict=model.predict(x_test)
    print('Accuracy', metrics.accuracy_score(y_test,y_predict))
    '''
'''#knn_class = knn_classifier()
    X = [[1,2],[13,14],[3,4],[15,16]]   

    #n = len(ar1)


    score=knn_classifier.get_eu_dist(X[0],X[1])
    score2 =knn_classifier.manhattan_distance(X[0],X[2],len(X[0]))                               
    print('Euclidean distance =',score)
    #print('man distance = ',score2)
    print(y)'''

#getting .get __getitem__ error when using classifier 
"""