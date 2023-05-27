#!/usr/bin/env python
# coding: utf-8

# # Import packages

# In[1]:


import pandas as pd

#--- Load packages for datasets---
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer

#--- Load packages for logistic regression and random forest---
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#--- Load packages for train/test split---
from sklearn.model_selection import train_test_split

#import plotting libraries to visualise underfitting/overfitting
import matplotlib.pyplot as plt
#import gridsearch to find optimum value
from sklearn.model_selection import GridSearchCV


# In[2]:


#To ignore max_iter warning
import warnings
warnings.filterwarnings("ignore")


# # Logistic Regression

# ## Creating functions for convenience of calculation

# In[3]:


#Returns the training error on a value of C
def lr_train(C):
    clf = LogisticRegression(C = C, random_state=3)
    clf.fit(X_train,y_train)
    
    return 1-clf.score(X_train,y_train)

#Returns the testing error on a value of C
def lr_test(C):
    clf = LogisticRegression(C = C, random_state=3)
    clf.fit(X_train,y_train)
    
    return 1-clf.score(X_test,y_test)


# ### Dataset Iris

# In[4]:


# Load Iris dataset
X, y = load_iris(return_X_y=True)
# Split train/test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = True, random_state = 3)


# In[5]:


train_acc = []
test_acc = []

testrange = [10**x for x in range(-10,3)]

for i in testrange:
    train_acc.append(lr_train(i))
    test_acc.append(lr_test(i))  


# In[6]:


#Plotting error

plt.plot(testrange,train_acc,'r',label='train')
plt.plot(testrange,test_acc,'b',label='test')
plt.xlabel("C")
plt.ylabel("Error")
plt.legend()
plt.show()


# In[7]:


train_acc2 = []
test_acc2 = []
testrange2 = [0.00001*x for x in range(1,10000)]
for i in testrange2:
    train_acc2.append(lr_train(i))
    test_acc2.append(lr_test(i))  

#Plotting error


# In[8]:


plt.plot(testrange2,train_acc2,'r',label='train')
plt.plot(testrange2,test_acc2,'b',label='test')
plt.xlabel("C")
plt.ylabel("Error")
plt.legend()
plt.show()


# In[9]:


train_acc3 = []
test_acc3 = []
testrange3 = [0.02*x for x in range(1, 301)]

for i in testrange3:
    train_acc3.append(lr_train(i))
    test_acc3.append(lr_test(i))  


# In[10]:


#Plotting error

plt.plot(testrange3,train_acc3,'r',label='train')
plt.plot(testrange3,test_acc3,'b',label='test')
plt.xlabel("C")
plt.ylabel("Error")
plt.axvspan(0.1, 1.7, color='g', alpha=0.3,label = "underfitting")
plt.axvspan(3.5, 6, color='y', alpha=0.3,label='overfitting')
plt.legend()
plt.show()


# ### Attempt at using Gridsearch to find optimum value

# In[11]:


searchspace = [10**x for x in range(-10,10)]
grid = {"C":searchspace}


# In[12]:


lr = LogisticRegression(random_state=3)
model_lr = GridSearchCV(estimator = lr, param_grid=grid, verbose = 4)


# In[13]:


model_lr.fit(X_train,y_train)


# In[14]:


print(model_lr.best_estimator_) 

#Best estimate within one order of magnitude is C=1


# In[15]:


grid = {"C":[0.1*i for i in range(1,200)]}
lr = LogisticRegression(random_state=3)
model_lr = GridSearchCV(estimator = lr, param_grid=grid, verbose = 4)
model_lr.fit(X_train,y_train)


# In[16]:


print(model_lr.best_estimator_) #Within epsilon of 0.1 , the best fit model is C = 1.8 
                                #which is in the same range as determined above


# In[17]:


model_lr.score(X_train,y_train)


# In[18]:


# #Begin bisection method on the train error and the test error on (0.1,10)

# epsilon = 1e-10

# left = 0.1
# right = 10

# while True:
#     mid = (left+right)/2
#     print(mid,lr_train(mid))
    
#     if (abs(right-left) < epsilon): # or (lr_train(left)==lr_train(right)):
#         print("C = ", mid)
#         ans = mid
#         break
    
#     elif lr_train(left) > lr_train(right):
#         right = mid
    
#     else:
#         left = mid

# lr_tuned = LogisticRegression(C = ans,random_state=3)
# lr_tuned.fit(X_train,y_train)

# print('train ',lr_tuned.score(X_train, y_train))

# print('test ',lr_tuned.score(X_test, y_test))  

# #This method is not useful as it leads to overfitting hence it is not implemented


# In[19]:


# Initialize a logistic regression model
# Here, you only need to tune the inverse regularization parameter `C`
#lr = LogisticRegression(C=1e-10, random_state=3) 


# In[20]:


# Start training
#lr.fit(X_train, y_train)


# In[21]:


# Show training set error
#1 - lr.score(X_train, y_train)
# Show testing set error
#1 - lr.score(X_test, y_test)


# ## Wine Dataset

# In[22]:


# Load Wine dataset
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = True, random_state = 1)


# In[23]:


train_acc = []
test_acc = []

testrange = [10**x for x in range(-10,10)]

for i in testrange:
    train_acc.append(lr_train(i))
    test_acc.append(lr_test(i))  


# In[24]:


#Plotting error

plt.plot(testrange,train_acc,'r',label='train')
plt.plot(testrange,test_acc,'b',label='test')
plt.xlabel("C")
plt.ylabel("Error")
plt.legend()
plt.show()


# In[25]:


train_acc2 = []
test_acc2 = []
testrange2 = [0.01,0.1,1,10,100]
for i in testrange2:
    train_acc2.append(lr_train(i))
    test_acc2.append(lr_test(i))  


# In[26]:



#Plotting error

plt.plot(testrange2,train_acc2,'r',label='train')
plt.plot(testrange2,test_acc2,'b',label='test')
plt.xlabel("C")
plt.ylabel("Error")
plt.legend()
plt.show()


# In[27]:


train_acc4 = []
test_acc4 = []
testrange4 = [0.001,0.01,0.1,1,10,100,1000,10000,100000]

for i in testrange4:
    train_acc4.append(lr_train(i))
    test_acc4.append(lr_test(i))  


# In[28]:


#Plotting error

xaxis = list(map(str,testrange4))

plt.plot(xaxis,train_acc4,'r',label='train')
plt.plot(xaxis,test_acc4,'b',label='test')
plt.xlabel("C")
plt.ylabel("Error")
plt.axvspan(0, 2.9, color='g', alpha=0.3,label='underfitting')
plt.axvspan(6, 8, color='y', alpha=0.3,label='overfitting')
plt.legend()
plt.show()


# In[29]:


searchspace = [10**x for x in range(-10,10)]
grid = {"C":searchspace}


# In[30]:


searchspace = [10**x for x in range(-10,10)]
grid = {"C":searchspace}
lr = LogisticRegression(random_state=3)
model_lr = GridSearchCV(estimator = lr, param_grid=grid, verbose = 4)


# In[31]:


model_lr = GridSearchCV(estimator = lr, param_grid=grid, verbose = 4)


# In[32]:


model_lr.fit(X_train,y_train)


# In[33]:


print(model_lr.best_estimator_) 

#Best estimate within one order of magnitude is C=10


# In[34]:


grid = {"C":[0.1*x for x in range(1, 1001)]}
lr = LogisticRegression(random_state=3)
model_lr = GridSearchCV(estimator = lr, param_grid=grid, verbose = 4)
model_lr.fit(X_train,y_train)


# In[35]:


print(model_lr.best_estimator_)

#Within epsilon = 0.1, the best value is C = 32, which is in the same region as determined above


# In[36]:


model_lr.score(X_train,y_train)


# In[37]:


model_lr.score(X_test,y_test)


# In[38]:


# lr = LogisticRegression(C=1e-10, random_state=3) 
# lr.fit(X_train, y_train)


# In[39]:


#choose C to display underfitting and overfitting
#Plot and show the difference
#For all values tested comment them out


# In[40]:


# 1 - lr.score(X_train, y_train)


# In[41]:


# 1 - lr.score(X_test, y_test)


# # Random Forest Classification

# In[42]:


#Returns the training error on a value of C
def rf_train_depth(C):
    rf = RandomForestClassifier(max_depth=C, max_samples=10, n_estimators=3, random_state=3)
    rf.fit(X_train,y_train)
    
    return 1-rf.score(X_train,y_train)

#Returns the testing error on a value of C
def rf_test_depth(C):
    rf = RandomForestClassifier(max_depth=C, max_samples=10, n_estimators=3, random_state=3)
    rf.fit(X_train,y_train)
    
    return 1-rf.score(X_test,y_test)

#Returns the training error on a value of C
def rf_train_samples(C):
    rf = RandomForestClassifier(max_depth=10, max_samples=C, n_estimators=3, random_state=3)
    rf.fit(X_train,y_train)
    
    return 1-rf.score(X_train,y_train)

#Returns the testing error on a value of C
def rf_test_samples(C):
    rf = RandomForestClassifier(max_depth=10, max_samples=C, n_estimators=3, random_state=3)
    rf.fit(X_train,y_train)
    
    return 1-rf.score(X_test,y_test)


# ## Iris dataset

# In[43]:


# Load Iris dataset
X, y = load_iris(return_X_y=True)
# Split train/test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = True, random_state = 3)


# ### Depth optimization on Iris

# In[44]:


train_acc = []
test_acc = []

testrange = range(1,200)

for i in testrange:
    train_acc.append(rf_train_depth(i))
    test_acc.append(rf_test_depth(i))  


# In[45]:


#Plotting error

plt.plot(testrange,train_acc,'r',label='train')
plt.plot(testrange,test_acc,'b',label='test')
plt.xlabel("max_depth")
plt.ylabel("Error")
plt.legend()
plt.show()


# In[46]:


train_acc2 = []
test_acc2 = []

testrange2 = range(1,10)

for i in testrange2:
    train_acc2.append(rf_train_depth(i))
    test_acc2.append(rf_test_depth(i))  


# In[47]:


#Plotting error

plt.plot(testrange2,train_acc2,'r',label='train')
plt.plot(testrange2,test_acc2,'b',label='test')
plt.xlabel("max_depth")
plt.ylabel("Error")
plt.axvspan(0.8, 1.2, color='g', alpha=0.3,label='underfitting')
# plt.axvspan(3.5, 8, color='y', alpha=0.3,label='overfitting')
plt.legend()
plt.show()


# In[48]:


classifier = RandomForestClassifier(random_state=3)

param_grid = { 
    'n_estimators': [3],
    'max_samples': [10],
    'max_depth' : [x for x in range(1,1000)]
}
classifier = GridSearchCV(estimator=classifier, param_grid=param_grid)

classifier.fit(X_train, y_train)

classifier.best_params_


# ### Sample Optimization on Iris

# In[49]:


train_acc = []
test_acc = []

testrange = range(1,int(0.7*X.shape[0])+1)

for i in testrange:
    train_acc.append(rf_train_samples(i))
    test_acc.append(rf_test_samples(i))  


# In[50]:


#Plotting error

plt.plot(testrange,train_acc,'r',label='train')
plt.plot(testrange,test_acc,'b',label='test')
plt.xlabel("max_samples")
plt.ylabel("Error")
plt.legend()
plt.show()


# In[51]:


train_acc2 = []
test_acc2 = []

testrange2 = range(1,int(0.7*X.shape[0])+1,3)

for i in testrange2:
    train_acc2.append(rf_train_samples(i))
    test_acc2.append(rf_test_samples(i))  


# In[52]:



#Plotting error

plt.plot(testrange2,train_acc2,'r',label='train')
plt.plot(testrange2,test_acc2,'b',label='test')
plt.xlabel("max_samples")
plt.ylabel("Error")
plt.axvspan(0, 20, color='g', alpha=0.3,label='underfitting')
plt.axvspan(65, 105, color='y', alpha=0.3,label='overfitting')
plt.legend()
plt.show()


# In[53]:


classifier = RandomForestClassifier(random_state=3)

param_grid = { 
    'n_estimators': [3],
    'max_samples': [x for x in range(1,int(0.7*X.shape[0])+1)],
    'max_depth' : [10]
}
classifier = GridSearchCV(estimator=classifier, param_grid=param_grid)

classifier.fit(X_train, y_train)

classifier.best_params_


# In[54]:


# Initialize a random forest model
# Here, you need to take turns to tune max_depth/max_samples for showing cases of underfitting/overfitting
# Note that when you tune max_depth, please leave max_samples unchanged!
# Similarly, when you tune max_samples, leave max_depth unchanged!

# rf = RandomForestClassifier(max_depth=10, max_samples=10, n_estimators=3, random_state=3)
# rf.fit(X_train, y_train)


# In[55]:


# 1-rf.score(X_train, y_train)


# In[56]:


# 1-rf.score(X_test, y_test)


# ## Breast Cancer Dataset

# In[57]:


# Load Breast Cancer dataset for random forest
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = True, random_state = 3)


# In[58]:


# # Initialize a random forest model
# rf = RandomForestClassifier(max_depth=10, max_samples=10, n_estimators=3, random_state=3) # change max_depth
# # rf.fit(X_train, y_train)


# ### Depth optimization Breast Cancer

# In[59]:


train_acc = []
test_acc = []

testrange = range(1,5000,5)

for i in testrange:
    train_acc.append(rf_train_depth(i))
    test_acc.append(rf_test_depth(i))  


# In[60]:


#Plotting error

plt.plot(testrange,train_acc,'r', label='train')
plt.plot(testrange,test_acc,'b', label='test')
plt.xlabel("max_depth")
plt.ylabel("Error")
plt.legend()
plt.show()


# In[61]:


train_acc2 = []
test_acc2 = []

testrange2 = range(1,10)

for i in testrange2:
    train_acc2.append(rf_train_depth(i))
    test_acc2.append(rf_test_depth(i))  


# In[62]:


#Plotting error

plt.plot(testrange2,train_acc2,'r',label='train')
plt.plot(testrange2,test_acc2,'b',label='test')
plt.xlabel("max_depth")
plt.ylabel("Error")
plt.legend()
plt.show()


# In[63]:


classifier = RandomForestClassifier(random_state=3)

param_grid = { 
    'n_estimators': [3],
    'max_samples': [10],
    'max_depth' : [x for x in range(1,1000)]
}
classifier = GridSearchCV(estimator=classifier, param_grid=param_grid)

classifier.fit(X_train, y_train)

classifier.best_params_


# ### Sample optimization Breast Cancer

# In[64]:


train_acc = []
test_acc = []

testrange = range(1,int(0.7*X.shape[0])+1)

for i in testrange:
    train_acc.append(rf_train_samples(i))
    test_acc.append(rf_test_samples(i))  


# In[65]:


#Plotting error

plt.plot(testrange,train_acc,'r',label="train")
plt.plot(testrange,test_acc,'b',label="test")
plt.xlabel("max_samples")
plt.ylabel("Error")
plt.legend()
plt.show()


# In[66]:


train_acc2 = []
test_acc2 = []

testrange2 = range(150,int(0.7*X.shape[0])+1)

for i in testrange2:
    train_acc2.append(rf_train_samples(i))
    test_acc2.append(rf_test_samples(i))  


# In[67]:


#Plotting error

plt.plot(testrange2,train_acc2,'r',label="train")
plt.plot(testrange2,test_acc2,'b',label="test")
plt.xlabel("max_samples")
plt.ylabel("Error")
plt.legend()
plt.show()


# In[68]:


train_acc3 = []
test_acc3 = []

testrange3 = range(1,int(0.7*X.shape[0])+1,10)

for i in testrange3:
    train_acc3.append(rf_train_samples(i))
    test_acc3.append(rf_test_samples(i))  


# In[69]:


plt.plot(testrange3,train_acc3,'r',label="train")
plt.plot(testrange3,test_acc3,'b',label="test")
plt.xlabel("max_samples")
plt.ylabel("Error")
plt.axvspan(0, 120, color='g', alpha=0.3,label='underfitting')
plt.axvspan(360, 400, color='y', alpha=0.3,label='overfitting')
plt.legend()
plt.show()


# In[70]:


classifier = RandomForestClassifier(random_state=3)

param_grid = { 
    'n_estimators': [3],
    'max_samples': [x for x in range(1,int(0.7*X.shape[0])+1)],
    'max_depth' : [10]
}
classifier = GridSearchCV(estimator=classifier, param_grid=param_grid)

classifier.fit(X_train, y_train)

classifier.best_params_

