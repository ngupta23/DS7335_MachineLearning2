# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 11:18:35 2019

@author: Chris
"""

import numpy as np

# Decision making with Matrices

# This is a pretty simple assingment.  You will do something you do everyday, but today it will be with matrix manipulations.

# The problem is: you and your work firends are trying to decide where to go for lunch. You have to pick a resturant thats best for everyone.  
# Then you should decided if you should split into two groups so eveyone is happier.

# Displicte the simplictiy of the process you will need to make decisions regarding how to process the data.

# This process was thoughly investigated in the operation research community.  This approah can prove helpful on any number of 
# decsion making problems that are currently not leveraging machine learning.

# You asked your 10 work friends to answer a survey. They gave you back the following dictionary object.


#to determine random values for weights
print(np.array([np.random.dirichlet(np.ones(4),size=1)]))

people = {'Jane': {'willingness to travel': 0.1596993,
                  'desire for new experience':0.67131344,
                  'cost':0.15006726,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.01892,
                  },
          'Bob': {'willingness to travel': 0.63124581,
                  'desire for new experience':0.20269888,
                  'cost':0.01354308,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.15251223,
                  },

          }

# Transform the user data into a matrix(M_people). Keep track of column and row ids.
                                       # convert each person's values to a list

peopleKeys, peopleValues = [], []
lastKey = 0
for k1, v1 in people.items():
    row = []
    
    for k2, v2 in v1.items():
        peopleKeys.append(k1+'_'+k2)
        if k1 == lastKey:
            row.append(v2)      
            lastKey = k1
            
        else:
            peopleValues.append(row)
            row.append(v2)   
            lastKey = k1
            

#here are some lists that show column keys and values
print(peopleKeys)
print(peopleValues)



peopleMatrix = np.array(peopleValues)

peopleMatrix.shape


# Next you collected data from an internet website. You got the following information.

#1 is bad, 5 is great
restaurants  = {'flacos':{'distance' : 2,
                        'novelty' : 3,
                        'cost': 4,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 5
                        },
              'Joes':{'distance' : 5,
                        'novelty' : 1,
                        'cost': 5,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 3
                      }

}


# Transform the restaurant data into a matrix(M_resturants) use the same column index.


restaurantsKeys, restaurantsValues = [], []

for k1, v1 in restaurants.items():
    for k2, v2 in v1.items():
        restaurantsKeys.append(k1+'_'+k2)
        restaurantsValues.append(v2)

#here are some lists that show column keys and values
print(restaurantsKeys)
print(restaurantsValues)

len(restaurantsValues)
#reshape to 2 rows and 4 columns

#converting lists to np.arrays is easy
restaurantsMatrix = np.reshape(restaurantsValues, (2,4))

restaurantsMatrix

restaurantsMatrix.shape

# Matrix multiplication
# Dot products are the matrix multiplication of a row vectors and column vectors (m,n) * (n,p)
#  shape example: ( 2 X 4 ) * (4 X 2) = 2 * 2
a = np.array([[1, 0], [0, 1]])
b = np.array([[1],[2]])

a.shape, b.shape

# when 2D arrays are involved, np.dot gives the matrix product.
np.dot(a,b)
np.matmul(a,b)

# documentation: https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html
# intuition: https://www.mathsisfun.com/algebra/matrix-multiplying.html
c = np.array([[1, 2, 3], [4, 5, 6]])
d = np.array([[7,8],[9,10],[11,12]])

c.shape, d.shape

np.dot(c,d)
# What is a matrix product?
# https://en.wikipedia.org/wiki/Matrix_multiplication
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.matmul.html#numpy.matmul
# matmul give the matrix product, too.
np.matmul(a,b)
#However, this won't work...
np.matmul(restaurantsMatrix, peopleMatrix)

#Why?
restaurantsMatrix.shape, peopleMatrix.shape
#We need to swap axis on peopleMatrix
#newPeopleMatrix = np.swapaxes(peopleMatrix, 1, 0)

#https://docs.scipy.org/doc/numpy/reference/generated/numpy.swapaxes.html
newPeopleMatrix = np.swapaxes(peopleMatrix, 0, 1)

peopleMatrix.T

peopleMatrix
#look at the matrices
peopleMatrix
newPeopleMatrix

restaurantsMatrix

restaurantsMatrix.shape, newPeopleMatrix.shape

# The most imporant idea in this project is the idea of a linear combination.

# Informally describe what a linear combination is and how it will relate to our resturant matrix.

    #This is for you to answer! However....https://en.wikipedia.org/wiki/Linear_combination
    # essentially you are multiplying each term by a constant and summing the results.

# Choose a person and compute(using a linear combination) the top restaurant for them.  
# What does each entry in the resulting vector represent?

print(peopleKeys)
print(peopleValues)

newPeopleMatrix

print(restaurantsKeys)
restaurantsMatrix


#Build intuition..
#Jane's score for Flacos
2*0.1596993 + 3*0.67131344 + 4*0.15006726 + 5*0.01892

#Bob's score for Flacos
2*0.63124581 + 3*0.20269888 + 4*0.01354308 + 5*0.15251223

#Jane's score for Joes
5*0.1596993 + 1*0.67131344 + 5*0.15006726 + 3*0.01892

#Bob's score for Joes
5*0.63124581 + 1*0.20269888 + 5*0.01354308 + 3*0.15251223


# Next compute a new matrix (M_usr_x_rest  i.e. an user by restaurant) from all people.  What does the a_ij matrix represent?
#Let's check our answers
results = np.matmul(restaurantsMatrix, newPeopleMatrix)
results                               


# Sum all columns in M_usr_x_rest to get optimal restaurant for all users.  What do the entry’s represent?
# I believe that this is what John and  is asking for, sum by columns
np.sum(results, axis=1)


# Now convert each row in the M_usr_x_rest into a ranking for each user and call it M_usr_x_rest_rank.   
# Do the same as above to generate the optimal resturant choice.
results

# Say that rank 1 is best

#reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
# Argsort returns the indices that would sort an array - https://stackoverflow.com/questions/17901218/numpy-argsort-what-is-it-doing
# By default, argsort is in ascending order, but below, we make it in descending order and then add 1 since ranks start at 1
sortedResults = results.argsort()[::-1] +1
sortedResults

sortedResults = results.argsort()[::-1]
sortedResults
                               
