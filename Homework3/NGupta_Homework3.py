#%%

# Decision making with Matrices

# This is a pretty simple assingment.  You will do something you do everyday, but today it will be with matrix manipulations. 

# The problem is: you and your work firends are trying to decide where to go for lunch. You have to pick a resturant thats best for everyone.  Then you should decided if you should split into two groups so eveyone is happier.  

# Displicte the simplictiy of the process you will need to make decisions regarding how to process the data.
  
# This process was thoughly investigated in the operation research community.  This approah can prove helpful on any number of decsion making problems that are currently not leveraging machine learning.  

#%%

import numpy as np
import matplotlib.pyplot as plt
verbose = 0

#%%
def print_question(question_number):
    print("\n\n")
    print("*"*12)
    print("Problem " + str(question_number))
    print("*"*12)    


#%%
#######################
#### Problem Setup ####
#######################

print_question("Setup")

print("Performing problem setup ...")

# You asked your 10 work friends to answer a survey. They gave you back the following dictionary object.  

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
          'Mary': {'willingness to travel': 0.49337138 ,
                  'desire for new experience': 0.41879654,
                  'cost': 0.05525843,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.03257365,
                  },
          'Mike': {'willingness to travel': 0.08936756,
                  'desire for new experience': 0.14813813,
                  'cost': 0.43602425,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.32647006,
                  },
          'Alice': {'willingness to travel': 0.05846052,
                  'desire for new experience': 0.6550466,
                  'cost': 0.1020457,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.18444717,
                  },
          'Skip': {'willingness to travel': 0.08534087,
                  'desire for new experience': 0.20286902,
                  'cost': 0.49978215,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.21200796,
                  },
          'Kira': {'willingness to travel': 0.14621567,
                  'desire for new experience': 0.08325185,
                  'cost': 0.59864525,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.17188723,
                  },
          'Moe': {'willingness to travel': 0.05101531,
                  'desire for new experience': 0.03976796,
                  'cost': 0.06372092,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.84549581,
                  },
          'Sara': {'willingness to travel': 0.18780828,
                  'desire for new experience': 0.59094026,
                  'cost': 0.08490399,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.13634747,
                  },
          'Tom': {'willingness to travel': 0.77606127,
                  'desire for new experience': 0.06586204,
                  'cost': 0.14484121,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.01323548,
                  }                  
          }
      

# Transform the user data into a matrix(M_people). Keep track of column and row ids.   

def dict_to_array(dictionary):
    nrows = len(dictionary.keys())
    ncols = [len(value) for key, value in dictionary.items()][0]

    first_level_keys, second_level_keys, values = [], [], []
    for k1, v1 in dictionary.items():
        first_level_keys.append(k1)
        for k2, v2 in v1.items():
            if (k2 not in second_level_keys):
                second_level_keys.append(k2)
            values.append(v2)
    
    return(first_level_keys, second_level_keys, np.asarray(values).reshape(nrows, ncols))

people_names, people_rating_categories, M_people = dict_to_array(people)
if (verbose >= 1):
    print("Names of friends", people_names)
    print("\nPreferences")
    print(M_people)

# Transpose to allow for multiplication later
M_people = M_people.T
if (verbose >= 1):
    print("\nAfter transposing preference matrix")
    print(M_people)


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
                      },
              'Poke':{'distance' : 4,
                        'novelty' : 2,
                        'cost': 4,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 4
                      },                      
              'Sush-shi':{'distance' : 4,
                        'novelty' : 3,
                        'cost': 4,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 4
                      },
              'Chick Fillet':{'distance' : 3,
                        'novelty' : 2,
                        'cost': 5,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 5
                      },
              'Mackie Des':{'distance' : 2,
                        'novelty' : 3,
                        'cost': 4,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 3
                      },
              'Michaels':{'distance' : 2,
                        'novelty' : 1,
                        'cost': 1,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 5
                      },
              'Amaze':{'distance' : 3,
                        'novelty' : 5,
                        'cost': 2,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 4
                      },
              'Kappa':{'distance' : 5,
                        'novelty' : 1,
                        'cost': 2,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 3
                      },
              'Mu':{'distance' : 3,
                        'novelty' : 1,
                        'cost': 5,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 3
                      },
              'Chuys':{'distance' : 4,
                        'novelty' : 4,
                        'cost': 5,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 2
                      },        
}

#%%
# Transform the restaurant data into a matrix(M_resturants) use the same column index.

restaurant_names, resturant_rating_categories, M_restaurant = dict_to_array(restaurants)
if (verbose >= 1):
    print("Names of restaurants\n {}".format(restaurant_names))
    print("\nRestaurant Details")
    print(M_restaurant)

#%%
def print_question(question_number):
    print("\n\n")
    print("*"*12)
    print("Problem " + str(question_number))
    print("*"*12)    


#%%
###################
#### Problem 1 ####
###################

print_question(1)
    
# The most imporant idea in this project is the idea of a linear combination.  
# Informally describe what a linear combination is  and how it will relate to our resturant matrix.
    
print("\nA linear combination is a concept in linear algebra where you multiple 2 vectors together (element wise) to obtain a new vector and then sum up the elements of this new vector to get an aggregate number.\n\nThe way it applies to our problem is that we have a rating for each resturant in particular categories and each person has also given their preference for those same categories. Hence by multiplying the preferences and ratings appropriately and summing them up, we can get an aggregate score for that person for a particular resturant.\n\nThis is especially helpful since now we dont have to have every person rate every resturant individually. All they have to do is provide their preferences and based on how the resturant scores on those preferences, we can obtain an aggretae score easily.")

#%%
###################
#### Problem 2 ####
###################

# Choose a person and compute(using a linear combination) the top restaurant for them.  What does each entry in the resulting vector represent. 

print_question(2)

# Choosing the 1st person
first_person_score = np.dot(M_restaurant, M_people[:,0])
print("\n{} has scored the restaurants as follows".format(people_names[0]))
print(restaurant_names)
print(first_person_score)
print("\nEach entry in the result represents {}'s score for each restaurant based on the preferences".format(people_names[0]))

max_index = np.argmax(first_person_score)
print("\n{} >> 1st Preference: {} | Score: {}".format(people_names[0], restaurant_names[max_index], first_person_score[max_index]))

#%%
###################
#### Problem 3 ####
###################
# Next compute a new matrix (M_usr_x_rest  i.e. an user by restaurant) from all people.  What does the a_ij matrix represent?

print_question(3)

M_usr_x_rest = np.dot(M_restaurant, M_people)
print(M_usr_x_rest)

print("\nEach row in the matrix represents 1 restaurant and each column in the matrix represents 1 person.",
      "\nEach cell in the matrix represents the score given by the person (represented by the column) for that restaurant (represnted by the row)")

#%%
###################
#### Problem 4 ####
###################
# Sum all columns in M_usr_x_rest to get optimal restaurant for all users.  What do the entry’s represent?

print_question(4)

print(restaurant_names)

M_usr_x_rest_cum_rating = np.sum(M_usr_x_rest, axis=1)
print(M_usr_x_rest_cum_rating)
print("\nEach entry represents the cumulative weighted score for each restaurant across all users")

print("\nResturant pick for the group based on Score: {}".format(restaurant_names[np.argmax(M_usr_x_rest_cum_rating)]))

#%%
###################
#### Problem 5 ####
###################
# Now convert each row in the M_usr_x_rest into a ranking for each user and call it M_usr_x_rest_rank.   Do the same as above to generate the optimal resturant choice.

print_question(5)

print("\nScore Matrix")
print(M_usr_x_rest)
temp = M_usr_x_rest.argsort(axis=0)  # Returns the indices that would sort the array
M_usr_x_rest_rank = np.arange(len(M_usr_x_rest))[temp.argsort(axis=0)]+1 # Higher number is better rank (done to make consistent with Score concept from previous question)

print("\nRank Matrix (by person)")
print(M_usr_x_rest_rank)
print("\nSum of Ranks per Restaurant (Higher is better)")
print(restaurant_names)

M_usr_x_rest_rank_cum_rating = np.sum(M_usr_x_rest_rank, axis=1)
print(M_usr_x_rest_rank_cum_rating)

print("\nResturant pick for the group based on Rank: {}".format(restaurant_names[np.argmax(M_usr_x_rest_rank_cum_rating)]))


#%%
###################
#### Problem 6 ####
###################
# Why is there a difference between the two?  What problem arrives?  What does represent in the real world?

print_question(6)

print("\nAlthough there was no difference in the pick in the above case, there can be a difference in the restaurant we decide to visit depending on whether we use the absolute score or the ranking to select. This is due to the fact that the scores have a wider range and can be skewed in one direction, whereas the ranking has a fixed range and can not be skewed.",
      "\n\nThe problem is that if we use the base (raw) score and one score is very high, it can give that resturant an unfair advantage at the expense of the rating of all other people in the voting pool. For example, one (or a few) user(s) can put a very high preference for a feature for which a given resturant has a high rating and that has potential to skew the results in that restaruant's favor (esentially game the scoring system by skewing the results).",
      "\n\nOn the other hand, using ranking will create an artificial gap between resturant that have a very similar score. For example, if the 1st resturant had cumulatove score of 88 and the 2nd one had 87, there is essentially no difference in their ratings, but if we converted it into rankings, the 1st one will be Rank 1 and the second one will be Rank 2 and we would lose all sense of the absolute difference which might be important to know in order to make a meaningful decision.",
      "\n\nIn real life, this represents a dilema of what metric to use to score the task at hand. The use of different metrics can have a very different outcome.")



#%%
###################
#### Problem 7 ####
###################
# How should you preprocess your data to remove this problem. 

print_question(7)

 
print("\nThe problem of skewness described above can be exacerbated if we have some people who are highly opinionated. These folks may have a rating of 1 (lowest) for some resturants while they may have a rating of 5 (highest) for others. On the other hand, less opinionated people may rate all resturants close to each other. \n\nIn some sense, the rating is arbirtaty since there is no reference on how to use the scale. Essentially what someone might rate a 3, someone else might rate a 2 or a 1 (all else being equal).\n\nIn order to remove this issue, we might want to assume that the underlying ratings all follow a certain distribution such as a gaussian distribution. If we make this assumption, then we can map each person's ratings to this gaussian scale and that would normalize the ratings of opinionated people vs. those that were not as opinionated.\n\nsklearn provides useful functions to perform these transformatrions and they can be found here: https://scikit-learn.org/stable/auto_examples/preprocessing/plot_map_data_to_normal.html")


#%%
###################
#### Problem 8 ####
###################
# Find user profiles that are problematic, explain why?

print_question(8)

print("\nIn order to check the problematic user profiles, we can look at the range of the ratings provided by the user for the resturants and check which profiles has a wide range compared to the mean score provided by them. This metric is called 'Coefficient of Variance'\n")

mean_per_user = M_usr_x_rest.mean(axis=0)
std_per_user = M_usr_x_rest.std(axis=0)
coeff_variance = std_per_user/mean_per_user

if(verbose >= 1):
    print(mean_per_user)
    print(std_per_user)

print(people_names)
print(coeff_variance)

plt.boxplot(coeff_variance, showmeans=True)
plt.show()

print("\nThe boxplot above does not show any obvious outliers, hence we can conclude that none of the profiles are particularly problematic. However, Jane with a Coeffieicnt of Variance of 0.32 is on the slightly extreme end (more opinionanted) compared to her colleagues. Mike on the other hand with a Coefficient of Variance of 0.17 is (less opinionated compared to his colleagues.") 


#%%
###################
#### Problem 9 ####
###################
# Think of two metrics to compute the disatistifaction with the group.  

print_question(9)

def calculate_dissatisfaction(resturant_ratings, resturant_names, group_title="", verbose = 1):
    """
    I created a function that takes the resturant ratings of all resturants (resturant_ratings) for a group and returns the dissatisfaction level for the best (chosen) resturant
        
    Dissatisfaction level is calculated as such
    Step 1. Find the Highest rating for the best resturant
    Step 2. Subtract each individual rating for the best resturant from its highest rating
    Step 3. 
        Metric 1: Take the Standard Deviation of the resulting values. The higher this number, the more dissatisfied the group since it implies that there is more disagreement in the ratings of the best resturant.
        Metric 2: If we suspect that the results are skewed, we can take the Interquartile Range instead of SD. This can be used instead of SD since the IQR is not affected by outliers. 
    """
    
    resturant_cum_rating = resturant_ratings.sum(axis=1) # Cumulative Rating to pick best resturant
    best_resturant_ratings = resturant_ratings[np.argmax(resturant_cum_rating),:]
    
    print("")
    print("-"*75)
    if (verbose >= 1):
        print("Individual Ratings:\n{}\n".format(resturant_ratings))
        print("Cumulative Ratings:\n{}\n".format(resturant_cum_rating))

    print("Best Resturant Pick for group {} --> {}\n".format(group_title, restaurant_names[np.argmax(resturant_cum_rating)]))
    if (verbose >= 1):
        print("Best resturant ratings:\n{}\n".format(best_resturant_ratings))
        
    max_best_resurtant_ratings = np.max(best_resturant_ratings)
    dissatisfaction =  max_best_resurtant_ratings - best_resturant_ratings
    
    if (verbose >= 2):
        print("Absolute Dissatisfaction:\n{}\n".format(dissatisfaction))
    
    from scipy.stats import iqr
    dis_sd = np.std(dissatisfaction)
    dis_iqr = iqr(dissatisfaction)
    
    print("Dissatisfaction for group {} >>".format(group_title))
    print("  1. Based on Standard Deviation = {}".format(dis_sd))
    print("  2. Based on IQR = {}".format(dis_iqr))

print(calculate_dissatisfaction.__doc__)

#%%
####################
#### Problem 10 ####
####################
# Should you split in two groups today? 

print_question(10)

print("IDEA: Apply K-Means on the individual preference with 2 clusters. Then calculate the metric on each cluster to see if it gives a better result than for a single group.")


# Divide the people into 2 groups based on the similarity of their preferences
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(M_people.T) 
groups = kmeans.predict(M_people.T) # "groups" gives the group to which each person belongs

print("\nAnalysis of the groups using cluster centers >>")
print(people_rating_categories)
print(kmeans.cluster_centers_)

print("\nLooks like Group 0 has a higher preference for 'Willingness to Travel' and 'Desire for new experiences' and a low preference for 'cost' and 'vegetarian'.\nOn the other hand, the Group 1 is quite the opposite.")

# Split into 2 groups based on KMeans output
# Group 0
people_names_group0 = [people_names[i] for i in np.where(groups == 0)[0]]
M_people_group0 = M_people.T[groups == 0].T

# Group 1
people_names_group1 = [people_names[i] for i in np.where(groups == 1)[0]]
M_people_group1 = M_people.T[groups == 1].T    

# Find ratings for each group
M_usr_x_rest_group0 = np.dot(M_restaurant, M_people_group0) # Rating for Group 0
M_usr_x_rest_group1 = np.dot(M_restaurant, M_people_group1) # Rating for Group 1

loVerbose = 0
calculate_dissatisfaction(M_usr_x_rest_group0, restaurant_names, group_title = '0', verbose = loVerbose)
calculate_dissatisfaction(M_usr_x_rest_group1, restaurant_names, group_title = '1', verbose = loVerbose)
calculate_dissatisfaction(M_usr_x_rest, restaurant_names, group_title = '(Single Group)', verbose = loVerbose)

print("\nLooks like breaking into 2 groups might be better since the dissatisfaction level will be lower overall")


#%%
####################
#### Problem 11 ####
####################
# Ok. Now you just found out the boss is paying for the meal. How should you adjust. Now what is best restaurant?

print_question(11)

print("\nSince the boss is paying, we will remove cost from the equation by setting it to 0 for everyone.")

M_people_cost0 = M_people.copy()
M_people_cost0[2,:] = 0 

M_usr_x_rest_cost0 = np.dot(M_restaurant, M_people_cost0)
if (verbose >= 2):
    print(M_usr_x_rest_cost0)

calculate_dissatisfaction(M_usr_x_rest_cost0, restaurant_names, group_title = '(without cost consideration)', verbose = 0)

#%%
####################
#### Problem 12 ####
####################
# Tommorow you visit another team. You have the same restaurants and they told you their optimal ordering for restaurants.  Can you find their weight matrix? 

print_question(12)

print("\nJust having the ordering information is not enough to calculate the weight matrix. We need to know the raw scores that they have given to each restraunt in order to calculate the weight matrix. Given the raw scores S and the rating of the resturants R, we can compute their individual preferences P (weights) using the equation R * P' = S. However this is not possible to compute if we are just given their rankings since we lose the raw score information.")
