# Decision making with Matrices

# This is a pretty simple assingment.  You will do something you do everyday, but today it will be with matrix manipulations. 

# The problem is: you and your work firends are trying to decide where to go for lunch. You have to pick a resturant thats best for everyone.  Then you should decided if you should split into two groups so eveyone is happier.  

# Displicte the simplictiy of the process you will need to make decisions regarding how to process the data.
  
# This process was thoughly investigated in the operation research community.  This approah can prove helpful on any number of decsion making problems that are currently not leveraging machine learning.  



# You asked your 10 work friends to answer a survey. They gave you back the following dictionary object.  
people = {'Jane': {'willingness to travel':
                  'desire for new experience':
                  'cost':
                  'indian food':
                  'mexican food':
                  'hipster points':
                  'vegitarian': 
                  }
        
          }          

# Transform the user data into a matrix(M_people). Keep track of column and row ids.   



# Next you collected data from an internet website. You got the following information.

resturants  = {'flacos':{'distance' : 
                        'novelty' :
                        'cost': 
                        'average rating': 
                        'cuisine':
                        'vegitarians'
                        }
  
}


# Transform the restaurant data into a matrix(M_resturants) use the same column index.

# The most imporant idea in this project is the idea of a linear combination.  
# Informally describe what a linear combination is  and how it will relate to our resturant matrix.

# Choose a person and compute(using a linear combination) the top restaurant for them.  What does each entry in the resulting vector represent. 

# Next compute a new matrix (M_usr_x_rest  i.e. an user by restaurant) from all people.  What does the a_ij matrix represent? 

# Sum all columns in M_usr_x_rest to get optimal restaurant for all users.  What do the entry’s represent?

# Now convert each row in the M_usr_x_rest into a ranking for each user and call it M_usr_x_rest_rank.   Do the same as above to generate the optimal resturant choice.  

# Why is there a difference between the two?  What problem arrives?  What does represent in the real world?

# How should you preprocess your data to remove this problem. 

# Find  user profiles that are problematic, explain why?

# Think of two metrics to compute the disatistifaction with the group.  

# Should you split in two groups today? 

# Ok. Now you just found out the boss is paying for the meal. How should you adjust. Now what is best restaurant?

# Tommorow you visit another team. You have the same restaurants and they told you their optimal ordering for restaurants.  Can you find their weight matrix? 



