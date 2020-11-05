#We are going to build a simple item similarity based recommender system.
#The first thing we need to do is to import pandas and numpy.
import pandas as pd 
import numpy as np
import warnings
import os
#Next we load in the data set using pandas read_csv() utility. 
#The dataset is tab separated so we pass in \t to the sep parameter. 
#We then pass in the column names using the names parameter.
os.chdir('c:\\Users\\Rohil\\Documents\\Github\\recommendation-system') 
warnings.filterwarnings('ignore')
df = pd.read_csv('ratings.csv', names=['user_id','item_id','rating','titmestamp'])
#Now let’s check the head of the data to see the data we are dealing with.
df.head()
#It would be nice if we can see the titles of the movie instead of just dealing with the IDs. 
#Let’s load in the movie titles and merge it with this dataset.
movie_titles = pd.read_csv('movies.csv', names=['item_id','title','genres'])
movie_titles.head()
#Since the item_id columns are the same we can merge these datasets on this column.
df1 = pd.merge(df,movie_titles, on='item_id')
df1.head()
#Using the describe or info commands we can get a brief description of our dataset. 
#This is important in order to enable us understand the dataset we are working with.
df1.describe()
#We can tell that the average rating is 3.53 and the max is 5.
'''
Let’s now create a dataframe with the average rating for each movie and the number of ratings. 
We are going to use these ratings to calculate the correlation between the movies later. 
Movies that have a high correlation coefficient are the movies that are most similar to each other. 
In our case we shall use the Pearson correlation coefficient. This number will lie between -1 and 1. 
1 indicates a positive linear correlation while -1 indicates a negative correlation. 0 indicates no linear correlation. 
Therefore movies with a zero correlation are not similar at all. 
In order to create this dataframe we use pandas groupby functionality. 
We group the dataset by the title column and compute its mean to obtain the average rating for each movie.
'''
ratings = pd.DataFrame(df1.groupby('title')['rating'].mean())
ratings.head()
ratings['number_of_ratings'] = df1.groupby('title')['rating'].count()
ratings.head()
import matplotlib.pyplot as plt
#Let’s now plot a Histogram using pandas plotting functionality to visualize the distribution of the ratings
%matplotlib inline
ratings['rating'].hist(bins=50)
#We can see that most of the movies are rated between 2.5 and 4. 
#Next let’s visualize the number_of_ratings column in as similar manner.
ratings['number_of_ratings'].hist(bins=60)
#From the above histogram it is clear that most movies have few ratings. 
#Movies with most ratings are those that are most famous.
import seaborn as sns
#Let’s now check the relationship between the rating of a movie and the number of ratings. 
#We do this by plotting a scatter plot using seaborn. 
#Seaborn enables us to do this using the jointplot() function.
sns.jointplot(x='rating', y='number_of_ratings', data=ratings)
#From the diagram we can see that their is a positive relationship between the average rating of a movie and the number of ratings. 
#The graph indicates that the more the ratings a movie gets the higher the average rating it gets.
'''
So not that we know about the data that we are dealing with. 
Let’s now create a simple item based recommender system. 
In order to do this we need to convert our dataset into a matrix with the movie titles as the columns, the user_id as the index and the ratings as the values.
We shall use this matrix to compute the correlation between the ratings of a single movie and the rest of the movies in the matrix. 
We use pandas pivot_table utility to create the movie matrix.
'''
movie_matrix = df1.pivot_table(index='user_id', columns='title', values='rating')
movie_matrix.head()
'''
Next let’s look at the most rated movies and choose two of them to work with in this simple recommender system. 
We use pandas sort_values utility and set ascending to false in order to arrange the movies from the most rated. 
We then use the head() function to view the top 10.
'''
ratings.sort_values('number_of_ratings', ascending=False).head(10)
'''
Let’s assume that a user has watched forest gump (1997) and Shawshank redemption (1997). 
We would like like to recommend movies to this user based on this watching history. 
The goal is to look for movies that are similar to Forest gump (1997) and Shawshank redemption (1997 which we shall recommend to this user. 
We can achieve this by computing the correlation between these two movies’ ratings and the ratings of the rest of the movies in the dataset. 
The first step is to create a dataframe with the ratings of these movies from our movie_matrix.
'''
forest_gump_user_rating = movie_matrix['Forrest Gump (1994)']
Shawshank_Redemption_user_rating = movie_matrix['Shawshank Redemption, The (1994)']
forest_gump_user_rating.head()
Shawshank_Redemption_user_rating.head()
'''
We now have the dataframes showing the user_id and the rating they gave the two movies. 
Let's take a look at them below.
'''
similar_to_forest_gump=movie_matrix.corrwith(forest_gump_user_rating)
similar_to_forest_gump.head(20)
'''
In order to compute the correlation between two dataframes we use pandas corwith functionality. 
Corrwith computes the pairwise correlation of rows or columns of two dataframe objects. 
Let's use this functionality to get the correlation between each movie's rating and the ratings of the Shawshank redemption movie.
'''
similar_to_Shawshank_Redemption = movie_matrix.corrwith(Shawshank_Redemption_user_rating)
similar_to_Shawshank_Redemption.head(20)
corr_Shawshank_Redemption = pd.DataFrame(similar_to_Shawshank_Redemption, columns=['Correlation'])
corr_Shawshank_Redemption.dropna(inplace=True)
corr_Shawshank_Redemption.head()
corr_forest_gump = pd.DataFrame(similar_to_forest_gump, columns=['correlation'])
corr_forest_gump.dropna(inplace=True)
corr_forest_gump.head()
corr_forest_gump = corr_forest_gump.join(ratings['number_of_ratings'])
corr_Shawshank_Redemption = corr_Shawshank_Redemption.join(ratings['number_of_ratings'])
corr_forest_gump .head()
corr_Shawshank_Redemption.head()
corr_forest_gump[corr_forest_gump['number_of_ratings'] > 100].sort_values(by='correlation', ascending=False).head(10)
corr_Shawshank_Redemption[corr_Shawshank_Redemption['number_of_ratings'] > 100].sort_values(by='Correlation', ascending=False).head(10)