import pandas as pd 
import numpy as np
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('C:\\Users\\Rohil\\Downloads\\ml-25m\\ml-25m\\ratings.csv', names=['user_id','item_id','rating','titmestamp'])
df.head()
movie_titles = pd.read_csv('C:\\Users\\Rohil\\Downloads\\ml-25m\\ml-25m\\movies.csv', names=['item_id','title','genres'])
movie_titles.head()
df1 = pd.merge(df,movie_titles, on='item_id')
df1.head()
df1.describe()
ratings = pd.DataFrame(df1.groupby('title')['rating'].mean())
ratings.head()
ratings['number_of_ratings'] = df1.groupby('title')['rating'].count()
ratings.head()
import matplotlib.pyplot as plt
%matplotlib inline
ratings['rating'].hist(bins=50)
ratings['number_of_ratings'].hist(bins=60)
import seaborn as sns
sns.jointplot(x='rating', y='number_of_ratings', data=ratings)
movie_matrix = df1.pivot_table(index='user_id', columns='title', values='rating')
movie_matrix.head()
ratings.sort_values('number_of_ratings', ascending=False).head(10)
forest_gump_user_rating = movie_matrix['Forrest Gump (1994)']
Shawshank_Redemption_user_rating = movie_matrix['Shawshank Redemption, The (1994)']
forest_gump_user_rating.head()
Shawshank_Redemption_user_rating.head()
similar_to_forest_gump=movie_matrix.corrwith(forest_gump_user_rating)
similar_to_forest_gump.head(20)
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