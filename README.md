# Getting into clustering or cluster-analysis

## What is clustering or cluster analysis


Clustering or cluster analysis is an aspect of statistical data analysis where you segment the datasets into groups based on similarities. Here, the data points that are close to one another are grouped together to form clusters. Clustering is one of the core data mining techniques and comes under umbrella of unsupervised Machine Learning. Clustering, being an unsupervised learning technique, does not require much human intervention and any pre-existing labels. This technique is generally used for exploratory data analysis.


Lets understand the use of cluster analysis with an example. Lets assume the mayor of Kathmandu wants to plan efficient allocation of budget for road safety but isn't exactly sure about which areas to focus on. Here, using cluster analysis they find out that the incidents of road accidents are clustered in few specific areas within the city. Therefore, he plans the infrastructure development and allocation of traffic personnel focused on those areas which in turn significantly reduces the number of accidents. Clustering can be a great way to find patterns and relationships within a large data set. There are various uses of clustering in various industries. Some common application of clustering are : 

- Customer segmenting for targeted marketing
- Recommendation system to suggest songs, movies, contents to users with similar preferences
- Detecting anomalities for fraud detection, predictive maintenance and network security
- Image segmentation for tumor detection through medical imaging, land cover classification and computer vision for self driving cars
- Spatial data analysis for city planning, disease survellience, crime analysis, etc.


While there are numerous clustering techniques, the K-means clustering is used most widely. There is no best clustering technique and the most favorable technique is determined by the properties of data and the purpose of analysis. As of now there are more than 100 different types of algorithms used in clustering. However, the clustering techniques most commonly used can be divided in following categories: 

- Partitional clustering : K-means, K-medoids, Fuzzy C-means
- Hieriarchial clustering : Agglomerative (Bottom-up approach), Divisive (Top-down approach)
- Density-Based clustering : DBSCAN, OPTICS, DENCLUE
- Grid-Based clustering : STING, WaveCluster
- Model-Based clustering : EM, COBWEB, SOM

# Customer segmentation using Clustering in Python


In this project, we will be conducting Agglomerative and K-means clustering on customer demographic data to segment customers for targeted marketing. Major steps we will take in the project are :

1. Feature engineering and data cleaning
2. Scaling the data
3. Reducing the dimensionality using PCA
4. Using Elbow method to determine optimal no of clusters
5. Clustering using Agglomerative and K-means clustering
6. Checking the quality of the results using Silhoutte score
7. Checking the results of clustering
8. Interpreting the results of clustering
   

The dataset used in this project is : https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis/data


# The dataset we will be working has following attributes:

People

- ID: Customer's unique identifier
- Year_Birth: Customer's birth year
- Education: Customer's education level
- Marital_Status: Customer's marital status
- Income: Customer's yearly household income
- Kidhome: Number of children in customer's household
- Teenhome: Number of teenagers in customer's household
- Dt_Customer: Date of customer's enrollment with the company
- Recency: Number of days since customer's last purchase
- Complain: 1 if the customer complained in the last 2 years, 0 otherwise
  
Products

- MntWines: Amount spent on wine in last 2 years
- MntFruits: Amount spent on fruits in last 2 years
- MntMeatProducts: Amount spent on meat in last 2 years
- MntFishProducts: Amount spent on fish in last 2 years
- MntSweetProducts: Amount spent on sweets in last 2 years
- MntGoldProds: Amount spent on gold in last 2 years
  
Promotion

- NumDealsPurchases: Number of purchases made with a discount
- AcceptedCmp1: 1 if customer accepted the offer in the 1st campaign, 0 otherwise
- AcceptedCmp2: 1 if customer accepted the offer in the 2nd campaign, 0 otherwise
- AcceptedCmp3: 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
- AcceptedCmp4: 1 if customer accepted the offer in the 4th campaign, 0 otherwise
- AcceptedCmp5: 1 if customer accepted the offer in the 5th campaign, 0 otherwise
- Response: 1 if customer accepted the offer in the last campaign, 0 otherwise
  
Place

- NumWebPurchases: Number of purchases made through the company’s website
- NumCatalogPurchases: Number of purchases made using a catalogue
- NumStorePurchases: Number of purchases made directly in stores
- NumWebVisitsMonth: Number of visits to company’s website in the last month
