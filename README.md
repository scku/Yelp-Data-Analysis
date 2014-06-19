Yelp-Data-Analysis
==================
#####Partners
Stanley Ku - mr.stanleyku@gmail.com <br />
Brian Liu - liubrian7@gmail.com<br>

This is a collection of analysis that we did on the Yelp datashet challenge (2014). The work is organized into IPython notebooks. For easy viewing, use NBViewer: http://nbviewer.ipython.org/github/scku/Yelp-Data-Analysis/tree/master/ 

We separated our work into 3 majors parts.

###Part I - Review Rating Classification
The goal is to classify restaurant review ratings. We started off with a binary classifcation: 1-star or a 5-star rating and generalized it to all 5 classes. We used Latent Direchlet Allocation to modeling the topics in the reviews. We counted the most likely topics for a given class and found a difference in the topic distribution between the 1-star and 5-star reviews. Given this disparity, we used the given topic probabilities from the LDA model as the feature set. We trained various models and found that Linear Discrimination Analysis resulted in upwards of 85% accuracy in the binary classification.

###Part II - Check-in Times and Restaurant Star Ratings
We fitted and examined the distribution of check-in times and restaurant star ratings to gain some insight of the data. We found the a few hours on Friday represented 10% of the total check-in times. We tried using the check-in times as the feature set for predicating a restaurant star rating but found that the check-in times had weak predictive power.

###Part III - Social Network Analysis
We studied the structure and topology of the Yelp social network. We found evidence of a small world network. A small subset of users act as hubs and the main connected component has a dimaeter of 6. Lastly, we tried some visualization technique to generate the network graph.

####Reference:
http://www.yelp.com/dataset_challenge/
