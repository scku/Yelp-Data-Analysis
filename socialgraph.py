import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import csv
import random
import math
from mayavi import mlab

DEBUG = 0

''' Preprocessing '''
def preprocessing(a_list):
	new_list = []
	for item in a_list:
		new_item = re.sub(',', '', item)
		new_item = re.sub("u'", '', new_item)
		new_item = re.sub("'", ',', new_item)
		new_item = re.sub(",]", '', new_item)
		new_item = re.sub("]", '', new_item)
		new_item = re.sub("\\[", '', new_item)
		new_list.append(new_item)
	return new_list

''' 2D Graph ''' 
def draw_graph2d(nodes, adj_list):
	print "Drawing 2D graph"
	num_of_friends = {}

	G = nx.Graph()

	for i in range(len(nodes)):
		if len(adj_list[i]) > 0:
			G.add_node(nodes[i])

	for i in range(len(nodes)):
		if len(adj_list[i]) > 0:
			num_of_friends[nodes[i]] = math.log(len(adj_list[i]))/math.log(10)+1
			for friend in adj_list[i]:
				if friend != '':
					G.add_edge(nodes[i], friend)

	fig = plt.figure(figsize=(10, 10))

	values = [num_of_friends.get(x) for x in num_of_friends.keys()]
	#values = [0 if x is None else x for x in values]
	#print "Degrees: {0}".format(values)
	nx.draw(G, node_size=50, node_color=values, cmap=plt.cm.Reds, edge_color='#B2B2B2',
	        with_labels=False, alpha=0.7, pos=nx.spring_layout(G))

	plt.title('Yelp Connectivity Graph')
	fig.set_facecolor('#194775')
	plt.savefig('yelp_graph_2d.png')
	plt.show()

''' 3D Graph ''' 
def draw_graph3d(nodes, adj_list, graph_colormap='Reds', bgcolor=(25/255.0, 43/255.0, 75/255.0),
                 node_size=0.01,
                 edge_color=(0.9, 0.9, 0.5), edge_size=0.0005,
                 text_size=0.008, text_color=(0, 0, 0)):
	print "Drawing 3D graph"
	num_of_friends = {}

	G = nx.Graph()

	'''
	for node in nodes:
		G.add_node(node)
	'''

	for i in range(len(nodes)):
		if len(adj_list[i]) > 0:
			G.add_node(nodes[i])

	for i in range(len(nodes)):
		if len(adj_list[i]) > 0:
			num_of_friends[nodes[i]] = math.log(len(adj_list[i]))/math.log(10)+1
			for friend in adj_list[i]:
				if friend != '':
					G.add_edge(nodes[i], friend)

	G = nx.convert_node_labels_to_integers(G)
	graph_pos = nx.spring_layout(G, dim=3, k=1/math.sqrt(len(G.nodes())/2))
	#graph_pos = nx.random_layout(G, dim=3)

	# numpy array of x,y,z positions in sorted node order
	xyz = np.array([graph_pos[v] for v in sorted(G)])

	# scalar colors
	#scalars = np.array(G.nodes())+5

	#print num_of_friends
	values = [num_of_friends.get(x) for x in num_of_friends.keys()]
	#values = [0 if x is None else x for x in values]
	if DEBUG: print values

	mlab.figure(1, bgcolor=bgcolor)
	mlab.clf()

	pts = mlab.points3d(xyz[:,0], xyz[:,1], xyz[:,2],
		                values,
		                scale_factor=node_size,
		                scale_mode='none',
		                colormap=graph_colormap,
		                resolution=20)

	pts.mlab_source.dataset.lines = np.array(G.edges())
	tube = mlab.pipeline.tube(pts, tube_radius=edge_size)
	mlab.pipeline.surface(tube, color=edge_color, opacity=0.03)
	mlab.show()

def main():
	user_df = pd.read_csv("yelp_academic_dataset_user.csv")
	print "Dimension: {0}".format(user_df.shape)
	if DEBUG: print "Column names: {0}".format(user_df.columns)

	user_df = user_df[['user_id', 'friends']]

	# Remove users with no friends
	user_df = user_df[user_df['friends'] != "[,]"]
	print "Dimension of user dataset w/ friends: {0}".format(user_df.shape)
	friends = user_df['friends']

	friends = preprocessing(friends)
	if DEBUG: print len(friends)

	user_df['friends'] = friends
	#print "Dimension: {0}".format(user_df.shape)
	#print user_df['friends']

	''' Sample subset '''
	index = random.sample(user_df.index, 10000)
	subset_df = user_df.ix[index]
	print "Subset dimension: {0}".format(subset_df.shape)
	#print "User id: {0}".format(user_df['user_id'])
	#print "User id (subset): {0}".format(subset_df['user_id'])

	nodes = []
	adj_list = []
	num_of_friends = {}

	# Add users to nodes
	for user in subset_df['user_id']:
		nodes.append(user)

	# Create adjacency list for each user/node
	for friends_str in subset_df['friends']:
		cur_list = []
		friends_list = friends_str.strip().split(', ')
		for friend in friends_list:
			if friend in subset_df['user_id'].tolist():
				cur_list.append(friend)
		adj_list.append(cur_list)

	print "Number of nodes: {0}".format(len(nodes))
	print "Number of adjacency lists: {0}".format(len(adj_list))

	#draw_graph2d(nodes, adj_list)
	draw_graph3d(nodes, adj_list)
	return

if __name__ == "__main__":
	main()