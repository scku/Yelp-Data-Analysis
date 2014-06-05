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

def create_graph_data_struct(data):
	nodes = []
	adj_list = []

	# Add users to nodes
	for user in data['user_id']:
		nodes.append(user)
	# Create adjacency list for each user/node
	for friends_str in data['friends']:
		cur_list = []
		friends_list = friends_str.strip().split(', ')
		for friend in friends_list:
			#if friend in data['user_id'].tolist():
			if friend not in nodes:
				nodes.append(friend)
			cur_list.append(friend)
		adj_list.append(cur_list)

	print "Number of nodes: {0}".format(len(nodes))
	print "Number of adjacent lists: {0}".format(len(adj_list))
	return nodes, adj_list


def build_graph(nodes, adj_list):
	G = nx.Graph()
	#num_of_friends = {}

	print "Adding nodes"
	for i in range(len(nodes)):
		#if len(adj_list[i]) > 0:
		G.add_node(nodes[i])

	print "Adding edges"
	for i in range(len(adj_list)):
		if len(adj_list[i]) > 0:
			#num_of_friends[nodes[i]] = math.log(len(adj_list[i]))/math.log(2)+1
			for friend in adj_list[i]:
				if friend != '':
					G.add_edge(nodes[i], friend)
	return G

''' 2D Graph ''' 
def draw_graph2d(G):
	print "Drawing 2D graph"
	#num_of_friends = {}

	fig = plt.figure(figsize=(10, 10))

	values = [num_of_friends.get(x) for x in num_of_friends.keys()]
	#values = [0 if x is None else x for x in values]
	#print "Degrees: {0}".format(values)
	degrees = nx.degree_centrality(G).values()
	print len(degrees)
	#print degrees

	nx.draw(G, node_size=50, node_color=degrees, cmap=plt.cm.Reds, edge_color='#B2B2B2',
	        with_labels=False, alpha=0.7, pos=nx.spring_layout(G, dim=2))

	plt.title('Yelp Connectivity Graph')
	fig.set_facecolor('#194775')
	plt.savefig('yelp_graph_2d.png')
	plt.show()

''' 3D Graph ''' 
def draw_graph3d(G, graph_colormap='Reds', bgcolor=(25/255.0, 43/255.0, 75/255.0),
                 node_size=0.005,
                 edge_color=(0.9, 0.9, 0.5), edge_size=0.0005,
                 text_size=0.008, text_color=(0, 0, 0)):
	print "Drawing 3D graph"
	#num_of_friends = {}

	G = nx.convert_node_labels_to_integers(G)
	# k: node spacing, increase to spread out
	graph_pos = nx.spring_layout(G, dim=3, k=1/math.sqrt(len(G.nodes())))
	#graph_pos = nx.random_layout(G, dim=3)

	# numpy array of x,y,z positions in sorted node order
	xyz = np.array([graph_pos[v] for v in sorted(G)])

	#values = [num_of_friends.get(x) for x in num_of_friends.keys()]
	if DEBUG: 
		print "Length of values: {0}".format(len(values))
		print values
	degrees = nx.degree_centrality(G).values()
	print "Min: {0}, Max: {1}".format(min(degrees), max(degrees))
	min_degree = min(degrees)
	# Scaling
	#degrees = [math.sqrt(x/min_degree)/1000 for x in degrees]
	#degrees = [((math.log(x/min_degree)/math.log(1.1))+1) for x in degrees]

	print "After scaling - Min: {0}, Max: {1}".format(min(degrees), max(degrees))

	if DEBUG:
		print "Dimension of degrees: {0}".format(len(degrees))
		print degrees

	mlab.figure(1, bgcolor=bgcolor)
	mlab.clf()

	print "Drawing points"
	pts = mlab.points3d(xyz[:,0], xyz[:,1], xyz[:,2], degrees,
		                #scale_factor=node_size,
		                #scale_mode='vector',
		                colormap=graph_colormap,
		                opacity = 0.5,
		                resolution=20)

	pts.mlab_source.dataset.lines = np.array(G.edges())
	tube = mlab.pipeline.tube(pts, tube_radius=edge_size)
	mlab.pipeline.surface(tube, color=edge_color, opacity=0.01) #0.01 for dense graph
	mlab.show()

def analyze_graph(G):
	print nx.info(G)
	degree_freq = nx.degree_histogram(G)
	diameter = nx.diameter(G)
	print "Diameter: {0}".format(diameter)
	triangles = nx.triangles(G)
	triangles_values = sorted(triangles.values(), reverse=True)
	print "Triangles: {0}".format(triangles_values)

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

	user_df['friend_count'] = [len(x.strip().split(', ')) for x in user_df['friends']]
	print "Friend count: {0}".format(user_df['friend_count'])
	print "Dimension: {0}".format(user_df.shape)

	sorted_user_df = user_df.sort(['friend_count'], ascending=False)
	#print sorted_user_df


	''' Sample subset '''
	index = random.sample(user_df.index, 200)
	subset_df = user_df.ix[index]
	print "Subset dimension: {0}".format(subset_df.shape)
	#print "User id: {0}".format(user_df['user_id'])
	#print "User id (subset): {0}".format(subset_df['user_id'])

	top_5 = sorted_user_df[:5]
	top_10 = sorted_user_df[:10]
	top_20 = sorted_user_df[:20]
	top_100 = sorted_user_df[:100]
	top_500 = sorted_user_df[:500]
	top_1000 = sorted_user_df[:1000]
	top_5000 = sorted_user_df[:5000]
	if DEBUG:
		print "Dimension top_10: {0}".format(top_10.shape)
		print "Dimension top_20: {0}".format(top_20.shape)
		print "Dimension top_100: {0}".format(top_100.shape)
		print "Dimension top_1000: {0}".format(top_1000.shape)
		print "Dimension top_5000: {0}".format(top_5000.shape)

	nodes_subset, adj_list_subset = create_graph_data_struct(subset_df)
	nodes_top_5, adj_list_top_5 = create_graph_data_struct(top_5)
	nodes_top_10, adj_list_top_10 = create_graph_data_struct(top_10)
	nodes_top_100, adj_list_top_100 = create_graph_data_struct(top_100)
	nodes_top_1000, adj_list_top_1000 = create_graph_data_struct(top_1000)
	nodes_top_500, adj_list_top_500 = create_graph_data_struct(top_500)
	#nodes_top_5000, adj_list_top_5000 = create_graph_data_struct(top_5000)
	nodes_all, adj_list_all = create_graph_data_struct(sorted_user_df)

	''' Build graph G '''
	G_10 = build_graph(nodes_top_10, adj_list_top_10)
	G_100 = build_graph(nodes_top_100, adj_list_top_100)
	G_1000 = build_graph(nodes_top_1000, adj_list_top_1000)
	G_all = build_graph(nodes_all, adj_list_all)

	''' Analysis graph G '''
	print "--- Analysis of graph G_10 ---"
	#analyze_graph(G_10)
	print "--- Analysis of graph G_100 ---"
	#analyze_graph(G_100)
	print "--- Analysis of graph G_1000 ---"
	#analyze_graph(G_1000)
	print "--- Analysis of graph G_all ---"
	#analyze_graph(G_all)

	''' Draw graph G '''
	#draw_graph3d(G_5)
	#draw_graph3d(G_10)
	draw_graph3d(G_100)
	#draw_graph3d(nodes_top_500, adj_list_top_500)
	draw_graph3d(G_1000)
	#draw_graph2d(nodes_subset, adj_list_subset)

	return

if __name__ == "__main__":
	main()