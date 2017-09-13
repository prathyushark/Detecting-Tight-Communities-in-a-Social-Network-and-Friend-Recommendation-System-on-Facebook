# Databricks notebook source
#import data set
fbCombinedFileName = "/FileStore/tables/n3q0cgsv1501073339277/facebook_combined.txt"
fbCombinedFile = sc.textFile(fbCombinedFileName)
fbCombinedFile.take(10)

# COMMAND ----------

#retrieve vertex and edges from the data set
def get_vertex1_tuple(entry):
  row = entry.split(' ')
  return int(row[0])

def get_vertex2_tuple(entry):
  row = entry.split(' ')
  return int(row[1])

def get_edge_tuple(entry):
  row = entry.split(' ')
  return int(row[0]),int(row[1])

# COMMAND ----------

#create vertex and edges RDD
from pyspark.sql import Row
vertext1RDD = fbCombinedFile.map(get_vertex1_tuple).cache().distinct()
vertext2RDD = fbCombinedFile.map(get_vertex2_tuple).cache().distinct()
vertex1Union2 = vertext1RDD.union(vertext2RDD)
vertexRDD = vertex1Union2.distinct()
vertexCount = vertexRDD.count()
print (vertexCount)
print 'Vertices: %s' % vertexRDD.takeOrdered(5)
edgesRDD = fbCombinedFile.map(get_edge_tuple).cache()
ecount = edgesRDD.count()
print (ecount)
print 'edges: %s' % edgesRDD.take(5)


# COMMAND ----------

#import igraph package 
from igraph import *

#build igraph with vertices and edges from the dataset
vertices = vertexRDD.collect()
edges = edgesRDD.collect()
g = Graph(vertex_attrs={"label":vertices}, edges=edges, directed=False)

# COMMAND ----------

#overall dataset analysis on the built graph
print g.is_connected(mode=STRONG)
print g.farthest_points(directed=False, unconn=True, weights=None)
nwDiameter = g.diameter(directed=False, unconn=True, weights=None)
print nwDiameter
print g.get_diameter(directed=False, unconn=True, weights=None)
nwBetweeness = g.betweenness(vertices=None, directed=False, cutoff=None, weights=None, nobigint=True)
meanNwBetweeness= reduce(lambda x, y: x + y, nwBetweeness) / len(nwBetweeness)
print meanNwBetweeness

# COMMAND ----------

#check degree distribution of the network
nwDegrees = g.degree()
meanNwDegree= reduce(lambda x, y: x + y, nwDegrees) / len(nwDegrees)
print meanNwDegree
from operator import add
nwDegreesRDD = sc.parallelize(nwDegrees)
counts = nwDegreesRDD.map(lambda x: (x, 1)).reduceByKey(add)
output = counts.collect()
for (degree, count) in output:
  print("%s %i" % (degree, count))

# COMMAND ----------

#identify insignificant nodes
island_list = []
for v in vertices:
  friends_list = g.neighbors(vertex=v, mode=ALL)
  if (len(friends_list) < 2):
    island_list.append(v)
print set(island_list)

island_degree_list=[]
for i in island_list:
  island_degree_list.append(g.degree(i))
print  set(island_degree_list)

#remove island nodes from the graph 
g.delete_vertices(island_list)
newVertices = []
newEdges = []
for v in g.vs:
    newVertices.append(v["label"])
for e in g.es:
    newEdges.append(e.tuple) 
print len(set(vertices))    
print len(set(island_list))    
print len(set(newVertices))    

# COMMAND ----------

#identify significant nodes
core_node_list = []
core_degree_list = []
for v in g.vs:
  v_degree = g.degree(v)
  if(v_degree > 300): 
    core_node_list.append(v.index)
    core_degree_list.append(v_degree)
print set(core_node_list)
mean_core_degree = reduce(lambda x, y: x + y, core_degree_list) / len(core_degree_list)
print mean_core_degree
#sub graph focussing on node "0" which was identified as significant node
node0_friends_list = g.neighbors(vertex=0, mode=ALL)
freinds_of_friends = g.neighborhood(vertices=0, order=2, mode=ALL)
print len(node0_friends_list)
print len(freinds_of_friends)

node0_friends_list.append(0)
node0_alters = []
node0_graph = g.subgraph(node0_friends_list, implementation = "auto")

for e in node0_graph.es:
    print e.tuple
    node0_alters.append(e.tuple)

# COMMAND ----------

#identify cliques on the subgraph
cliques_0 = node0_graph.maximal_cliques(min =2 , max =10)
print cliques_0

# COMMAND ----------

#community detection with centrality based approach using edge betweeness
communities = node0_graph.community_edge_betweenness(directed=False)
clusters = communities.as_clustering()
print clusters.modularity
print clusters


# COMMAND ----------

#community detection using fast greedy algorithm
fastGreedy = node0_graph.community_fastgreedy()
FGcluster = fastGreedy.as_clustering()
print FGcluster.modularity
print FGcluster

# COMMAND ----------

#community detection using walk trap algorithm
walkTrap = node0_graph.community_walktrap() 
WTcluster = walkTrap.as_clustering()
print WTcluster.modularity
print WTcluster

# COMMAND ----------

#community detection using info map algorithm
infoMap = node0_graph.community_infomap()
print infoMap.modularity
print infoMap.as_cover()

# COMMAND ----------

#Part 2 - Friend Recommendation based on clusters detected
#Extract tuples from dataset
def returnTuple(entry):
  row = entry.split(' ')
  return int(row[0]),int(row[1]),-1

egoRDD = fbCombinedFile.map(returnTuple)


# COMMAND ----------

#Detect no.of.mutual friends for any two pairs of nodes from the graph 
mutualFriends=[]
def generate(x):
  toNodes=[]
  for row in egoRDD.collect():
    if row[0]==x:
      toNodes.append(row[1])
  for i in range(0,len(toNodes)-1):
    mutualFriends.append([toNodes[i],toNodes[i+1],1])
    
prev = -1
  
for row in egoRDD.collect():
  if row[0]!=prev:
    generate(row[0])
  prev=row[0]
  
def predict(entry):
  return (entry[0],entry[1]),entry[2]
  
mutualFriendsRDD =sc.parallelize(mutualFriends)
prediction=mutualFriendsRDD.map(predict)

PredictionRDD = prediction.reduceByKey(lambda a, b: a + b)
sortedRdd=PredictionRDD.sortBy(lambda a: -a[1])
print sortedRdd.collect()

# COMMAND ----------

#Select one user for whom friend suggestion has to be made
fromuser=115
#Filter mutual friend list for the selected user
suggestions_115_1 = sortedRdd.filter(lambda x:x[0][0]==fromuser).map(lambda x:(x[0][1],x[1]))
suggestions_115_2 = sortedRdd.filter(lambda x:x[0][1]==fromuser).map(lambda x:(x[0][0],x[1]))
suggestions_115 = suggestions_115_1.union(suggestions_115_2)
suggestions_115_sorted = suggestions_115.sortBy(lambda x:-x[1])
suggestions_115_RDD = suggestions_115_sorted.map(lambda x:x[0])
print suggestions_115_RDD.collect()

# COMMAND ----------

#Get all friends of user 115
friends_115_1= egoRDD.filter(lambda x:x[0]==fromuser).map(lambda x:x[1])
friends_115_2= egoRDD.filter(lambda x:x[1]==fromuser).map(lambda x:x[0])
friends_115 = friends_115_1.union(friends_115_2)
print friends_115.collect()

# COMMAND ----------

#Get all non friends of user 115
already_friends = suggestions_115_RDD.intersection(friends_115)
suggestions = suggestions_115_RDD.subtract(already_friends)
print suggestions.collect()

# COMMAND ----------

#Narrow down suggestion based on communities
#Communities detected by fastgreedy is opted because of better modularity
suggestion_list = suggestions.collect()
community_based_suggestion=[]
for cluster_index in range(8):
  for member in suggestion_list:
    if member in FGcluster[cluster_index] and 115 in FGcluster[cluster_index]:
      community_based_suggestion.append(member)

print community_based_suggestion

# COMMAND ----------


