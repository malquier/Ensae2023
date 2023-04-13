from graph import Graph, graph_from_file
from Fileprio import *
from Union_set import * 


## Initialisation:

data_path = "input/"
file_name_network = "network.2.in"
file_name_route = "routes.2.in"
file_name_truck = "trucks.2.in"

g2 = graph_from_file(open(data_path + file_name_network,"r").read())
routes = route_from_file(open(data_path + file_name_route,"r").read())
trucks = trucks2(open(data_path + file_name_truck,"r").read())
budget = 10**9


## Simplification du graphe par kruskal:

# g = kruskal(g2)


## Puissance minimale par kruskal :

# print(puissance_power_kruskal(g2,1,500))

## Knapsack par génération :

# print(knapsack_generative(g2, routes, trucks, budget))

##
