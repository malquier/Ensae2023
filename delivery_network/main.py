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

## Test du temps d'éxécution de puissance_mini_power :

def test(filename_network,filename_route):
    g = graph_from_file(open(filename_network).read())
    g,u = kruskal(g)
    m = matrice(open(filename_route).read())
    routes0 = []
    n = len(m)
    t1 = time.time()
    for i in range(1,n):
        if len(m[i][0])>0 and len(m[i][1])>0:
            src,dest = int((m[i][0])),int((m[i][1]))
            src,dest = src-1,dest-1
            routes0.append(Union_set.path(u, src, dest)[0])
    t2 = time.time()
    return(t2-t1,routes0)
  
  print(test(filename_network, filename_route))

## Knapsack par génération :

# print(knapsack_generative(g2, routes, trucks, budget))
