import numpy
import copy
import time
import graphviz
import random
import itertools
from Union_set import *
from Fileprio import *

class Graph:
    """
    A class representing graphs as adjacency lists and implementing various algorithms on the graphs. Graphs in the class are not oriented. 
    Attributes: 
    -----------
    nodes: NodeType
        A list of nodes. Nodes can be of any immutable type, e.g., integer, float, or string.
        We will usually use a list of integers 1, ..., n.
    graph: dict
        A dictionnary that contains the adjacency list of each node in the form
        graph[node] = [(neighbor1, p1, d1), (neighbor1, p1, d1), ...]
        where p1 is the minimal power on the edge (node, neighbor1) and d1 is the distance on the edge
    nb_nodes: int
        The number of nodes.
    nb_edges: int
        The number of edges. 
    """

    def __init__(self, nodes=[]):
        self.nodes = nodes
        self.graph = dict([(n, []) for n in nodes])
        self.nb_nodes = len(nodes)
        self.nb_edges = 0

    def __str__(self):
        """Prints the graph as a list of neighbors for each node (one per line)"""
        if not self.graph:
            output = "The graph is empty"
        else:
            output = f"The graph has {self.nb_nodes} nodes and {self.nb_edges} edges.\n"
            for source, destination in self.graph.items():
                output += f"{source}-->{destination}\n"
        return output

    def add_edge(self, node1, node2, power_min, dist=1):
        self.graph[node1] = self.graph[node1] + [(node2, power_min, dist)]
        self.graph[node2] = self.graph[node2] + [(node1, power_min, dist)]
        self.nb_edges += 1

    """ 
    Composantes connexes d'un graphe
    Parcours en Profondeur
    """

    def connected_components_set(self):
        n = self.nb_nodes
        """ On suppose qu'il n'y a jamais le noeud 0 """
        atraiter = [False]*(n+1)
        comp = []

        def pp(u):
            l = [u]
            c = [u]
            atraiter[u] = True
            while len(l) > 0:
                vois = self.graph[l[-1]]
                l.pop()
                for p in vois:
                    if not atraiter[p[0]]:
                        l.append(p[0])
                        atraiter[p[0]] = True
                        c.append(p[0])
            return(c)
        for i in range(1, n+1):
            if not atraiter[i]:
                comp.append(frozenset(pp(i)))
        return(set(comp))

    """ Avant de créer la fonction get_path_with_power, on créé des sous fonctions
    qui vont nous permettre de simplifier le code. On va alors coder les fonctions 
    adj et power liée à la classe Graph et une fonction include indépendante de la
    classe Graph. 
        La première fonction adj prend en argument le numéro d'un noeud 
    et renvoie la liste de numéros de tous les noeuds adjacents.
        La fonction power prend en argument deux noeuds supposés adj, et renvoie 
    la puissance minimale nécessaire afin d'aller d'un noeud à l'autre.
        La fonction include (en bas du code) prend en argument deux listes et renvoie
    True si les éléments de la première liste sont inclus dans la deuxième et sinon il
    renvoie un élément appartenant à la première liste mais pas à la seconde.
    """

    def adj(self, i):
        graph = copy.deepcopy(self.graph)
        adj = []
        for noeud in graph[i]:
            adj.append(noeud[0])
        return adj

    def power(self, i, j):
        graph = self.graph
        for noeud in graph[i]:
            if noeud[0] == j:
                return(noeud[1])


    def get_path_with_power(self, src, dest, p, visited=None, real_visited=None, p_tot=0):
        if visited == None:
            """ La liste visited contiendra tous les noeuds parcourus durant la récursivité"""
            visited = [src]

        if real_visited == None:
            """ La liste real_visited contiendra seulement les noeuds du chemin qui est en train d'être explorer. 
            La liste est donc inclue dans visited"""
            real_visited = [src]

        adj_set = self.connected_components_set()
        """ On vérifie tout d'abord si un chemin existe entre la source et al destination"""
        for s in adj_set:
            if src in s and dest not in s:
                return (None)

        if src == dest:
            """Lorsque l'on a trouvé un chemin possible, on distingue deux cas : le cas où la puissance du camion est suffisante et le cas
            où ce n'est pas le cas."""
            if p >= p_tot:
                return(real_visited, p_tot)

            else:
                """ Si la puissance du chemin actuel est plus élevée que la puissance maximale du camion, on retourne alors deux noeuds avant 
                afin d'éviter de revenir en boucle sur ce même chemin. On suppose églement que la puissance minimale entre deux noeuds adjacents
                est la puissance liant ces deux noeuds."""
                if len(real_visited) > 2:
                    new_p_tot = p_tot - \
                        self.power(real_visited[-1], real_visited[-2]) - \
                        self.power(real_visited[-2], real_visited[-3])
                    real_visited.pop()
                    real_visited.pop()
                    #print(real_visited, new_p_tot)
                    return(self.get_path_with_power(real_visited[-1], dest, p, visited, real_visited, new_p_tot))

                else:
                    return ("Pas assez de puissance")

        """ Si le nouveau noeud n'est pas la destination à atteindre, on regarde alors quels noeuds n'ont pas été parcouru
        afin de continuer la récursivité."""

        adj = self.adj(src)
        if Graph.include(adj, visited) == True:
            if len(real_visited) == 1:
                return (None)
            else:
                real_visited.remove(src)
                new_p_tot = p_tot - self.power(src, real_visited[-1])
                #print('Retour',visited, real_visited, new_p_tot)
                return(self.get_path_with_power(real_visited[-1], dest, p, visited, real_visited, new_p_tot))

            """ Si tous les neouds adjacents à la source ont été parcouru, alors on revient en arrière sur le chemin afin de 
            trouver une autre branche menant vers la destination."""

        else:
            new_src = Graph.include(adj, visited)
            new_visited = visited + [new_src]
            new_real_visited = real_visited + [new_src]
            new_p_tot = p_tot + self.power(src, new_src)
            #print(new_visited,new_real_visited, new_p_tot)
            return(self.get_path_with_power(new_src, dest, p, new_visited, new_real_visited, new_p_tot))

    """
    Min_power 1 
    """

    """
    On peut maintenant écrire la fonction min_power qui va optimiser le trajet allant de src vers dest
    en minimisant la puissance nécessaire pour le trajet. Cette fonction qui prend en argument la source et la destination 
    et renvoie le trajet et la puissance minimale.
    """

    """ 
    Afin d'optimiser le réseau, il nous faut créer une fonction qui renvoie le trajet entre deux noeuds minimisant
    la puissance nécessaire. Pour celà, on utilise le même code que précedemment mais en modifiant quelques lignes,
    en particulier le fait que la fonction s'arretera quand elle aura parcouru tout le graphe.
    """

    def min_power(self, src, dest, p_min=None, visited=None, real_visited=None, p=None, t_min=[]):
        if p_min == None:
            p_min = numpy.inf

        if p == None:
            p = [0]

        if visited == None:
            """ La liste visited contiendra tous les noeuds parcourus durant la récursivité"""
            visited = [src]

        if real_visited == None:
            """ La liste real_visited contiendra seulement les noeuds du chemin qui est en train d'être explorer. 
            La liste est donc inclue dans visited"""
            real_visited = [src]

        adj_set = self.connected_components_set()
        """ On vérifie tout d'abord si un chemin existe entre la source et al destination"""
        for s in adj_set:
            if src in s and dest not in s:
                return (None)

        if src == dest:
            """Lorsque l'on a trouvé un chemin possible, on distingue deux cas : le cas où la puissance du camion est suffisante et le cas
            où ce n'est pas le cas."""
            if p_min > p[-1]:
                p_min = p[-1]
                t_min = real_visited

            else:
                """ Si la puissance du chemin actuel est plus élevée que la puissance maximale du camion, on retourne alors deux noeuds avant 
                afin d'éviter de revenir en boucle sur ce même chemin. On suppose églement que la puissance minimale entre deux noeuds adjacents
                est la puissance liant ces deux noeuds."""
                if len(real_visited) > 2:
                    if self.power(real_visited[-1], real_visited[-2]) == p[-1]:
                        p.pop()
                    if self.power(real_visited[-2], real_visited[-3]):
                        p.pop()
                    real_visited.pop()
                    real_visited.pop()
                    print(real_visited, p)
                    return(self.min_power(real_visited[-1], dest, p_min, visited, real_visited, p, t_min))

                else:
                    return ("Pas assez de puissance")

        """ Si le nouveau noeud n'est pas la destination à atteindre, on regarde alors quels noeuds n'ont pas été parcouru
        afin de continuer la récursivité."""

        adj = self.adj(src)
        if Graph.include(adj, visited) == True:
            if len(real_visited) == 1:
                return (t_min, p_min)
            else:
                if p[-1] == self.power(real_visited[-1], real_visited[-2]):
                    p.pop()
                real_visited.remove(src)
                print('Retour', visited, real_visited, p)
                return(self.min_power(real_visited[-1], dest, p_min, visited, real_visited, p, t_min))

            """ Si tous les neouds adjacents à la source ont été parcouru, alors on revient en arrière sur le chemin afin de 
            trouver une autre branche menant vers la destination."""

        else:
            new_src = Graph.include(adj, visited)
            new_visited = visited + [new_src]
            new_real_visited = real_visited + [new_src]
            if self.power(src, new_src) >= p[-1]:
                p.append(self.power(src, new_src))
            print(new_visited, new_real_visited, p)
            return(self.min_power(new_src, dest, p_min, new_visited, new_real_visited, p, t_min))

    def min_power1(self, i, j, t=[], p=None):
        if p == None:
            p = numpy.inf
        path = self.get_path_with_power(i, j, p)
        if path == None:
            return t, p
        else:
            return self.min_power(i, j, path[0], path[1]-1)

    """
    Min_power 2
    """

    def dijikstra(self, src):
        n = self.nb_nodes
        d = [numpy.inf]*n
        h = [False]*n
        F = Fileprio(0, n)
        f = [False]*n
        prec = [i for i in range(n)]
        Fileprio.enfiler(F, src-1, 0)
        f[src-1] = True
        d[src-1] = 0
        while F.n > 0:
            u = Fileprio.supprimer_min_fp(F)
            l = self.graph[u+1]
            for v in l:
                if (not f[v[0]-1]) and (not h[v[0]-1]):
                    Fileprio.enfiler(F, v[0]-1, d[v[0]-1])
                    f[v[0]-1] = True
                if d[v[0]-1] > max(d[u], v[1]):
                    d[v[0]-1] = max(d[u], v[1])
                    F.poids[v[0]-1] = max(d[u], v[1])
                    prec[v[0]-1] = u
            h[u] = True
        return(prec, d)

    def min_power2(self, src, dest):
        prec, d = Graph.dijikstra(self, src)
        l = [dest]
        u = dest-1
        while u != (src-1):
            u = prec[u]
            l.append(u+1)
        l.reverse()
        return(l, d[dest-1])

    """
    Min_power 3
    """

    """ 
    Nous allons pour  get_path_with_power
    implémenter un algorithme permettant 
    de déterminer la puissance minimale 
    pour tout trajet possible à l'aide d'un algo de Floyd-Warshall
    """

    def edge_i_j(self, i, j):
        l = self.graph[i]
        for k in range(len(l)):
            if l[k][0] == j:
                return(l[k][1])
        return(numpy.inf)

    def matrice_adj(self):
        n = self.nb_nodes
        m = numpy.array([[numpy.inf]*n]*n)
        for i in range(n):
            for j in range(n):
                m[i][j] = Graph.edge_i_j(self, i+1, j+1)
        return(m)

    def matrice_pi(self):
        m = Graph.matrice_adj(self)
        n = len(m)
        pi = numpy.array([[numpy.inf]*n]*n)
        for i in range(n):
            for j in range(n):
                if m[i][j] < numpy.inf:
                    pi[i][j] = j
        return(pi)

    def floydwarshall(self):
        n = self.nb_nodes
        mat = Graph.matrice_adj(self)
        m = mat.copy()
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    m[i][j] = min(max(m[i][k], m[k][j]), m[i][j])
        return(m)

    def plus_court_chemin(self):
        n = self.nb_nodes
        mat = Graph.matrice_adj(self)
        pi = Graph.matrice_pi(self)
        m = mat.copy()
        c = pi.copy()
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if max(m[i][k], m[k][j]) < m[i][j]:
                        c[i][j] = c[i][k]
                    m[i][j] = min(max(m[i][k], m[k][j]), m[i][j])
        return(m, c)

    def chemin_pi(self, i, j):
        m, pi = Graph.plus_court_chemin(self)
        power = m[i-1][j-1]
        l = [i]
        j0 = j-1
        i0 = i-1
        while i0 != j0:
            i0 = int(pi[i0][j0])
            l.append(i0+1)
        return(l, int(power))

    def get_path_with_power2(self, i, j, p):
        l, power = Graph.chemin_pi(self, i, j)
        if p < power:
            return(None)
        else:
            return(l)

    def min_power3(self, i, j):
        return(Graph.chemin_pi(self, i, j))

    """
    Bonus pour la question 5, nous réécrivons les fonctions
    précédentes. Il suffit juste de mondifier la fonction edge
    et de prendre la distance au lieu de la puissance
    """

    def edge_i_j_dist(self, i, j):
        l = self.graph[i]
        for k in range(len(l)):
            if l[k][0] == j:
                return(l[k][2])
        return(numpy.inf)

    def visualisation_graphe(self):
        """
        Draws the graph with the weight and the distance of each edges
        Parameters: 
        -----------
        t: list
        represents the traject to draw
        """
        dot = graphviz.Graph('graphe', comment='Le Graphe')
        g = self.graph
        ar, = []
        for i in self.nodes:
            dot.node(str(i), str(i))
            l = g[i]
            for j in l:
                if ({i, j[0]} not in ar):
                    dot.edge(str(i), str(
                        j[0]), "weight = {} \n distance = {} ".format(j[1], j[2]))
                    ar.append({i, j[0]})
        dot.view()


    def visualisation_graphe_chemin(self, t):
        """
        Draws the graph and highlights the route t with the weight and the distance of each edges

        """
        dot = graphviz.Graph('graphe', comment='Le Graphe')
        g = self.graph
        n = len(t)
        ar, no, c = [], [t[-1]], []
        dot.node(str(t[0]), str(t[0]), color="red", fontcolor="red")
        for i in range(n-1):
            c.append({t[i], t[i+1]})
        for i in self.nodes:
            if (not i in no) and (i in t):
                dot.node(str(t[i]), str(t[i]), color="red", fontcolor="red")
            if not i in no:
                dot.node(str(i), str(i))
            l = g[i]
            for j in l:
                if ({i, j[0]} not in ar) and ({i, j[0]} in c):
                    dot.edge(str(i), str(j[0]), "weight = {} \n distance = {} ".format(
                        j[1], j[2]), color="red", fontcolor="red")
                    ar.append({i, j[0]})
                if ({i, j[0]} not in ar):
                    dot.edge(str(i), str(
                        j[0]), "weight = {} \n distance = {} ".format(j[1], j[2]))
                    ar.append({i, j[0]})
        dot.view()
    
    
    def tree(self, root = None):
        if root == None:
            root = self.nodes[0]
        graph = self.graph
        visited = set()
        parent = {root: None}
        depths = {}
        stack = [root]
        depths[root] = 0
        while stack:
            node = stack.pop()
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor[0] not in visited:
                    parent[neighbor[0]] = (node, self.power(node,neighbor[0]))
                    depths[neighbor[0]] = depths[node] + 1
                    stack.append(neighbor[0])
        return parent, depths


    def commun_ancestor(self, node1, node2, parents = None, depths = None):
        """ Return the minimal power of the truck to have in order to go from node1 to node2
        Parameters :
        -----------
        node1, node2 : node from graph
        parents : dictionary with the parent of each key in a tree predefined
        dephts : dictionary with the depth of each node in a tree predefined
        """
        min_power = 0
        if parents == None :
            parents,depths = self.tree(node1)
        while node1 != node2 :
            if depths[node1] >= depths[node2] :
                power = self.power(node1, parents[node1][0])
                if power > min_power :
                    min_power = power
                node1 = parents[node1][0]
            elif depths[node2] > depths[node1] :
                power = self.power(node2, parents[node2][0])
                if power > min_power :
                    min_power = power
                node2 = parents[node2][0]
        return(node2, min_power)
                
    
def min_power_tree(g,route):
    g = kruskal(g)
    parents, depths = g.tree(g.nodes[0])
    t_tot = 0
    routes = route.graph
    visited = []
    for node in route.nodes :
        for node2 in routes[node]:
            if (node2,node) not in visited : 
                visited.append((node,node2))
                t0 = time.time()
                g.commun_ancestor(routes[node][0][0],node)
                tf = time.time()
                t_tot += (tf-t0)
    return(t_tot)
    


def min_power_union_set(u,src,dest):
    p,_=Union_set.path(u,src - 1, dest - 1)
    return (p)
 

def test(filename1,filename2):
    g = graph_from_file(filename1)
    g,u = kruskal(g)
    m = route_from_file(filename2)
    n = len(m)
    t1 = time.time()
    for i in range(1,n):
        if len(m[i][0])>0 and len(m[i][1])>0:
            src,dest = int((m[i][0])),int((m[i][1]))
            src,dest = src-1,dest-1
            print(min_power_union_set(u,src,dest))
    t2 = time.time()
    return(t2-t1)


        
def include(l1, l2):
    """ Return if the first list is included in the second one i.e all elements of the first list
    is in the second list
    Parameters :
    -----------
    l1, l2 : list
    """
    for val in l1:
        if val not in l2:
            return(val)
    return(True)


def matrice(filename):
    """
    Reads a text file and returns a matrice wich represents each "word" of the file
    Parameters: 
    -----------
    filename: str
        The name of the file
    """
    s = filename.split("\n")
    n = len(s)
    for i in range(n):
        s[i] = s[i].split(" ")
    return(s)


def graph_from_file(filename):
    """
    Reads a text file and returns the graph as an object of the Graph class.
    The file should have the following format: 
        The first line of the file is 'n m'
        The next m lines have 'node1 node2 power_min dist' or 'node1 node2 power_min' (if dist is missing, it will be set to 1 by default)
        The nodes (node1, node2) should be named 1..n
        All values are integers.
    Parameters: 
    -----------
    filename: str
        The name of the file
    Outputs: 
    -----------
    g: Graph
        An object of the class Graph with the graph from file_name.
    """
    s = matrice(filename)
    n, m = int(s[0][0]), int(s[0][1])
    g = Graph([i for i in range(1, n+1)])
    for j in range(1, m+1):
        node1, node2, power_min = int(s[j][0]), int(s[j][1]), int(s[j][2])
        if len(s[j]) == 4:
            dist = int(s[j][3])
        else:
            dist = 1
        Graph.add_edge(g, node1, node2, power_min, dist)
    return(g)

def route_from_file(filename):
    file = matrice(filename)
    routes = []
    n = len(file)
    for i in range(1,n):
        if len(file[i][0])*len(file[i][0]) > 0 :
            routes.append([int(file[i][0]),int(file[i][1]),int(file[i][2])])
    return(routes)
    
    
def truck_from_file(filename):
    file = matrice(filename)
    truck=[]
    n = len(file)
    for i in range(1,n):
        if len(file[i][0])*len(file[i][0]) > 0 :
            truck.append([int(file[i][0]),int(file[i][1])])
    return(truck)


def graph_from_file_route(filename):
    """
    Reads a text file and returns the graph as an object of the Graph class.
    The file should have the following format: 
        The first line of the file is 'n m'
        The next m lines have 'node1 node2 power_min dist' or 'node1 node2 power_min' (if dist is missing, it will be set to 1 by default)
        The nodes (node1, node2) should be named 1..n
        All values are integers.
    Parameters: 
    -----------
    filename: str
        The name of the file
    Outputs: 
    -----------
    g: Graph
        An object of the class Graph with the graph from file_name.
    """
    s = matrice(filename)
    n = int(s[0][0])
    g = Graph([i for i in range(1, n+1)])
    for j in range(1, n+1):
        node1, node2, power_min = int(s[j][0]), int(s[j][1]), int(s[j][2])
        if len(s[j]) == 4:
            dist = int(s[j][3])
        else:
            dist = 1
        Graph.add_edge(g, node1, node2, power_min, dist)
    return(g)

def unionset_graph(u, m):
    """
    Transform a tree implemented thanks to the Union_set Structure into a graph in the Graph class
    Parameters: 
    -----------
    u: Union_set
        The tree
    m: numpy array 
        a matrix which stock the weight of each edge in the graph

    Outputs:
    -----------
    g: Graph
        a graph associate to u
    """
    n = u.n
    l = u.p
    g = Graph([i for i in range(n)])
    for i in range(n):
        Graph.add_edge(g, i, l[i], m[i][l[i]])
    return(g)


    
def kruskal(g):
    """
    Transforms a graph into a covering tree minimum using the Union_set class.
    The complexity of this function is thank to the optimised Union_set operation in O((n+a)log(n))
    where n is the number of nodes and a the number of edges
    Parameters:
    -----------
    g: Graph
    Outputs:
    ---------
    g0: Graph
        Reresents the minimun tree covering the Graph g
    Note: We stocked the edges in a matrice to ease the operations, we could used the adjancy list with the same complexity
    but the function would be less clear. The problem of this function is a complexity in space of O(n^2) in comparaison
    ajancy list would use a space complexity in O(n)
    """
    no = g.nodes
    n = g.nb_nodes
    m0 = g.nb_edges
    l = []
    for el in no:
        adj = g.graph[el]
        for el0 in adj:
            l.append([el0[1],el-1,el0[0]-1])
    l.sort()
    u = Union_set(n)
    g = Graph([i+1 for i in range(n)])
    k,i = 0,0
    while k < n and i < 2*m0 :  
        p,x,y =int(l[i][0]), l[i][1],l[i][2]
        rx,_ = Union_set.rep(u,x)
        ry,_= Union_set.rep(u,y)
        if rx != ry:
            Union_set.fusion(u,x,y,p)
            Graph.add_edge(g,x+1, y+1, p )
            k+= 1
        i+= 1
    return(g,u)


def puissance_mini_kruskal(g, src, dest):
    """
    Returns the min_power of a path (src,dest) using the kruskal algorithm
    The complexity is in O((n+a)log(n)). Calling min_power as a complexity in O(n) because this time the graph as n edges and nodes 

    Parameters: 
    -----------
    g: Graph
    src,dest: int, int
        the path in the Graph
    """
    g,u = kruskal(g)
    p,_= Union_set.path(u, src, dest)
    return(p)
