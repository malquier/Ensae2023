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
        for i in range(1,n+1): 
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
    
    def adj(self,i):
        graph = copy.deepcopy(self.graph)
        adj = []
        for noeud in graph[i]:
            adj.append(noeud[0])
        return adj
    
    def power(self,i,j):
        graph = self.graph
        for noeud in graph[i]:
            if noeud[0] == j:
                return(noeud[1])
    
    def include(l1,l2):
        for val in l1:
            if val not in l2:
                return(val)
        return(True)

    def get_path_with_power(self, src, dest, p, visited = None, real_visited = None, p_tot = 0):
        if visited == None :
            """ La liste visited contiendra tous les noeuds parcourus durant la récursivité"""
            visited = [src]
            
        if real_visited == None :
            """ La liste real_visited contiendra seulement les noeuds du chemin qui est en train d'être explorer. 
            La liste est donc inclue dans visited"""
            real_visited = [src]

        adj_set = self.connected_components_set()
        """ On vérifie tout d'abord si un chemin existe entre la source et al destination"""
        for s in adj_set :
            if src in s and dest not in s :
                return (None)
        
        if src == dest:
            """Lorsque l'on a trouvé un chemin possible, on distingue deux cas : le cas où la puissance du camion est suffisante et le cas
            où ce n'est pas le cas."""
            if p>=p_tot:
                return(real_visited, p_tot)
            
            else :
                """ Si la puissance du chemin actuel est plus élevée que la puissance maximale du camion, on retourne alors deux noeuds avant 
                afin d'éviter de revenir en boucle sur ce même chemin. On suppose églement que la puissance minimale entre deux noeuds adjacents
                est la puissance liant ces deux noeuds."""
                if len(real_visited)>2:
                    new_p_tot = p_tot - self.power(real_visited[-1],real_visited[-2]) - self.power(real_visited[-2],real_visited[-3])
                    real_visited.pop()
                    real_visited.pop()
                    #print(real_visited, new_p_tot)
                    return(self.get_path_with_power(real_visited[-1], dest, p, visited, real_visited, new_p_tot))
                
                else :
                    return ("Pas assez de puissance")
        
        """ Si le nouveau noeud n'est pas la destination à atteindre, on regarde alors quels noeuds n'ont pas été parcouru
        afin de continuer la récursivité."""
        
        adj = self.adj(src)
        if Graph.include(adj,visited) == True:
            if len(real_visited) == 1:
                return (None)
            else :
                real_visited.remove(src)
                new_p_tot = p_tot - self.power(src,real_visited[-1])
                #print('Retour',visited, real_visited, new_p_tot)
                return( self.get_path_with_power(real_visited[-1], dest, p, visited, real_visited, new_p_tot))
        
            """ Si tous les neouds adjacents à la source ont été parcouru, alors on revient en arrière sur le chemin afin de 
            trouver une autre branche menant vers la destination."""
        
        else :
            new_src = Graph.include(adj,visited)
            new_visited = visited + [new_src]
            new_real_visited = real_visited + [new_src]
            new_p_tot = p_tot + self.power(src,new_src)
            #print(new_visited,new_real_visited, new_p_tot)
            return( self.get_path_with_power(new_src, dest, p, new_visited, new_real_visited, new_p_tot))
        
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
    
    def min_power(self, src, dest, p_min = None, visited = None, real_visited = None, p = None, t_min = []):
        if p_min == None :
            p_min = numpy.inf
            
        if p == None :
            p=[0]
            
        if visited == None :
            """ La liste visited contiendra tous les noeuds parcourus durant la récursivité"""
            visited = [src]
            
        if real_visited == None :
            """ La liste real_visited contiendra seulement les noeuds du chemin qui est en train d'être explorer. 
            La liste est donc inclue dans visited"""
            real_visited = [src]
            
        adj_set = self.connected_components_set()
        """ On vérifie tout d'abord si un chemin existe entre la source et al destination"""
        for s in adj_set :
            if src in s and dest not in s :
                return (None)
            
        if src == dest:
            """Lorsque l'on a trouvé un chemin possible, on distingue deux cas : le cas où la puissance du camion est suffisante et le cas
            où ce n'est pas le cas."""
            if p_min>p[-1]:
                p_min = p[-1]
                t_min = real_visited
            
            else :
                """ Si la puissance du chemin actuel est plus élevée que la puissance maximale du camion, on retourne alors deux noeuds avant 
                afin d'éviter de revenir en boucle sur ce même chemin. On suppose églement que la puissance minimale entre deux noeuds adjacents
                est la puissance liant ces deux noeuds."""
                if len(real_visited)>2:
                    if self.power(real_visited[-1], real_visited[-2]) == p[-1]:
                        p.pop()
                    if self.power(real_visited[-2], real_visited[-3]):
                        p.pop()
                    real_visited.pop()
                    real_visited.pop()
                    print(real_visited, p)
                    return(self.min_power(real_visited[-1], dest, p_min, visited, real_visited, p, t_min))
                
                else :
                    return ("Pas assez de puissance")
        
        """ Si le nouveau noeud n'est pas la destination à atteindre, on regarde alors quels noeuds n'ont pas été parcouru
        afin de continuer la récursivité."""
        
        adj = self.adj(src)
        if Graph.include(adj,visited) == True:
            if len(real_visited) == 1:
                return (t_min, p_min)
            else :
                if p[-1] == self.power(real_visited[-1],real_visited[-2]):
                    p.pop()
                real_visited.remove(src)
                print('Retour',visited, real_visited, p)
                return( self.min_power(real_visited[-1], dest, p_min, visited, real_visited, p, t_min))
        
            """ Si tous les neouds adjacents à la source ont été parcouru, alors on revient en arrière sur le chemin afin de 
            trouver une autre branche menant vers la destination."""
        
        else :
            new_src = Graph.include(adj,visited)
            new_visited = visited + [new_src]
            new_real_visited = real_visited + [new_src]
            if self.power(src,new_src) >= p[-1]:
                p.append(self.power(src,new_src))
            print(new_visited,new_real_visited, p)
            return( self.min_power(new_src, dest, p_min, new_visited, new_real_visited, p, t_min))
    
    
    def min_power1(self, i, j, t=[], p = None):
        if p == None :
            p = numpy.inf
        path = self.get_path_with_power(i, j, p)
        if path == None:
            return t, p
        else :
            return self.min_power(i, j, path[0], path[1]-1)
        
    """
    Min_power 2
    """
    def dijikstra(self,src):
        n = self.nb_nodes
        d = [numpy.inf]*n
        h = [False]*n
        F = Fileprio(0,n)
        f = [False]*n
        prec = [i for i in range(n)]
        Fileprio.enfiler(F,src-1,0)
        f[src-1] = True 
        d[src-1] = 0 
        while F.n > 0:
            u = Fileprio.supprimer_min_fp(F)
            l = self.graph[u+1]
            for v in l: 
                if (not f[v[0]-1]) and (not h[v[0]-1]):
                    Fileprio.enfiler(F,v[0]-1,d[v[0]-1])
                    f[v[0]-1] = True
                if d[v[0]-1] > max(d[u],v[1]):
                    d[v[0]-1] = max(d[u],v[1])
                    F.poids[v[0]-1] = max(d[u],v[1])
                    prec[v[0]-1] = u
            h[u] = True
        return(prec,d)
    
    def min_power2(self,src,dest):
        prec,d = Graph.dijikstra(self,src)
        l = [dest]
        u = dest-1
        while u != (src-1):
            u = prec[u]
            l.append(u+1)
        l.reverse()
        return(l,d[dest-1])
    
    """
    Min_power 3
    """
   
    """ 
    Nous allons pour  get_path_with_power
    implémenter un algorithme permettant 
    de déterminer la puissance minimale 
    pour tout trajet possible à l'aide d'un algo de Floyd-Warshall
    """
    
    def edge_i_j(self,i,j):
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
                m[i][j] = Graph.edge_i_j(self,i+1,j+1)
        return(m)
    
    def matrice_pi(self): 
        m = Graph.matrice_adj(self)
        n = len(m) 
        pi = numpy.array([[numpy.inf]*n]*n )
        for i in range(n): 
            for j in range(n): 
                if m[i][j] < numpy.inf : 
                    pi[i][j] = j
        return(pi)
    
    def floydwarshall(self):
        n = self.nb_nodes
        mat = Graph.matrice_adj(self)
        m = mat.copy()
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    m[i][j] = min(max(m[i][k],m[k][j]),m[i][j])
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
                        if max(m[i][k],m[k][j]) < m[i][j]:
                            c[i][j] = c[i][k]
                        m[i][j] = min(max(m[i][k],m[k][j]),m[i][j])
        return(m,c)            

    def chemin_pi(self,i,j):
        m,pi = Graph.plus_court_chemin(self)
        power = m[i-1][j-1]
        l = [i]
        j0 = j-1
        i0 = i-1
        while i0 != j0: 
            i0 = int(pi[i0][j0])
            l.append(i0+1)
        return(l,int(power))
    
    def get_path_with_power2(self,i,j,p):
        l,power = Graph.chemin_pi(self,i,j)
        if p < power: 
            return(None)
        else: 
            return(l)
    
    def min_power3(self,i,j):
        return(Graph.chemin_pi(self,i,j))
    
    """
    Bonus pour la question 5, nous réécrivons les fonctions
    précédentes. Il suffit juste de mondifier la fonction edge
    et de prendre la distance au lieu de la puissance
    """

    def edge_i_j_dist(self,i,j):
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
                    dot.edge(str(i), str(j[0]), "weight = {} \n distance = {} ".format(j[1], j[2]))
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
                    dot.edge(str(i), str(j[0]), "weight = {} \n distance = {} ".format(j[1], j[2]))
                    ar.append({i, j[0]})
        dot.view()
