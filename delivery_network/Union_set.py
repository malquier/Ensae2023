class Union_set:
    """
    A class representing trees as an adjacency lists and a list wich represent the depth of each nodes implementing various algorithms on the trees.
    Trees in the class are not oriented.
    Attributes:
    -----------
    n: int
       The number of nodes.
    p: list
        The adjancy list which represents the trees
    r: list
        A list which represent the depth of each nodes
    """
    def __init__(self,n=1):
        """
        Initializes the union_set with a number of nodes, and no edges.
        Parameters:
        -----------
        n: int, optional
        """
        self.n = n
        self.p = [i for i in range(n)]
        self.r = [1 for i in range(n)]
        self.pi =[0 for i in range(n)]
       
    def rep(self,x):
        """
        Return the representant of x in the Union_set
        Parameters:
        -----------
        x: int, a node

        """
        u = self.p
        pi = self.pi
        i = x
        po = pi[x]
        while u[i]!= i:
            i = u[i]
            if pi[i]!= 0:
                po = max(pi[i],po)
        return(i,po)

    def rang(self,x):
        r0,_ = Union_set.rep(self,x)
        return(self.r[r0])
   
    def fusion(self,x,y,p):
        """
        Merges the two parties which contains x and y
        The complexity is very smoothed almost in O(1) for n reasonable
        Parameters:
        -----------
        x,y: int, nodes
        """
        rx,_ = Union_set.rep(self,x)
        ry,_= Union_set.rep(self,y)
        Rx,Ry = Union_set.rang(self,x), Union_set.rang(self,y)
        if Rx > Ry:
            self.p[ry] = rx
            self.pi[ry] = p
        elif Rx < Ry:
            self.p[rx] = ry
            self.pi[rx] = p
        else:
            self.p[rx] = ry
            self.pi[rx] = p
            self.r[ry] +=1
     
    def path_racine(self,x):
        u = self.p
        pi = self.pi
        i = x
        po = pi[x]
        l = [(x,po)]
        while u[i]!= i:
            i = u[i]
            if pi[i]!= 0:
                po = max(pi[i],po)
            l.append((i,po))
        return(l)
   
    def path(self,x,y):
        l1 = Union_set.path_racine(self,x)
        l1.reverse()
        l2 = Union_set.path_racine(self,y)
        l2.reverse()
        n = min(len(l1),len(l2))
        i = 0
        while i<n and l1[i][0] == l2[i][0]:
            i+=1
        i -= 1
        p = max(l1[i][1],l2[i][1])
        l0 = max(l1[i],l2[i])
        l= l1[i+1:]
        l.reverse()
        l.append(l0)
        return(p,l +l2[i+1:])
