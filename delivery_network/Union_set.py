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
        
    def rep(self,x):
        """
        Return the representant of x in the Union_set
        Parameters: 
        -----------
        x: int, a node

        """
        u = self.p
        i = x 
        while u[i] != i: 
            i = u[i] 
        return(i)
    
    def rang(self,x):
        r0 = Union_set.rep(self,x)
        return(self.r[r0])
    
    def fusion(self,x,y):
        """
        Merges the two parties which contains x and y 
        The complexity is very smoothed almost in O(1) for n reasonable
        Parameters: 
        -----------
        x,y: int, nodes
        """
        rx,ry = Union_set.rep(self,x), Union_set.rep(self,y)
        Rx,Ry = Union_set.rang(self,x), Union_set.rang(self,y)
        if Rx > Ry:
            self.p[ry] = rx
            self.r[rx] +=1 
        else: 
            self.p[rx] = ry
            self.r[ry] +=1 
