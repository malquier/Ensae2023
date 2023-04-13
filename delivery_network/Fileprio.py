class Fileprio:
    """
    A class representing priority Queue as a list
    This is very useful because all the basis operation that we programmed: "monter noeud",
    " descendre_noeud", "enfiler" and "supprimer_min_fp" have a complexity in 0(log(n)) 
    where n is the number of nodes in the Queue. 
    This class will be uselful to optimize min_power
    Attributes: 
    -----------
    n: int
        The number of nodes in the Queue
    tab: list
        A list of adjancy which represents the tree associate with the Queue
    poids: list
        A list which represent the priority of each node
        In our function the priority of each node will be the min_power to go to the node from a source 
    """

    def __init__(self, n=0, m=1):
        """
        Initializes the Fileprio
        Parameters: 
        -----------
        n: int
            Represents the length of the Fileprio
        m : int
            Represents the capacity of the Fileprio
        """
        self.n = n
        self.tab = [i for i in range(m)]
        self.poids = [0 for i in range(m)]

    """
    We have implemented some basic functions to manipulate a Fileprio more easily:

    fg and fd: gives the son of the node i in the tree
    pere: gives the father of the node i in the tree
    echanger: exchange to node in the Fileprio

    """

    def fg(i):
        return(2*i + 1)

    def fd(i):
        return(2*i + 2)

    def pere(i):
        return((i-1)//2)

    def echanger(self, i, j):
        self.poids[self.tab[i]], self.poids[self.tab[j]
                                            ] = self.poids[self.tab[j]], self.poids[self.tab[j]]
        self.tab[i], self.tab[j] = self.tab[j], self.tab[i]

    def monter_noeud(self, i):
        """
        Rises a node in the Fileprio in keeping the structure of Fileprio_min
        Parameters: 
        -----------
        i: int, the node to increase
        """
        i0 = Fileprio.pere(i)
        t, p = self.tab, self.poids
        if i != 0 and p[t[i0]] > p[t[i]]:
            Fileprio.echanger(self, i, i0)
            Fileprio.monter_noeud(self, i0)

    def descendre_noeud(self, i):
        """
        Takes down the node i in the Fileprio
        Parameters: 
        -----------
        i: int, the node to take down 
        """
        j = i
        n = self.n
        t, p = self.tab, self.poids
        g, d = Fileprio.fg(i), Fileprio.fd(i)
        if g < n and p[t[g]] > p[t[i]]:
            j = g
        if d < n and p[t[d]] > p[t[j]]:
            j = d
        if i != j:
            Fileprio.echanger(self, i, j)
            Fileprio.descendre_noeud(self, j)

    def enfiler(self, i, p):
        """
        Adds a node i whith the prio p in an File_prio
        Parameters: 
        -----------
        i,p: int,int 
            Represent the node i and the priority p
        """
        n = self.n
        self.tab[n] = i
        self.poids[i] = p
        self.n = n + 1
        Fileprio.monter_noeud(self, n)

    def supprimer_min_fp(self):
        """
        Delete the min of the FilePrio and send it back in a O(log(n)) complexity
        """
        self.n -= 1
        Fileprio.echanger(self, 0, self.n)
        Fileprio.descendre_noeud(self, 0)
        return(self.tab[self.n])
