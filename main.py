import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from matplotlib import colors
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

class House(object):
    
    def __init__(self, value, npeople, status):
        self.value = value
        self.npeople = npeople
        self.status = status
        
class Grid(object):
      
    def __init__(self, N, p):
        """
        N grid size (NxN)
        p chance that a family wants to move
        """
        self.grid = np.random.randint(0,3,(N,N))
        self.N = N
        self.p = p
        
    def __call__(self, rounds, Print = True):
        
        if Print:        
            for i in range(rounds):
                self.plot_matrix(i, self.getGrid())
                self.timeStep(self.N, self.p)
        else:
            for i in range(rounds):
                self.timeStep(self.N, self.p)
        self.plot_matrix(i, self.getGrid())    
        
    def getGrid(self):
        return self.grid
        
    def changeGrid(self, i, j, value):
        self.grid[i][j] = value
    
  
    def getEmptyHouses(self):
        """
        Returns the coordinates of all empty houses in the grid
        """
    
        empty = []                
        for i in range(self.N):
            for j in range(self.N):
                if(self.grid[i][j] == 0):
                    empty.append((i, j))
                    
        return empty
                    
    def getNeighbors(self, i, j):
        '''        
        Use the Moore neighbourhood to find the neighbours of a cell and store the class of each neighbour
        '''
        neighbors = []        
        neighbour = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
        
        for shift in neighbour:
            n_row = i + shift[0]
            n_col = j + shift[1]
            if (n_row >= 0 and n_row < self.N) and (n_col >= 0 and n_col < self.N):
                neighbors.append(self.grid[n_row][n_col])
            
        return neighbors
    
    def timeStep(self, N, p):
        '''        
        For each cell, there is a chance p that this cell get's updated.
        '''
        for i in np.random.permutation(np.arange(N)):
            for j in np.random.permutation(np.arange(N)):
                if(np.random.randint(1,1/p + 1)==1) and self.grid[i][j] != 0:
                    self.update(i, j)
                    

    def getNeighborhood(self, neighbors):
        return Counter(neighbors)
        
        
    def evaluateNeighborhoodLeaving(self, neighborhood, neighbors, state):
        '''        
        Used to check if the family should leave the house
        Leave = True
        Stay = False
        If 35% of the neighbors is other state, leave
        '''
        if(neighborhood.get(1) != None):
           if neighborhood.get(1)>0.35*len(neighbors) and state == 2:  
                return True
                
        elif(neighborhood.get(2) != None):
            if neighborhood.get(2)>0.35*len(neighbors) and state == 1:
                return True
        else:
            return False
            
    def evaluateNeighborhoodSearching(self, neighborhood, neighbors, state):
         '''    
         Used to check if the house is a good fit
         MoveIn = True
         StayAway = False
         If 25% of the neighbors is other don't move here
         '''
         if(neighborhood.get(1) != None):
             if neighborhood.get(1)<=0.25*len(neighbors) and state == 1:
                 
                 return True
                 
         elif(neighborhood.get(2) != None):
             if neighborhood.get(2)<=0.25*len(neighbors) and state == 2:
                 return True
         else:
             return False        
         
         
    def closestEmptyHouse(self, emptyHouses, i, j):

        return emptyHouses[np.argmin(euclidean_distances(emptyHouses, (i,j)))]

    
    def sortByEuclidean(self, emptyHouses,i,j):
        distances = euclidean_distances(emptyHouses,(i,j))
        
        return [x for _,x in sorted(zip(distances, emptyHouses))]
        
    
    def leave(self, i, j):
        '''        
        Cell at place i,j leaves his house and goes to the closest avaiblable house that he likes.
        '''
        emptyHouses = self.getEmptyHouses()
        emptySorted = self.sortByEuclidean(emptyHouses,i,j)       

        for q in range(len(emptySorted)-1):
            newI, newJ = emptySorted[q]
         
            neighbors = self.getNeighbors(newI, newJ)
            neighborhood = self.getNeighborhood(neighbors) 
            
            if (self.evaluateNeighborhoodSearching(neighborhood,neighbors,self.grid[i][j])):
                self.changeGrid(newI, newJ, self.grid[i][j])
                self.changeGrid(i,j,0)
                break
            
         
    def update(self, i, j):
        """
        neighbors = i,j places of the neighbors
        neighborhood = classes of neighbors
        """ 
        neighbors = self.getNeighbors(i, j)
        neighborhood = self.getNeighborhood(neighbors) 
        if self.evaluateNeighborhoodLeaving(neighborhood, neighbors, self.grid[i][j]):
            self.leave(i,j)
             
    def plot_matrix(self, rounds, rm):
        '''        
        Plots the current state of the grid
        '''
        #cmap = colors.ListedColormap(['white','gray','black'])
        cmap = colors.ListedColormap(['white', 'blue', 'red'])
        plt.title(rounds)
        plt.imshow(rm, interpolation='nearest', cmap=cmap)
        plt.tight_layout()
        plt.draw()
        plt.show()
        # plt.savefig(str(rounds) + ".png", dpi = 300)
        plt.pause(0.05)
        
        
        
grid = Grid(25, 0.2)
#print("before")
before = grid.getGrid()
#print(before)
#print("after")
grid(100,False)
after = grid.getGrid()
#print(after)

