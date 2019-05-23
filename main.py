import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from matplotlib import colors
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


"ALL NUMBERS HAVE TO BE MULTIPLIED BY 100 to get it to 'real'numbers"

NUMBERSOFTRYS = 5  #Number of times before a family moves even if there is no house that they like

class Family(object):
    
    def __init__(self, income, npeople):
        self.income = income
        self.npeople = npeople
        self.nrTries = 0
        
    def getIncome(self):
        return self.income
    
    def getTries(self):
        return self.nrTries
    
    def addATry(self):
        self.nrTries += 1
        
    def resetTries(self):
        self.nrTries = 0
        
class House(object):
    
    def __init__(self, value, family):
        self.value = value
        self.family = family
        
        
    def __str__(self):
        "to string function of the House class"
        return str(self.value)
    
    def isEmpty(self):
        if(self.family is None):
            return True
        else:
            return False
        
    def incomeOfHousehold(self):
        return self.family.getIncome()
        
    def setFamily(self, fam):
        self.family = fam
        if not(fam is None):
            self.family.resetTries()
        
    def getFam(self):
        return self.family
    

        
class Grid(object):
      
    def __init__(self, N, p):
        """
        N grid size (NxN)
        p chance that a family wants to move
        """
        self.grid = self.fillGrid(N)
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
        #self.plot_matrix(i, self.getGrid())    
        
    def fillGrid(self, N):
        "Fill the grid with Households for now only 3 different households exists for test purposes"
        gridint = np.random.randint(0,3,(N,N))
        grid = np.empty((N,N),dtype=object)
        for i in range(N):
            for j in range(N):
                if gridint[i][j] == 1:
                    grid[i][j] = House(200,Family(20,1))
                elif gridint[i][j] == 2:
                    grid[i][j] = House(500,Family(100,1))
                else:
                    grid[i][j] = House(250, None)
        return grid
        
        
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
                if(self.grid[i][j].isEmpty()):
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
                if(np.random.randint(1,1/p + 1)==1) and not self.grid[i][j].isEmpty():
                    self.update(i, j)
                    

    def getNeighborhood(self, neighbors):
        return Counter(neighbors)
        
    def averageIncomeNeighborhood(self, neighbors):
        "Calculate the averageincome of the neighborhood"
        n = 0
        total = 0
        for i in range(len(neighbors)-1):
            if not neighbors[i].isEmpty():
                n += 1
                total += neighbors[i].value
        if n == 0:
            return total
        return total/n
    """   
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
    """

    def evaluateNeighborhoodLeaving(self, neighborhoodIncome, house):
        if(neighborhoodIncome > 1.25*house.incomeOfHousehold() or neighborhoodIncome < 0.75*house.incomeOfHousehold() ):
            return True
        else:
            return False

    """    
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
             """
    def evaluateNeighborhoodSearching(self, neighborhoodIncome, house):
        if(neighborhoodIncome > 1.2*house.incomeOfHousehold() or neighborhoodIncome < 0.8*house.incomeOfHousehold() ):
            return False
        else:
            return True

         
    def closestEmptyHouse(self, emptyHouses, i, j):

        return emptyHouses[np.argmin(euclidean_distances(emptyHouses, (i,j)))]

    
    def sortByEuclidean(self, emptyHouses,i,j):
        distances = euclidean_distances(emptyHouses,(i,j))
        
        return [x for _,x in sorted(zip(distances, emptyHouses))]
        
    
    def leave(self, i, j):
        '''        
        Family at place i,j leaves his house and goes to the closest avaiblable house that he likes.
        '''
        emptyHouses = self.getEmptyHouses()
        emptySorted = self.sortByEuclidean(emptyHouses,i,j)       
        
        moved = 0
        
        for q in range(len(emptySorted)-1):
            newI, newJ = emptySorted[q]
         
            neighbors = self.getNeighbors(newI, newJ)
            neighborhoodIncome = self.averageIncomeNeighborhood(neighbors) 
            
            if (self.evaluateNeighborhoodSearching(neighborhoodIncome,self.grid[i][j])):
                self.grid[newI][newJ].setFamily(self.grid[i][j].getFam())
                self.grid[i][j].setFamily(None)
                moved = 1
                break
            
        if moved == 0 and self.grid[i][j].getFam().getTries() == 5:
            newI, newJ = emptySorted[len(emptySorted)-1]   #Maybe house furthest away from own house? so len(emptysortd - 1) or just closest?
            self.grid[newI][newJ].setFamily(self.grid[i][j].getFam())
            self.grid[i][j].setFamily(None)
            
        else:
            self.grid[i][j].getFam().addATry()
            
         
    def update(self, i, j):
        """
        neighbors = i,j places of the neighbors
        neighborhoodIncome = average income of neighbors
        """ 
        neighbors = self.getNeighbors(i, j)
        neighborhoodIncome = self.averageIncomeNeighborhood(neighbors) 
        #if self.evaluateNeighborhoodLeaving(neighborhood, neighbors, self.grid[i][j]):
        #    self.leave(i,j)
        if self.evaluateNeighborhoodLeaving(neighborhoodIncome, self.grid[i][j]):
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
print(grid.grid[12][12])
#print("before")
#before = grid.getGrid()
#print(before)
#print("after")
grid(10,False)
print(grid.grid[12][12])
#after = grid.getGrid()
#print(after)

