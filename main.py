import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from matplotlib import colors
import warnings
import ShannonIndex as ShannonIndex
import math
warnings.filterwarnings("ignore", category=DeprecationWarning) 

"ALL NUMBERS HAVE TO BE MULTIPLIED BY 100 to get it to 'real'numbers"

NUMBERSOFTRYS = 5  #Number of times before a family moves even if there is no house that they like

class Family(object):
    
    def __init__(self, s_e_status):
        self.s_e_status = s_e_status
        self.nrTries = 0
        
    def getStatus(self):
        return self.s_e_status
    
    def getTries(self):
        return self.nrTries
    
    def addATry(self):
        self.nrTries += 1
        
    def resetTries(self):
        self.nrTries = 0
        
    def setStatus(self,s_e_status):
        self.s_e_status = s_e_status
        
        
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
        
    def getStatusOfHousehold(self):
        if self.family is None:
            return 0
        else:
            return self.family.getStatus()
        
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
        
    def __call__(self, rounds, Print = True, Homogenity = True):
        
        hScore = np.empty(rounds)
        if Print:        
            for i in range(rounds):
                self.plot_matrix(i, "status")
                hScore[i] = self.calculateHomogenityScore()
                print(hScore[i])
                self.timeStep(self.N, self.p)
                
        else:
            for i in range(rounds):
                hScore[i] = self.calculateHomogenityScore()
                print(hScore[i])
                self.timeStep(self.N, self.p)
            
        self.plot_matrix(i, "income")    
        #plt.plot(np.arange(rounds),hScore)
        
    def fillGrid(self, N):
        "Fill the grid with Households for now only 3 different households exists for test purposes"
        gridint = np.random.randint(0,6,(N,N))
        grid = np.empty((N,N),dtype=object)
        for i in range(N):
            for j in range(N):
                if gridint[i][j] == 1:
                    grid[i][j] = House(150,Family(1))
                elif gridint[i][j] == 2:
                    grid[i][j] = House(200,Family(2))
                elif gridint[i][j] == 3:
                    grid[i][j] = House(300,Family(3))
                elif gridint[i][j] == 4:
                    grid[i][j] = House(400,Family(4))
                elif gridint[i][j] == 5:
                    grid[i][j] = House(500,Family(5))
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
                total += neighbors[i].getStatusOfHousehold()
        if n == 0:
            return total
        return total/n
        

  
  
    #def evaluateNeighborhoodLeaving(self, neighborhoodIncome, house):
    #    if(neighborhoodIncome > 1.25*house.getStatusOfHousehold() or neighborhoodIncome < 0.75*house.getStatusOfHousehold() ):
    #        return True
    #    else:
    #        return False


    #def evaluateNeighborhoodSearching(self, neighborhoodIncome, house):
    #    if(neighborhoodIncome > 1.2*house.getStatusOfHousehold() or neighborhoodIncome < 0.8*house.getStatusOfHousehold() ):
    #        return False
    #    else:
    #        return True

     
     
    def closestEmptyHouse(self, emptyHouses, i, j):

        return emptyHouses[np.argmin(euclidean_distances(emptyHouses, (i,j)))]

    
    def sortByEuclidean(self, emptyHouses,i,j):
        distances = euclidean_distances(emptyHouses,(i,j))
        
        return [x for _,x in sorted(zip(distances, emptyHouses))]
        
    
    def leave(self, i, j, x):
        '''        
        Family at place i,j leaves his house and goes to the closest avaiblable house that he likes.
        '''
        emptyHouses = self.getEmptyHouses()
        emptySorted = self.sortByEuclidean(emptyHouses,i,j)       
        status = self.grid[i][j].getStatusOfHousehold()
        moved = 0
    
        for q in range(len(emptySorted)-1):
            newI, newJ = emptySorted[q]
         
            neighbors = self.getNeighbors(newI, newJ)
            total,difference = self.evaluateNeighbours(neighbors,status) 
            
            if (difference/total < x):
                self.grid[newI][newJ].setFamily(self.grid[i][j].getFam())
                self.grid[i][j].setFamily(None)
                moved = 1
                break
            
        if (moved == 0 and self.grid[i][j].getFam().getTries() == 5):
            newI, newJ = emptySorted[len(emptySorted)-1]   #Maybe house furthest away from own house? so len(emptysortd - 1) or just closest?
            self.grid[newI][newJ].setFamily(self.grid[i][j].getFam())
            self.grid[i][j].setFamily(None)
            
        elif (moved == 0):
            self.grid[i][j].getFam().addATry()
            
         
    def update(self, i, j):
        """
        neighbors = i,j places of the neighbors
        neighborhoodIncome = average income of neighbors
        """ 
        neighbors = self.getNeighbors(i, j)
        total,difference = self.evaluateNeighbours(neighbors,self.grid[i][j].getStatusOfHousehold()) 
        #if self.evaluateNeighborhoodLeaving(neighborhood, neighbors, self.grid[i][j]):
        #    self.leave(i,j)
        x = difference/total
        p = math.exp(x)/(math.exp(x)+1)
        if (np.random.randint(1,1/p + 1)==1):
            self.leave(i,j,x)
     
    def evaluateNeighbours(self,neighbors,status):
        total = 0
        difference = 0
        for i in range(len(neighbors)-1):
            if not neighbors[i].isEmpty():
                total += 4
                difference += abs(neighbors[i].getStatusOfHousehold()-status)
        if total == 0:
            return 1,0
        return total,difference
     
    
    def calculateHomogenityScore(self):
        """
        Calculates the homogenity score based on income of a family.
        SUM(states_j) SUM(Neighbors_i) 1/((|incomeState_j-incomeNeighor_i|)+1)
        Not normalized between 0 and 1 yet
        """
        totalScore = 0
        for i in range(self.N):
            for j in range(self.N):
                for neighbor in self.getNeighbors(i,j): 
                    totalScore += 1/(np.abs(self.grid[i][j].getStatusOfHousehold()-neighbor.getStatusOfHousehold())+1)
        return totalScore
        
     
    def plot_matrix(self, rounds, attribute):
        '''        
        Plots the current state of the grid
        '''
        #cmap = colors.ListedColormap(['white','gray','black'])
        
        attributeGrid = np.zeros((self.N, self.N))
        # moet mooier
        if(attribute.lower() == "income"):
            for i in range(self.N):
                for j in range(self.N):
                    attributeGrid[i][j] = self.grid[i][j].getStatusOfHousehold()
                        
        
        # cmap = colors.ListedColormap(['white', 'blue', 'red'])
        plt.legend()
        plt.title(attribute +" at round: " + str(rounds))
        plt.imshow(attributeGrid, interpolation='nearest')
        plt.tight_layout()
        plt.draw()
        plt.show()
        # plt.savefig(str(rounds) + ".png", dpi = 300)
        plt.pause(0.05)
    
    def peopleLoseJob(self, p):
        for i in range(self.N):
            for j in range(self.N):
                if(np.random.randint(1,1/p + 1)==1) and not self.grid[i][j].isEmpty():
                    self.grid[i][j].getFam().setIncome(self.grid[i][j].getFam().getStatus()*0.5)
                    
                    
                
        
        
grid = Grid(25, 0.4)
grid(25,True, True)
#grid.peopleLoseJob(0.3)
grid(25,True,True)


