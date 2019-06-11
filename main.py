import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from matplotlib import colors
import warnings
from ShannonIndex import ShannonIndex
import math
warnings.filterwarnings("ignore", category=DeprecationWarning) 

"ALL NUMBERS HAVE TO BE MULTIPLIED BY 100 to get it to 'real'numbers"

NUMBERSOFTRYS = 5  #Number of times before a family moves even if there is no house that they like
TOTAL_ENTROPY = []

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
            
    def raiseStatus(self):
        if not self.isEmpty():
            status = self.family.getStatus()
            if(status<5):
                self.family.setStatus(status+1)
            
    def lowerStatus(self):
        if not self.isEmpty():
            status = self.family.getStatus()
            if(status>1):
                self.family.setStatus(status-1)
        
    def getFam(self):
        return self.family
    

        
class Grid(object):
      
    def __init__(self, N, p, nrFam):
        """
        N grid size (NxN)
        p chance that a family wants to move
        """
        self.grid = self.fillGrid(N,nrFam)
        self.N = N
        self.p = p
        
    def __call__(self, rounds, Print = True, Homogenity = True):
        
        hScore = np.empty(rounds)
        if Print:        
            for i in range(rounds):
                self.plot_matrix(i, "social_economic_status")
                hScore[i] = self.calculateHomogenityScore()
                print(hScore[i])
                self.timeStep(self.N, self.p)
                
        else:
            for i in range(rounds):
                hScore[i] = self.calculateHomogenityScore()
                print("Round {}: {}".format(i,hScore[i]))
                self.timeStep(self.N, self.p)
            
        self.plot_matrix(i, "social_economic_status")    
        #plt.plot(np.arange(rounds),hScore)
        
    def fillGrid(self, N,nrFam):
        "Fill the grid with Households"
        self.nrFam = nrFam
        
        gridint = np.random.randint(1,nrFam+1,(N,N))
        grid = np.empty((N,N),dtype=object)
        FreeHouses = (round) ((1/6)*N*N)
          
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
                elif gridint[i][j] == 6:
                    grid[i][j] = House(500,Family(6))
                elif gridint[i][j] == 7:
                    grid[i][j] = House(500,Family(7))
                elif gridint[i][j] == 8:
                    grid[i][j] = House(500,Family(8))
                elif gridint[i][j] == 9:
                    grid[i][j] = House(500,Family(9))
                elif gridint[i][j] == 10:
                    grid[i][j] = House(500,Family(10))
        
        for z in range(FreeHouses):
            i = np.random.randint(0,N,FreeHouses)
            j = np.random.randint(0,N,FreeHouses)
            grid[i[z]][j[z]] = House(200,None)
            
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
        for _ in range(N*N):
          
            i = np.random.randint(0,self.N)
            j = np.random.randint(0,self.N)
            if(np.random.rand()<=p) and not self.grid[i][j].isEmpty():
                self.update(i, j)
                    

    #def getNeighborhood(self, neighbors):
    #    return Counter(neighbors)
        
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
    
        for emptyHouse in range(len(emptySorted)-1):
            newI, newJ = emptySorted[emptyHouse]
         
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
        Checks if a family wants to leave based on sigmoid function
        """ 
        neighbors = self.getNeighbors(i, j)
        total,difference = self.evaluateNeighbours(neighbors,self.grid[i][j].getStatusOfHousehold()) 
        #if self.evaluateNeighborhoodLeaving(neighborhood, neighbors, self.grid[i][j]):
        #    self.leave(i,j)
        p = difference/total
        
        
        if (np.random.rand()<=p):
            self.leave(i,j,p)
     
    def evaluateNeighbours(self,neighbors,status):
        """
        returns the total possible difference in social economic status in the neighberhood and the actual difference
        """
        total = 0
        difference = 0
        for i in range(len(neighbors)-1):
            if not neighbors[i].isEmpty():
                total += self.nrFam-1
                diff = abs(neighbors[i].getStatusOfHousehold()-status) 
                difference += self.getCostOfNeighbour(diff)
        if total == 0:
            return 1,0
        return total,difference
     
        
    def getCostOfNeighbour(self,diff):
        if diff == 1:
            return 1
        elif diff == 2:
           return 2
        elif diff == 3:
            return 3
        elif diff == 4:
            return 4
        elif diff == 5:
            return 5
        elif diff == 6:
            return 6
        elif diff == 7:
            return 7
        elif diff == 8:
            return 8
        elif diff == 9:
            return 9
        else:
            return 0
    
    def burglary(self, i, j):
        '''
        Burglary in house at location i,j. The resident moves to furthest free house.
        Neighbors decrease in social economic status by -1.
        '''
        neighbors = self.getNeighbors(i,j)
        for neighbor in neighbors:
            neighbor.lowerStatus()
        emptyHouses = self.getEmptyHouses()
        emptySorted = self.sortByEuclidean(emptyHouses,i,j)  
    
        newI, newJ = emptySorted[len(emptySorted)-1]   #Maybe house furthest away from own house? so len(emptysortd - 1) or just closest?
        self.grid[newI][newJ].setFamily(self.grid[i][j].getFam())
        self.grid[i][j].setFamily(None)     
    
    
    def promotion(self, i,j):
        self.grid[i][j].raiseStatus()


    def goGreen(self, i,j):
        '''
        Goverment 'Go green!' policy on a Moore neighborhood. All residents have
        their status increased by 1.
        '''
        neighbors = self.getNeighbors(i,j)
        self.grid[i][j].raiseStatus()
        for neighbor in neighbors:
            neighbor.raiseStatus()
    
    def calculateHomogenityScore(self):
        """
        Calculates the homogenity score based on income of a family.
        SUM(states_j) SUM(Neighbors_i) 1/((|incomeState_j-incomeNeighor_i|)+1)
        Not normalized between 0 and 1 yet
        """
        totalScore = 0
        for i in range(self.N):
            for j in range(self.N):
                neighbors = self.getNeighbors(i,j)
                neighborhood = []
                for neighbor in neighbors:
                    neighborhood.append(neighbor.getStatusOfHousehold())
                
                totalScore += ShannonIndex(neighborhood)
                #totalScore += 1/(np.abs(self.grid[i][j].getStatusOfHousehold()-neighbor.getStatusOfHousehold())+1)
         
        TOTAL_ENTROPY.append(totalScore) 
        return totalScore
        
     
    def plot_matrix(self, rounds, attribute):
        '''        
        Plots the current state of the grid
        '''
        #cmap = colors.ListedColormap(['white','gray','black'])
        
        attributeGrid = np.zeros((self.N, self.N))
        # moet mooier
        if(attribute.lower() == "social_economic_status"):
            for i in range(self.N):
                for j in range(self.N):
                    attributeGrid[i][j] = self.grid[i][j].getStatusOfHousehold()
                        
        
        # cmap = colors.ListedColormap(['white', 'blue', 'red'])
        plt.legend()
        #plt.title(attribute +" at round: " + str(rounds))
        plt.imshow(attributeGrid, interpolation='nearest')
        plt.tight_layout()
        plt.draw()
        plt.show()
        
        plt.pause(0.05)
    
               
    def createHeatMap(self):
        heatMap = []
        for i in range(self.N):
            for j in range(self.N):
                neighbors = self.getNeighbors(i,j)
                neighborhood = []
                for neighbor in neighbors:
                    neighborhood.append(neighbor.getStatusOfHousehold())
                    
                             
                heatMap.append(ShannonIndex(neighborhood))
                
 
        plt.imshow(np.array(heatMap).reshape((self.N, self.N)))
        plt.colorbar()        
        plt.title("Heatmap of entropy")
        plt.show()
        
    def plotTotalEntropy(self):
        plt.xlabel('Steps')
        plt.ylabel('Total Entropy')
        plt.title("Total Entropy over time")
        plt.plot(TOTAL_ENTROPY)
        plt.show()
        
    
'''
Defaults:
Grid Size = 25
Categories = 5
Probability = 0.3
#Epochs = 150
Costs = 1:1 2:2 3:3 4:4
'''      

plt.clf()
grid = Grid(25, 0.3, 5)
grid(100, False, True) 
plt.savefig('exp8_2_100Grid.png', dpi=600, bbox_inches='tight')
input("Press Enter to continue...")
plt.clf()
grid.plotTotalEntropy()
plt.savefig('exp8_2_100_Entropy.png', dpi=600, bbox_inches='tight')
input("Press Enter to continue...")
plt.clf()
grid.createHeatMap()
plt.savefig('exp8_2_100_Heatmap.png', dpi=600, bbox_inches='tight')
plt.clf()

grid(29, False, True) 
grid.burglary(10,10)
grid.burglary(13,13)
grid.burglary(8,8)

grid(1,False,True)
plt.savefig('exp8_2_130Grid.png', dpi=600, bbox_inches='tight')
input("Press Enter to continue...")
plt.clf()
grid.plotTotalEntropy()
plt.savefig('exp8_2_130_Entropy.png', dpi=600, bbox_inches='tight')
input("Press Enter to continue...")
plt.clf()
grid.createHeatMap()
plt.savefig('exp8_2_130_Heatmap.png', dpi=600, bbox_inches='tight')
plt.clf()

grid(20, False, True) 
plt.savefig('exp8_2_150Grid.png', dpi=600, bbox_inches='tight')
input("Press Enter to continue...")
plt.clf()
grid.plotTotalEntropy()
plt.savefig('exp8_2_150_Entropy.png', dpi=600, bbox_inches='tight')
input("Press Enter to continue...")
plt.clf()
grid.createHeatMap()
plt.savefig('exp8_2_150_Heatmap.png', dpi=600, bbox_inches='tight')
plt.clf()


'''
plt.savefig('exp7_6_Grid.png', dpi=600, bbox_inches='tight')
input("Press Enter to continue...")
plt.clf()
grid.plotTotalEntropy()
plt.savefig('exp7_6_Entropy.png', dpi=600, bbox_inches='tight')
input("Press Enter to continue...")
plt.clf()
grid.createHeatMap()
plt.savefig('exp7_6_Heatmap.png', dpi=600, bbox_inches='tight')
plt.clf()

#grid.goGreen(13,13)
#grid.goGreen(2,2)
#grid.goGreen(18,18)
#grid.goGreen(13,4)
#grid.goGreen(5,18)
#grid.goGreen(1,19)
#grid.plotTotalEntropy()
'''

'''
grid(25, False, True) 
input("Press Enter to continue...")
grid.plotTotalEntropy()


grid.createHeatMap()
#IETS GEBEUREN 
#grid(25,False,True)


'''