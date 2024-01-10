import pygame
import random as rand
import math
import numpy as np
import models
import torch
import pathlib




#-------------------------- variables used for the interface------------------------------------------------

SCREEN_WIDTH = 700
SCREEN_HEIGHT = 700
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
GRAY=(200,200,200)

pygame.font.init()
font = pygame.font.SysFont("Arial", 48)
font.set_bold(True)


#window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
#pygame.display.set_caption(("ATTAXX"))

# PROJECT DESCRIPTION
# for the board in the project, RED pieces are defines as the number -1 and BLUE pieces are defined as the number 1 in the board
# the board is represented by a matrix composed of three numbers which corespond to existing a piece in that position or not(-1 or 1 for existing and 0 for not existing)

#--------------------------------------- Importante functions for the game --------------------

class Board:  
    
    def __init__(self, Row_count: int, Collumn_Count: int):
        self.row_count = Row_count
        self.collumn_count = Collumn_Count
        self.BoardMatrix = [[] for i in range(self.row_count)]
        self.player=-1
        self.vict=0
        self.CreateBoard()

#---------------------------------- create board and draw everything on the interface --------------------------

    
    def CreateBoard(self):
        for row in range(self.row_count):
            for collumn in range(self.collumn_count):
                self.BoardMatrix[row].append(0)
        self.BoardMatrix[0][0] = -1
        self.BoardMatrix[self.collumn_count-1][0] = 1
        self.BoardMatrix[0][self.row_count-1] = 1
        self.BoardMatrix[self.collumn_count-1][self.row_count-1] = -1
        
    
 

    #Given a window position converts it to board position
    def PieceAt(self, Position):
        row, col = Position
        piece = self.BoardMatrix[row][col]
        return piece

    #Moves piece to the target location
    def MovePieceTo(self, CurrentPosition, TargetPosition, PieceIndex):
        x1, y1 = CurrentPosition
        x2, y2 = TargetPosition
        self.BoardMatrix[x1][y1] = 0
        self.BoardMatrix[x2][y2] = PieceIndex
        pass

    #Creates a new piece in the target position
    def MakeNewPieceAt(self, TargetPosition, PieceIndex):
        x, y = TargetPosition
        self.BoardMatrix[x][y] = PieceIndex
        pass
    #Calculates the distance between two points
    def GetDistanceBoardUnits(self, StartPosition, EndPosition):
        x1, y1 = StartPosition
        x2, y2 = EndPosition
        distanceX = abs(x2 - x1)
        distanceY = abs(y2 - y1)
        return distanceX, distanceY

    #Capture a piece 
    def CatchPiece(self, Position, PlayerIndex):
        x, y = Position
        RemovablePositions = []
        TangentPositions = [(x+1, y), (x+1, y+1), (x, y+1),
                            (x-1, y+1), (x-1, y), (x-1, y-1), (x, y-1), (x+1, y-1)]
        for p in range(len(TangentPositions)):
            xp, yp = TangentPositions[p]
            if not (xp >= 0 and xp < self.collumn_count and yp >= 0 and yp < self.row_count):
                RemovablePositions.append(TangentPositions[p])
                continue
            if self.BoardMatrix[xp][yp] == 0 or self.BoardMatrix[xp][yp] == PlayerIndex:
                RemovablePositions.append(TangentPositions[p])
                continue
        for i in range(len(RemovablePositions)):
            TangentPositions.remove(RemovablePositions[i])
        for a in range(len(TangentPositions)):
            x, y = TangentPositions[a]
            self.BoardMatrix[x][y] = PlayerIndex

#--------- determine the possible moves for the player and draw them on the interface ---------------------------    
        
#--------------------- check possible moves for the AI  --------------------------------------------
            
    # check all possible moves for AI
    def legal_actions(self,ai_player):
        self.possible=[]
        legal = []
        #print(self.player)
        for row in range(self.row_count):
            for col in range(self.collumn_count):
                #print("oi")
                #print(self.board[row][col])
                if self.BoardMatrix[row][col] == ai_player:
                    # Add legal actions for each piece of the current player
                    legal.extend(self.get_legal_moves_for_piece((row, col)))
            #print("/n")        
        return legal

    # check possible moves for each piece
    def get_legal_moves_for_piece(self, position):
        initial_position = position
        legal_moves = []
        for col in range(max(0, position[0] - 2), min(self.row_count, position[0] + 3)):
            for row in range(max(0, position[1] - 2), min(self.collumn_count, position[1] + 3)):
                if self.BoardMatrix[col][row] == 0:
                    self.possible.append((initial_position, (col, row)))
                    legal_moves.append(col*self.collumn_count+row)
        return legal_moves
    

    
    # check which turn is to use in MCTS
    def to_play(self):
        return 0 if self.player == -1 else 1


#-------------------------- important functions for the muzero -------------------------------------------

    # get the  observations for Muzero
    def get_observation(self):
        
        # Obtain the dimensions of the list
        rows = len(self.BoardMatrix)
        columns = len(self.BoardMatrix[0])
        
        # Create a 3D list with shape (3, rows, columns) and initialize with values from self.BoardMatrix
        observation = [[[0 for _ in range(columns)] for _ in range(rows)] for _ in range(3)]

        # Encode player 1 pieces as 1 in the first channel
        # Encode player 2 pieces as 1 in the second channel
        # Encode the current player's pieces in the third channel
        for i in range(rows):
            for j in range(columns):
                observation[0][i][j] = 1 if self.BoardMatrix[i][j] == 1 else 0
                observation[1][i][j] = 1 if self.BoardMatrix[i][j] == -1 else 0
                observation[2][i][j] = self.player

        return observation
    
    # execute Muzero action 
    def step(self, action,ai_player):
            flag=True
            t=self.legal_actions(ai_player)
            i=0
            
            while i<=len(t)-1 and flag:
                
                if(t[i]==action):
                    index=i
                    flag=False
                i+=1
           
            action=self.possible[index]  # get the coordinates of the start position and end position 
            start_position, end_position = action
            piecevalue = self.PieceAt(end_position)
            distanceX, distanceY = self.GetDistanceBoardUnits(start_position, end_position)
            if (distanceX == 1 and distanceY <= 1) or (distanceY == 1 and distanceX <= 1):
                    if piecevalue == 0:
                        self.MakeNewPieceAt(end_position, ai_player)
                        self.CatchPiece(end_position, ai_player)
                        
            elif (distanceX == 2 and distanceY <= 2) or (distanceY == 2 and distanceX <= 2):
                    if piecevalue == 0:
                        self.MovePieceTo(start_position, end_position, ai_player)
                        self.CatchPiece(end_position, ai_player)
                        
   
    
                  
#----------------------------- Use mcts and game history to find the best action to Muzero execute -----------------------------------
    
