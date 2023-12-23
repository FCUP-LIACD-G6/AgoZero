from sys import flags
from tkinter import *
from turtle import TurtleScreen
import numpy as np
import time
from typing import List
from soupsieve import select
from time import sleep
import random
import math
from copy import deepcopy
#import tkinter as tk
#from tkinter.constants import *


# tamanho do tabuleiro e verificação se está dentro dos limites
# nb  --  numero de colunas/linhas
# coords -> (0,0) -> (nb-1,nb-1)
# pixels -> (0,0) -> (599,599)
global tam_quadrado
global symbol_size
global nb
#nb = 7
#tam_quadrado = 600/nb
#symbol_size = (tam_quadrado*0.7-10)/2 *1.5
simb_espe = 2
simb1 = '#0041C2' # Azul    ( OU SEJA 2 )
simb2 = '#00A039' # Verde   ( OU SEJA 1 )
tam_tabu = 600
global turn
turn = 1   #verde = 1 azul = 2
menu = Tk()
menu.title('Ataxx')
canva=Canvas(menu, width=tam_tabu, height=tam_tabu, background="tan")
canva.pack()
fundo = PhotoImage(file= "fundo.png")
logo = PhotoImage(file= "LOGO.png")
global game_ended
game_ended = False
global mapa
global gamemode # 1 = PVSP , 2 = PVSM , 3 = MVSM
global pc1diff # 1 FACIL , 2 MEDIO , 3 DIFICIL
global pc2diff # same
global flagdiff # tecnico

class Ataxx():

    

    def __init__(self):
        menu.unbind('<Button-1>') # reset a qualquer possivel bind
        self.estabelece_tabuleiro() #define os mapas
        self.desenho_tabu() # desenha o tabluleiro e peças 
        menu.bind('<Button-1>', self.click) #click

    def mainloop(self):
        menu.mainloop()

    def estabelece_tabuleiro(self): # estabelece as posicoes do tabuleiro em cada mapa escolhido
        global mapa
        global nb
        global tam_quadrado
        global symbol_size

        if mapa == 1 :
            nb =4
        if mapa == 2:
            nb =5
        if mapa == 3 :
            nb =6

        tam_quadrado = 600/nb
        symbol_size = (tam_quadrado*0.7-10)/2 *1.5
        self.board = np.zeros(shape=(nb, nb))
        self.board[0][0] = 1
        self.board[0][nb-1] = 2
        self.board[nb-1][nb-1] = 1
        self.board[nb-1][0] = 2

    def desenho_tabu(self):    # atraves do tabuleiro estar estabelecido no self.board desenha as pecas dos jogadores e as paredes
        canva.delete("all")
        
        for i in range(nb-1):
            canva.create_line((i+1)*tam_quadrado, 0, (i+1)*tam_quadrado, tam_tabu)
        #for i in range(nb-1):
            canva.create_line(0,(i+1)*tam_quadrado, tam_tabu, (i+1)*tam_quadrado)

        for i in range(nb):
            for j in range(nb):

                if(self.board[i][j]==1): #desenha PEÇA VERDE
                    canva.create_oval(((i+1)*tam_quadrado-tam_quadrado/2) - symbol_size, ((j+1)*tam_quadrado-tam_quadrado/2) - symbol_size,
                                        ((i+1)*tam_quadrado-tam_quadrado/2) + symbol_size, ((j+1)*tam_quadrado-tam_quadrado/2) + symbol_size,
                                        width=simb_espe, outline='#234F1E',
                                        fill=simb2)
                elif(self.board[i][j]==2): # DESENHA PEÇA AZUL
                    canva.create_oval(((i+1)*tam_quadrado-tam_quadrado/2) - symbol_size, ((j+1)*tam_quadrado-tam_quadrado/2) - symbol_size,
                                        ((i+1)*tam_quadrado-tam_quadrado/2) + symbol_size, ((j+1)*tam_quadrado-tam_quadrado/2) + symbol_size, 
                                        width=simb_espe, outline='#151B54',
                                        fill=simb1)
                elif(self.board[i][j]==8): # DESENHA PAREDE
                    canva.create_rectangle((i*tam_quadrado),j*tam_quadrado+tam_quadrado,i*tam_quadrado+tam_quadrado,j*tam_quadrado,fill='black')



    def convert_pixels_to_coords(self, grid_pos):   # recebe a coordenada clicada em pixels e converte-a na posição respetiva do tabuleiro ( 0,0 ) -> (nb-1,nb-1)
        grid_pos = np.array(grid_pos)
        return np.array(grid_pos//(tam_tabu/nb), dtype=int)

    def convert_coords_to_pixels(self,coords_pos): # função inversa da de cima
        coords_pos = np.array(coords_pos, dtype=int)
        return (tam_tabu/nb)*coords_pos + tam_tabu/nb/2

    def pos_is_clear(self, coord): # retorna o valor da condição, (retorna 1 o tabuleiro tem espaço vazio na posição)
        return (self.board[coord[0]][coord[1]] == 0)
    
    def determine_possible_moves(self,coordenadas_clicadas): # recebe a coordenada clicada e retorna as jogadas possiveis
        jogadas_possiveis=[]
        for x in range(max(0,coordenadas_clicadas[0] - 2),min(nb,coordenadas_clicadas[0]+ 3)):
            for y in range(max(0,coordenadas_clicadas[1]-2),min(nb, coordenadas_clicadas[1]+3)):
                if(self.pos_is_clear([x,y])):
                    jogadas_possiveis.append([x,y])
        #print(jogadas_possiveis)            
        return jogadas_possiveis


    def draw_jogadas_possiveis(self, jogadas_possiveis): # recebe lista com jogadas possiveis e desenha a cinzento as jogadas possiveis
        moves=[0]*len(jogadas_possiveis)
        for i in range(len(jogadas_possiveis)):
            moves[i]=self.convert_coords_to_pixels(jogadas_possiveis[i])
            canva.create_oval(moves[i][0]-symbol_size, moves[i][1] - symbol_size,
                                    moves[i][0]+symbol_size, moves[i][1] + symbol_size,
                                    width=simb_espe, outline="gray", fill="gray", tags="possible")
    
    def clear_possiveis(self): # limpa as bolas cinzentas das jogadas possiveis
        canva.delete("possible")


    def jog_AI_facil(self,turn):  #jogada para o nivel facil da maquina / jogada aleatoria 
        c=0
        global coords_pos
        
        
        coords_pos=[0]
        AI_possivel=[]
        for i in range(nb):
            for j in range(nb):
                if(turn==2):
                    if(self.board[i][j]==2):
                        c+=1
                else:
                    if(self.board[i][j]==1):
                        c+=1    
        x=random.randint(1,c)
        c=0
        
        for i in range(nb):
            if(c==x):
                break
            for j in range(nb):
                if(turn==2):
                    if(self.board[i][j]==2):
                        c+=1
                else:
                    if(self.board[i][j]==1):
                        c+=1    
                if(c==x):
                    coords_pos=[i,j]
                    AI_possivel = self.determine_possible_moves(coords_pos)
                    break

        y=random.randint(0,len(AI_possivel)-1)
        self.execute_move(AI_possivel[y])

#....minimax..............................................................................................

    def heuristica(self,nb,board,flag2): #verificar jogada onde pode converter maior numero de pecas adversarias
        pontuacao_verdes=0
        pontuacao_azuis=0
        for i in range(nb):
            for j in range(nb):
                if (board[i][j] == 1):
                    pontuacao_verdes += 1
                if (board[i][j] == 2):
                    pontuacao_azuis += 1
        if(flag2==1):
            return (pontuacao_verdes-pontuacao_azuis)
        return(pontuacao_azuis-pontuacao_verdes)
        
    
    def heuristica2(self,nb,board,flag2): # verifica a jogada para obter maior numero de pecas 
        pontuacao_verdes=0                # independentemente da posicao das pecas adversarias
        pontuacao_azuis=0
        for i in range(nb):
            for j in range(nb):
                if (board[i][j] == 1):
                    pontuacao_verdes += 1
                if (board[i][j] == 2):
                    pontuacao_azuis += 1
        if(flag2 == 1):
            return(pontuacao_verdes)
        return pontuacao_azuis

    def minimax_med(self,board,depth,jogador,alpha,beta,nb,flag2): # minimax para a maquina usado na dificuldade media e dificil 
                                                                    # muda a profundidade sendo depth=2 no medio e depth=3 no dificil
        global game_ended  
        best_move = None                                         #utilizada quando funciona como jogador 1
        if depth==0 or game_ended:  
        
            return self.heuristica(nb,board,flag2),board
        
        if jogador == 1:
            maxEval = -math.inf
            for move in self.movimentos_totais_minimax(board, jogador, nb):
                evaluation = self.minimax_med(move, depth-1, 2, alpha, beta, nb,flag2)[0]
                
                if evaluation > maxEval:
                    
                    best_move = move
                    maxEval = evaluation
                alpha = max(alpha, evaluation)
                if beta <= alpha: 
                    break
                if(np.any(best_move) != True ):
                    best_move=move
            return maxEval, best_move
        else:
            minEval = math.inf
            for move in self.movimentos_totais_minimax(board, 2, nb):
                evaluation = self.minimax_med(move, depth-1, 1, alpha, beta, nb,flag2)[0]
                if evaluation < minEval:
                    minEval = evaluation
                beta = min(beta, evaluation)
                if beta <= alpha:
                    break
            return minEval, beta    


    def minimax_med2(self,board,depth,jogador,alpha,beta,nb,flag2): #minimax com as mesmas carateristicas ao anterior 
                                                                         # utilizada quando é o jogador 2  
        best_move = None
        global game_ended
        if depth==0 or game_ended:  
            
            return self.heuristica(nb,board,flag2),board
       
        if jogador == 2:
            maxEval = -math.inf
            for move in self.movimentos_totais_minimax(board, 2, nb):
                evaluation = self.minimax_med2(move, depth-1, 1, alpha, beta, nb,flag2)[0]
                
                if evaluation > maxEval:
                    
                    best_move = move
                    maxEval = evaluation
                alpha = max(alpha, evaluation)
                if beta <= alpha: 
                    break
                if(np.any(best_move) != True ):
                    best_move=move
            return maxEval, best_move
        else:
            minEval = math.inf
            for move in self.movimentos_totais_minimax(board, 1, nb):
                evaluation = self.minimax_med2(move, depth-1, 2, alpha, beta, nb,flag2)[0]
                if evaluation < minEval:
                    minEval = evaluation
                beta = min(beta, evaluation)
                if beta <= alpha:
                    break
            return minEval, beta    



    def tentativa_move(self,peca,move,board,player,nb):  # tenta simular todos movimento possivel de cada peca de um jogador 
        if player==1:                                   
            outro=2
        else:
            outro=1
        
        if (abs(move[0] - peca[0]) <= 1) and (abs(move[1] - peca[1]) <=1) :
                board[move[0]][move[1]] = player

        
        else:
            board[peca[0]][peca[1]] = 0
            board[move[0]][move[1]] = player   

        for x in range(max(0,move[0]-1),min(nb,move[0]+2)):
            for y in range(max(0,move[1]-1),min(nb,move[1]+2)):
                if(player==1):
                    if board[x][y] == outro:
                        board[x][y] = 1
                else:
                    if board[x][y] == outro:
                        board[x][y] = 2
                    
            
        
        
        return board
                    

    
    def movimentos_totais_minimax(self,board, player,nb): #Determina todos os movimentos possiveis para as peças da maquina
        moves = []
        moves_aval = []
        for peca in self.pecas(board, nb, player):
            moves_aval = self.determine_possible_moves(peca)
            
            for move in moves_aval:
                temp_board = deepcopy(board)
                temp_peca = peca
                # maxi,mini,
                new_board = self.tentativa_move(temp_peca, move, temp_board, player,nb)
                moves.append(new_board)
        return moves

    def pecas(self,board, nb, player): # procura todas as pecas que a maquina tem
        
        pecas = []
        
        for i in range(0,nb):
            for n in range(0,nb):
                if board[i][n] == player:
                    pecas.append((i,n))
        return pecas


    def jogada_minimax(self): #atraves do tabuleiro obtido apos a jogada do minimax desenha as pecas dos jogadores
        for i in range(nb):
            for j in range(nb):
                if(self.board[i][j]==1):
                    self.spawna_verde((i,j))
                if(self.board[i][j]==2):
                    self.spawna_azul((i,j))

                if(self.board[i][j]==0):
                    canva.create_oval(((i+1)*tam_quadrado-tam_quadrado/2) - symbol_size, ((j+1)*tam_quadrado-tam_quadrado/2) - symbol_size,
                                        ((i+1)*tam_quadrado-tam_quadrado/2) + symbol_size, ((j+1)*tam_quadrado-tam_quadrado/2) + symbol_size,
                                        width=7, outline="tan",
                                        fill="tan")



    def click(self, event): # verifica se foi clicada uma peça do turno respetivo, e se sim passa para a próxima "etapa"
        global turn
        global coords_pos
        global jogadas_possiveis
        if(gamemode==1):
            self.clear_possiveis()
            pixels_position = [event.x, event.y]
            coords_pos = self.convert_pixels_to_coords(pixels_position)
            jogadas_possiveis = []
            print(coords_pos)
            if (turn == 1):
                if(self.board[coords_pos[0]][coords_pos[1]] == 1) :
                    jogadas_possiveis = self.determine_possible_moves(coords_pos)
                    self.draw_jogadas_possiveis(jogadas_possiveis)
                    menu.bind('<Button-1>', self.click2)
            if (turn == 2):
                if(self.board[coords_pos[0]][coords_pos[1]] == 2) :
                    jogadas_possiveis = self.determine_possible_moves(coords_pos)
                    self.draw_jogadas_possiveis(jogadas_possiveis)
                    menu.bind('<Button-1>', self.click2)
        elif(gamemode==2):
            self.clear_possiveis()

            pixels_position = [event.x, event.y]
            
            coords_pos = self.convert_pixels_to_coords(pixels_position)
            
            jogadas_possiveis = []
            if (turn == 1):
                if(self.board[coords_pos[0]][coords_pos[1]] == 1) :
                    jogadas_possiveis = self.determine_possible_moves(coords_pos)
                    self.draw_jogadas_possiveis(jogadas_possiveis)
                    menu.bind('<Button-1>', self.click2)
            if(turn==2):
                if(pc1diff==1):
                    self.jog_AI_facil(turn)
                elif(pc1diff==2):
                    a,b=self.minimax_med2(self.board,2,2,-math.inf,math.inf,nb,2)
                    self.board=b
                    self.jogada_minimax()
                else:
                    a,b=self.minimax_med2(self.board,3,2,-math.inf,math.inf,nb,2)
                    self.board=b
                    self.jogada_minimax()
                self.refresh()
                turn=1
                
        elif(gamemode==3):
            if(turn==1):
                if(pc1diff==1):
                    self.jog_AI_facil(turn)
                elif(pc1diff==2):
                    a,b=self.minimax_med(self.board,2,1,-math.inf,math.inf,nb,1)
                    self.board=b
                    self.jogada_minimax()
                elif(pc1diff==3):
                    a,b=self.minimax_med(self.board,3,1,-math.inf,math.inf,nb,1)
                    self.board=b
                    self.jogada_minimax()
                self.refresh() # para evitar bugs
                turn=2
            else:
                if(pc2diff==1):
                    self.jog_AI_facil(turn)
                elif(pc2diff==2):
                    a,b=self.minimax_med2(self.board,2,2,-math.inf,math.inf,nb,2)
                    self.board=b
                    self.jogada_minimax()
                elif(pc2diff==3):
                    a,b=self.minimax_med2(self.board,3,2,-math.inf,math.inf,nb,2)
                    print(b)
                    self.board=b
                    self.jogada_minimax()
                self.refresh() # para evitar bugs
                turn=1
            
            

            



    def click2(self, event): # verifica validade da jogada
        menu.bind('<Button-1>', self.click)
        global jogadas_possiveis
        check = 0
        pixels_position = [event.x, event.y]
        coordenada_clicada = self.convert_pixels_to_coords(pixels_position)
        for i in range(len(jogadas_possiveis)):
            if jogadas_possiveis[i][0] == coordenada_clicada[0] and jogadas_possiveis[i][1] == coordenada_clicada[1] :
                check = 1 # verifica se o click coincidiu com alguma das jogadas possiveis
                break
        if(check == 1) :
            self.execute_move(coordenada_clicada)
            self.clear_possiveis() 
            jogadas_possiveis = []
        if(check == 0): 
            self.clear_possiveis()
            jogadas_possiveis = []
            

    def execute_move(self,coordenada_clicada): #coords_pos é a peça que cliquei, coordenada clicada é a possível
        if (abs(coordenada_clicada[0] - coords_pos[0]) <= 1) and (abs(coordenada_clicada[1] - coords_pos[1]) <=1) :
                self.jogada_adjacente(coordenada_clicada)
        else:
            self.jogada_jump(coordenada_clicada)

    def jogada_adjacente(self, coordenada_clicada): 
        global turn
        if turn == 1 :
            self.set_green(coordenada_clicada)
            self.spawna_verde(coordenada_clicada)
            self.converte(coordenada_clicada)
            turn = 2
        else:
            self.set_blue(coordenada_clicada)
            self.spawna_azul(coordenada_clicada)
            self.converte(coordenada_clicada)
            turn = 1

    def jogada_jump(self,coordenada_clicada):
        global turn
        self.make_it_empty_and_draw_tan()
        if turn == 1 :
            self.set_green(coordenada_clicada)
            self.spawna_verde(coordenada_clicada)
            self.converte(coordenada_clicada)
            turn = 2
        else:
            self.set_blue(coordenada_clicada)
            self.spawna_azul(coordenada_clicada)
            self.converte(coordenada_clicada)
            turn = 1

    def make_it_empty_and_draw_tan(self): # apaga a peça do mapa, quer no desenho quer na matrix
        global coords_pos
        self.board[coords_pos[0]][coords_pos[1]] = 0
        i = coords_pos[0]
        j = coords_pos[1]
        canva.create_oval(((i+1)*tam_quadrado-tam_quadrado/2) - symbol_size, ((j+1)*tam_quadrado-tam_quadrado/2) - symbol_size,
                                        ((i+1)*tam_quadrado-tam_quadrado/2) + symbol_size, ((j+1)*tam_quadrado-tam_quadrado/2) + symbol_size,
                                        width=simb_espe+5, outline="tan",
                                        fill="tan")



        
    def converte(self, coordenada_clicada): # verifica, depois da jogada, as peças a serem convertidas( comidas )
        global turn
        if turn == 1 :
            outro =2
        else :  # se turno for == 2
            outro = 1
        for x in range(max(0,coordenada_clicada[0]-1),min(nb,coordenada_clicada[0]+2)):
            for y in range(max(0,coordenada_clicada[1]-1),min(nb,coordenada_clicada[1]+2)):
                if self.board[x][y] == outro:
                    pos = [x,y]
                    self.board[x][y] = turn
            
                    if turn == 1:
                        self.spawna_verde(pos)
                    else :
                        self.spawna_azul(pos)
        self.refresh()

    def set_blue(self,coordenada): # altera um elemento da matriz
        self.board[coordenada[0]][coordenada[1]] = 2

    def set_green(self,coordenada): # igual, para a outra peça
        self.board[coordenada[0]][coordenada[1]] = 1


    def refresh(self): # para evitar bugs
        global jogadas_possiveis
        jogadas_possiveis = []
        self.clear_possiveis()
        self.score() # por fim, mostra a pontuação

    def score(self): # imprime a pontuação e verifica se o jogo acabou ou não
        global game_ended
        global turn
        flag = 0
        possiveis = []
        pontuacao_verde = 0
        pontuacao_azul = 0
        espacos_vazios = 0

        for i in range(nb):
            for j in range(nb):
                if (self.board[i][j] == 1):
                    pontuacao_verde += 1
                    if(turn == 2): # verifica para o turno oposto porque o "turn" ainda não mudou
                        possiveis = self.determine_possible_moves([i,j])
                        if len(possiveis) != 0 : # isto verifica se há possible moves para o outro turno, se não houver o jogo acaba
                            flag =1
                if (self.board[i][j] == 2):
                    if(turn == 1): # same here
                        possiveis = self.determine_possible_moves([i,j])
                        if len(possiveis) != 0 :
                            flag =1
                    pontuacao_azul += 1
                if (self.board[i][j] == 0):
                    espacos_vazios += 1
                

        print("------------------------------------")
        print("Player 1 ( GREEN ) SCORE : ", pontuacao_verde)
        print("")
        print("Player 2 ( BLUE ) SCORE :  ", pontuacao_azul)
        print("")
        print("(Espaços do tabuleiros vazios : ",espacos_vazios)
        print("------------------------------------")
        if(pontuacao_azul == 0):
            game_ended = True
            print("PLAYER 1 - GREEN WINS")
            greenwin = canva.create_text(300,300,fill="black",font=("Courier New",70),text="GREEN WINS")
        elif(pontuacao_verde == 0):
            game_ended = True
            print("PLAYER 2 - BLUE WINS")
            bluewin = canva.create_text(300,300,fill="black",font=("Courier New",70),text="BLUE WINS")

        elif(espacos_vazios == 0):
            game_ended = True
            if(pontuacao_azul > pontuacao_verde):
                print("PLAYER 2 - BLUE WINS")
                bluewin = canva.create_text(300,300,fill="black",font=("Courier New",70),text="BLUE WINS")
            if(pontuacao_azul < pontuacao_verde):
                print("PLAYER 1 - GREEN WINS")
                greenwin = canva.create_text(300,300,fill="black",font=("Courier New",70),text="GREEN WINS")
            elif(pontuacao_azul == pontuacao_verde):
                print("FANTASTIC!! IT'S A DRAW!")
                draw = canva.create_text(300,300,fill="black",font=("Courier New",100),text="DRAW")

        elif(flag == 0): 
            game_ended = True
            if(pontuacao_azul > pontuacao_verde):
                print("PLAYER 2 - BLUE WINS")
                bluewin = canva.create_text(300,300,fill="black",font=("Courier New",70),text="BLUE WINS")

            if(pontuacao_azul < pontuacao_verde):
                print("PLAYER 1 - GREEN WINS")
                greenwin = canva.create_text(300,300,fill="black",font=("Courier New",70),text="GREEN WINS")

            elif(pontuacao_azul == pontuacao_verde):
                print("FANTASTIC!! IT'S A DRAW!")
                draw = canva.create_text(300,300,fill="black",font=("Courier New",100),text="DRAW")


        if(game_ended == True):
            menu.unbind('all')
            self.board_clear()
            menu.bind('<Button-1>',self.passa) # simplesmente o click final para voltar ao menu inicial

    def passa(self,event):
        global game_ended
        menu.unbind('all')
        game_ended = False
        canva.delete('all')
        global turn
        turn = 1
        main_menu()

        


    def board_clear(self): # limpar o board
        global jogadas_possiveis
        global coords_pos
        global jogadas_possiveis
        coords_pos = None
        jogadas_possiveis = []
        for i in range(nb):
            for j in range(nb):
                self.board[i][j] = 0

    def spawna_azul(self, coordenada_clicada): # na posiçao dada desenha a peça do jogador azul, neste caso jogador 2
        i = coordenada_clicada[0]
        j = coordenada_clicada[1]
        canva.create_oval(((i+1)*tam_quadrado-tam_quadrado/2) - symbol_size, ((j+1)*tam_quadrado-tam_quadrado/2) - symbol_size,
                                        ((i+1)*tam_quadrado-tam_quadrado/2) + symbol_size, ((j+1)*tam_quadrado-tam_quadrado/2) + symbol_size, 
                                        width=simb_espe, outline='#151B54',
                                        fill=simb1)

    def spawna_verde(self, coordenada_clicada): # na posiçao dada desenha a peça do jogador verde, neste caso jogador 1
        i = coordenada_clicada[0]
        j = coordenada_clicada[1]
        canva.create_oval(((i+1)*tam_quadrado-tam_quadrado/2) - symbol_size, ((j+1)*tam_quadrado-tam_quadrado/2) - symbol_size,
                                        ((i+1)*tam_quadrado-tam_quadrado/2) + symbol_size, ((j+1)*tam_quadrado-tam_quadrado/2) + symbol_size,
                                        width=simb_espe, outline='#234F1E',
                                        fill=simb2)

def end__game():
    global game_ended
    menu.unbind('all')
    game_ended = False
    canva.delete('all')
    global turn
    turn = 1
    main_menu()
    


def PVSP(): #executa  o modo de jogo player/player
    menu.unbind('all')
    game = Ataxx()
    game = mainloop()   

def PVSM(): # executa o modo de jogo player/maquina
    menu.unbind('all')
    game = Ataxx()
    game = mainloop()
    

def MVSM(): # executa o modo de jogo maquina/maquina
    menu.unbind('all')
    game = Ataxx()
    game = mainloop()



def map_choose(): # aqui é apresentado os mapas que podem ser escolhidos
    menu.bind('<Button-1>', map_click)
    x = 100
    y = 50
    canva.delete('text1')
    canva.delete('dif1')
    canva.delete('dif2')
    
    choosemap = canva.create_text(300,140,fill='#F4D03F',font=("Courier New",40),text="Choose the Map")
    quatro_quatro = canva.create_text(300,300,fill='#F4D03F',font=("Courier New",40),text="4x4",tag = 'map')
    cinco_cinco = canva.create_text(300,400,fill='#F4D03F',font=("Courier New",40),text="5x5",tag = 'map')
    seis_seis = canva.create_text(300,500,fill='#F4D03F',font=("Courier New",40),text="6x6",tag = 'map')


def map_click(event): # aqui seleciona-se o mapa 
    global mapa
    global gamemode
    print(event.x,event.y)
    x=100

    if(event.x >200 and event.x < 400 and event.y > 160+x and event.y < 240+x) :
        mapa=1


    if(event.x >200 and event.x < 400 and event.y > 280+x and event.y < 340+x) :
        mapa=2
        

    if(event.x >200 and event.x < 400 and event.y > 380+x and event.y < 450+x) :
        mapa=3


    if (gamemode == 1 and mapa != None):
        PVSP()
    elif ( gamemode == 2 and mapa != None):
        PVSM()
    else :
        if (mapa != None):
            MVSM()
        
    

def diff_choose(): # aqui é apresentado as diferentes possibilidades possiveis
    global flagdiff
    global gamemode

    canva.delete('text1')
    machinediff1 = canva.create_text(300,200,fill='#F4D03F',font=("Courier New",32),text="Computer 1 difficulty",tag = 'dif1')
    facil = canva.create_text(300,300,fill='#F4D03F',font=("Courier New",40),text="EASY",tag = 'dif1')
    medio = canva.create_text(300,400,fill='#F4D03F',font=("Courier New",40),text="MEDIUM",tag = 'dif1')
    dificil = canva.create_text(300,500,fill='#F4D03F',font=("Courier New",40),text="HARD",tag = 'dif1')
    menu.bind('<Button-1>',diff_click)
    

def diff_click(event): # click usado para escolher a dificuldade
    global pc1diff
    global pc2diff
    global flagdiff
    x = 100

    if(event.x >200 and event.x < 400 and event.y > 160+x and event.y < 240+x) :
        canva.delete('dif1')
        #print("ez click")
        if(gamemode == 2):
            pc1diff = 1
            menu.unbind('all')
            map_choose()
        elif(flagdiff == 1):
            pc1diff = 1
            flagdiff =2
        elif(flagdiff == 2):
            pc2diff =1
            flagdiff=3

    if(event.x >200 and event.x < 400 and event.y > 280+x and event.y < 340+x) :
        canva.delete('dif1')
        #print("medium")
        if(gamemode == 2):
            pc1diff = 2
            menu.unbind('all')
            map_choose()
        elif(flagdiff == 1):
            pc1diff = 2
            flagdiff =2
        elif(flagdiff == 2):
            pc2diff =2
            flagdiff=3
    if(event.x >200 and event.x < 400 and event.y > 380+x and event.y < 450+x) :
        canva.delete('dif1')
        #print("hardzao")
        if(gamemode == 2):
            pc1diff = 3
            menu.unbind('all')
            map_choose()
        elif(flagdiff == 1):
            pc1diff = 3
            flagdiff =2
        elif(flagdiff == 2):
            pc2diff =3
            flagdiff=3

    if(gamemode==2):
        menu.unbind('all')
        map_choose()
        
    elif(flagdiff==2):
        print("here")
        canva.delete('dif1')
        machinediff2 = canva.create_text(300,200,fill='#F4D03F',font=("Courier New",32),text="Computer 2 difficulty",tag = 'dif2')
        facil2 = canva.create_text(300,300,fill='#F4D03F',font=("Courier New",40),text="EASY",tag = 'dif2')
        medio2 = canva.create_text(300,400,fill='#F4D03F',font=("Courier New",40),text="MEDIUM",tag = 'dif2')
        dificil2 = canva.create_text(300,500,fill='#F4D03F',font=("Courier New",40),text="HARD",tag = 'dif2')
    
    elif(flagdiff == 3):
        menu.unbind('all')
        map_choose()


    
def choice(event): #funçao utilizada no final do jogo para voltar ao menu inicial
    
    global flagdiff
    global gamemode
    global game_ended
    flagdiff = 0
    if(game_ended == True):
        end__game()


    if(event.x >110 and event.x < 500 and event.y > 160 and event.y < 240) :
        gamemode = 1
        map_choose()

    if(event.x >85 and event.x < 530 and event.y > 280 and event.y < 340) :
        gamemode = 2
        diff_choose()

    if(event.x >60 and event.x < 560 and event.y > 380 and event.y < 450) :
        gamemode = 3
        flagdiff = 1
        diff_choose()

 


def main_menu(): # representa o menu principal
    global mapa

    canva.create_image(200,300, image=fundo)
    canva.create_image(300, 60, image=logo)
    text1 = canva.create_text(300,200,fill='#F4D03F',font=("Courier New",30),text="Player vs Player", tag = 'text1')
    text2 = canva.create_text(300,300,fill='#F4D03F',font=("Courier New",30),text="Player vs Computer", tag = 'text1')
    text3 = canva.create_text(300,400,fill='#F4D03F',font=("Courier New",30),text="Treino", tag = 'text1')
    menu.bind('<Button-1>', choice)




    
    print("----------------------------------------------------------------")
    print("BEM VINDO AO ATAXX")
    print("----------------------------------------------------------------")




main_menu()
mainloop()
