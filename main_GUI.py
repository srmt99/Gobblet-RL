import pygame
from pygame.locals import *
import numpy as np
import copy 
import time
import os
import tensorflow as tf
from tensorflow import keras
from game_back import *

model = tf.keras.models.load_model('my_model.h5')

# Initialize pygame
# Solve play sounds latency
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (500,100)
pygame.init()

img_size = (100,100)

# Palette - RGB colors
blue = (78, 140, 243)
light_blue = (100, 100, 255)
red = (242, 89, 97)
light_red = (255, 100, 100)
dark_grey = (85, 85, 85)
light_grey = (100, 100, 100)
background_color = (255, 255, 255)

# Create the window


screen = pygame.display.set_mode((416, 400), RESIZABLE)
pygame.display.set_caption('THE Game of Gobblets')

reds = []
blues = []
temp = []
empty = []
for i in range(4):
    img = pygame.image.load(f"imgs/red{i+1}.png")
    img = pygame.transform.scale(img, img_size)
    temp.append(img.copy())
reds.append(copy.copy(temp))
temp = []
for i in range(4):
    img = pygame.image.load(f"imgs/red{i+1} - Copy.png")
    img = pygame.transform.scale(img, img_size)
    temp.append(img.copy())
reds.append(copy.copy(temp))
temp = []
for i in range(4):
    img = pygame.image.load(f"imgs/blue{i+1}.png")
    img = pygame.transform.scale(img, img_size)
    temp.append(img.copy())
blues.append(copy.copy(temp))
temp = []
for i in range(4):
    img = pygame.image.load(f"imgs/blue{i+1} - Copy.png")
    img = pygame.transform.scale(img,img_size)
    temp.append(img.copy())
blues.append(copy.copy(temp))

empty.append(pygame.transform.scale(pygame.image.load("imgs/empty.png"), img_size))
empty.append(pygame.transform.scale(pygame.image.load("imgs/empty - Copy.png"), img_size))

win_img = pygame.image.load("imgs/win.png").convert_alpha()
lose_img = pygame.image.load("imgs/lose.png").convert_alpha()
##############

# Menu Images
buttom1 = pygame.image.load('Data/Images/button1Img.png')
buttom1_rect = buttom1.get_rect()
buttom1_rect.center = (208, 183)
buttom2 = pygame.image.load('Data/Images/button2Img.png')
buttom2_rect = buttom2.get_rect()
buttom2_rect.center = (208, 303)
logo = pygame.image.load('Data/Images/logo.png')

def run_game():
    running = True
    while running:
        screen.fill(background_color)
        mx, my = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if buttom1_rect.collidepoint((mx, my)):
                    game(0)
                elif buttom2_rect.collidepoint((mx, my)):
                    game(1)
        screen.blit(logo, (8, 8))
        screen.blit(buttom2, (8, 120))
        screen.blit(buttom1, (8, 240))
        pygame.display.update()

def game(gameMode):
    board = board_init()
    screen = pygame.display.set_mode((412, 600), RESIZABLE)
    pl1_deck = build_deck()
    pl2_deck = build_deck()
    pygame.mouse.set_pos(150, 175)
    turn = 1
    # Game loop
    running = True
    selected = None
    hold = False
    filters = win_trails()
    while running:
        # Mouse
        mouse = pygame.mouse.get_pos()
        # Analyzes each game event
        for event in pygame.event.get():
            if event.type == pygame.QUIT: # close the game window if QUIT is pressed
                running = False
            if turn == 2: # player2 (computer) turn
                # board, pl2_deck = next_state(board, pl2_deck, G_CV, model=model, turn=turn, epsilon=0.01)
                board, pl2_deck, _ = search(board, pl1_deck, pl2_deck, G_V1, G_V1, model=model, max_depth=2, turn=turn, use_naive_value_function=False)
                turn = 1
                hold = False
                # print(board)
            elif turn == 1: # player1 turn
                if event.type == pygame.MOUSEBUTTONDOWN: # meaning mouse clicked
                    if selected == None: # first click
                        if 106 < mouse[1] < 506 and 0 < mouse[0] < 400: # click on the board itself:
                            row, col = ((mouse[0]-6)//100 , (mouse[1]-106)//100)
                            # print(board[row*4+col])
                            if (board[row*4+col][3-(board[row*4+col] == 0).sum()]) > 0 and any([max(abs(cell))<board[row*4+col][3-(board[row*4+col] == 0).sum()] for cell in board]):
                                selected = ('b', row*4+col)
                                row, col = ((mouse[0]-6)//100 , (mouse[1]-106)//100)
                                pos = (row * 100+6, col * 100+106)
                                cell_stack = board[row*4 + col]
                                top_piece1 = max(cell_stack)
                                top_piece2 = min(cell_stack)
                                if top_piece1 == top_piece2: # meaning cell is empty
                                    screen.blit(empty[0], pos)
                                elif top_piece1 > abs(top_piece2):
                                    screen.blit(reds[1][4-int(top_piece1)], pos) # meaning pl1 is on top
                                draw_grids()
                                pygame.display.update()
                                hold = True
                        elif 0 < mouse[0] < 350 and 506 < mouse[1] < 600 : # click on the deck
                            col = (mouse[0]-50) // 100
                            if len(pl1_deck[col]) > 0 :
                                pos = (col*100 + 50 , 506)
                                screen.blit(reds[1][4-int(pl1_deck[col][-1])], pos)
                                draw_grids()
                                pygame.display.update()
                                authorized_cells = set()
                                black_white = decode_state(board)
                                for f in filters:
                                    if (black_white[f]==-1).sum()>2: # opponent might be winning
                                        for row in f:
                                            cell = board[row]
                                            if max(abs(cell)) == 0 or (cell[cell != 0][-1] < 0 and abs(cell[cell != 0][-1]) < pl1_deck[col][-1]) :
                                            # meaning the top piece isn't the players and it is not bigger than our piece
                                                authorized_cells.add((row, (cell != 0).sum()))
                                                # print((row, (cell != 0).sum()))
                                if len(authorized_cells) > 0 or any([max(abs(cell))==0 for cell in board]): # there is at least one cell to put the piece
                                    selected = ('d',col)
                                    hold = True
                                else:
                                    hold = False
                    else:
                        if selected[0] == 'b': # last click was on the board
                            if 0 < mouse[0] < 400 and 106 < mouse[1] < 506: # this click on the board itself:
                                row, col = ((mouse[0]-6)//100 , (mouse[1]-106)//100)
                                if max(abs(board[row*4+col])) < board[selected[1]][3-(board[selected[1]] == 0).sum()] :  # we can reposition the piece
                                    board[row*4+col][(board[row*4+col] != 0).sum()] = board[selected[1]][3-(board[selected[1]] == 0).sum()]
                                    board[selected[1]][3-(board[selected[1]] == 0).sum()] = 0
                                    selected = None
                                    hold = False
                                    turn = 2
                        elif selected[0] == 'd': # last click was on the deck
                            if 0 < mouse[0] < 400 and 106 < mouse[1] < 506: # this click on the board itself:
                                row, col = ((mouse[0]-6)//100 , (mouse[1]-106)//100)
                                if max(abs(board[4*row+col]))==0 or (row*4)+col in [x[0] for x in authorized_cells]:
                                    board[row*4+col][(board[row*4+col] != 0).sum()] = pl1_deck[selected[1]].pop()
                                    selected = None
                                    hold = False
                                    turn = 2
        # update the screen accordingly
        if not hold:
            screen.fill(background_color)
            drawBoard(board, pl1_deck, pl2_deck)
            pygame.display.update()
            hold = True
        if terminal(board): # if terminal == True, it means the game is finished
            print("GAME FINISHED")
            rew = evaluate([board])[0]
            time.sleep(2)
            if rew<0:
                screen.blit(lose_img, (7,200))
            else:
                screen.blit(win_img, (7,200))
            pygame.display.update()
            time.sleep(5)
            running = False

def drawBoard(board, pl1_deck, pl2_deck):
    # drawing opponent deck
    for count,g in enumerate(pl2_deck):
        pos = (count*100 + 50 , 6)
        if len(g) == 0 :
            screen.blit(empty[0], pos)
        else:
            top_piece = max(g)
            screen.blit(blues[0][4-int(top_piece)], pos)

    # drawing own deck
    for count,g in enumerate(pl1_deck):
        pos = (count*100 + 50 , 506)
        if len(g) == 0 :
            screen.blit(empty[0], pos)
        else:
            top_piece = max(g)
            screen.blit(reds[0][4-int(top_piece)], pos)

    # Draws each cell
    for row in range(4):
        for col in range(4):
            pos = (row * 100+6, col * 100+106)
            cell_stack = board[row*4 + col]
            top_piece1 = max(cell_stack)
            top_piece2 = min(cell_stack)
            if top_piece1 == top_piece2: # meaning cell is empty
                screen.blit(empty[0], pos)
            elif top_piece1 < abs(top_piece2):
                screen.blit(blues[0][4-int(abs(top_piece2))], pos) # meaning pl2 is on top
            else:
                screen.blit(reds[0][4-int(top_piece1)], pos) # meaning pl1 is on top
    draw_grids()

def draw_grids():
    # Draws the grid
    width = 2
    color = dark_grey
    # vertical lines
    pygame.draw.line(screen, color, (6, 106), (6, 506), width)
    pygame.draw.line(screen, color, (106, 106), (106, 506), width)
    pygame.draw.line(screen, color, (206, 106), (206, 506), width)
    pygame.draw.line(screen, color, (306, 106), (306, 506), width)
    pygame.draw.line(screen, color, (406, 106), (406, 506), width)
    # horizental lines
    pygame.draw.line(screen, color, (6, 106), (406, 106), width)
    pygame.draw.line(screen, color, (6, 206), (406, 206), width)
    pygame.draw.line(screen, color, (6, 306), (406, 306), width)
    pygame.draw.line(screen, color, (6, 406), (406, 406), width)
    pygame.draw.line(screen, color, (6, 506), (406, 506), width)

board = board_init()
run_game()