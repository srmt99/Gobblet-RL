Gobblet-RL (BSc thesis project)
===========
**Learning the game of Gobblet through reinforcement learning**

This project aims to implement an agent that can learn the game of Gobblet and play it at an acceptable level.
**Reinforcement learning** is the method used for learning the game. The project has been implemented in two separate sections.

first section:
-----------------------
The logic of the game and the user interface is implemented in the first section, and the second section is about the intelligent agent who plays the game. Major topics of the first section include the implementation of the **rules** of the game, allowed moves, terminal state recognizer, and the **graphical user interface** for the player to see the game board and interact with it.

you can see a picture of the GUI of the game [here](https://raw.githubusercontent.com/srmt99/Gobblet-RL/main/REPORTS/GUI.PNG)

second section:
-----------------------
Major topics of the second section include the **Monte-Carlo learning**, the training of the agent's **neural network**, searching the game's state space where **MCTS** and **minimax** are used, and predicting future moves in order to play a successful game.

The game of Gobblet is a **two-player zero-sum game**, and thus, an **optimal strategy** for playing the game, corresponding to the **Nash equilibrium**, can be found using the minimax search. Yet due to the large space complexity of the game and its rather large branching factor (~40) it is almost impossible to achieve this solely by minimax search. Thats where **self-play** comes in. In fact, finding this (near) optimal strategy is the goal of the **second** section mentioned above.

An important point to remember is that these two sections are independent of each other, which means that the agent who plays the game has **no idea** about the rules or the standard practices of the game and learns the effective strategies solely based on many plays with itself. In other words, should the first section change (the rules of the game, for instance), the second section need not change significantly.

An picture of the whole process is available [here](https://raw.githubusercontent.com/srmt99/srmt99.github.io/main/data/RL_gobblet.jpg)


