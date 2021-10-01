Gobblet-RL (BSc thesis project)
===========
**Learning the game of Gobblet through reinforcement learning**

This project aims to implement an agent that can learn the game of Gobblet and play it at an acceptable level.
**Reinforcement learning** is the method used for learning the game. The project has been implemented in two separate sections.

first section:
-----------------------
The logic of the game and the user interface is implemented in the first section, and the second section is about the intelligent agent who plays the game. Major topics of the first section include the implementation of the **rules** of the game, allowed moves, terminal state recognizer, and the **graphical user interface** for the player to see the game board and interact with it.

second section:
-----------------------
Major topics of the second section include the learning algorithms, the training of the agent's neural network, searching the game's state space, and predicting future moves in order to play a successful game. This section includes the agent's **neural network**, **Monte-Carlo learning**, **MCTS**, and **minimax search**.

The game of Gobblet is a two-player zero-sum game, and thus, an optimal strategy, corresponding to the **Nash equilibrium**, can be found in it. In fact, finding this (near) optimal strategy is the goal of the **second** section mentioned above.

An important point to remember is that these two sections are independent of each other, which means that the agent who plays the game has **no idea** about the rules or the standard practices of the game and learns the effective strategies solely based on many plays with itself. In other words, should the first section change (the rules of the game, for instance), the second section need not change significantly.

An overview of the whole process is shown below:

![alt text](https://github.com/srmt99/srmt99.github.io/blob/main/data/RL_gobblet.jpg?raw=true)
