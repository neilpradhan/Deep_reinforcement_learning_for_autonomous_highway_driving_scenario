# Deep_Reinforcement_Learning_for_Autonomous_Highway_Driving_Scenario
This repository contains the code of my master thesis

## Table of contents
* [Abstract](#general-info)
* [Technologies](#technologies)
* [Description](#Description)

## Abstract:
We present an autonomous driving agent on a simulated highway driving scenario with vehicles such as cars and trucks moving with stochastically variable velocity profiles. The focus of the simulated environment is to test tactical decision making in highway driving scenarios. When an agent (vehicle) maintains an optimal range of velocity it is beneficial both in terms of energy efficiency and greener environment. In order to maintain an optimal range of velocity, in this thesis work I proposed two novel reward structures: (a) gaussian reward structure and (b) exponential rise and fall reward structure. I trained respectively two deep reinforcement learning agents to study their differences and evaluate their performance based on a set of parameters that are most relevant in highway driving scenarios. The algorithm implemented in this thesis work is double-dueling deep-Q-network with prioritized experience replay buffer. Experiments were performed by adding noise to the inputs, simulating Partially Observable Markov Decision Process in order to obtain reliability comparison between different reward structures. Velocity occupancy grid was found to be better than binary occupancy grid as input for the algorithm. Furthermore, methodology for generating fuel efficient policies has been discussed and demonstrated with an example. 

To see the complete report: [Thesis report](http://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-289444)
	
## Technologies
Project is created with:
* Python 3
* Scipy
* Pandas
* Tensorflow
* Open cv
* Numpy
* Matplotlib
* Math
	
## Description
Raw data is collected from the simulator as the agent covers the highway of 50,000 units in dense traffic. The simular functions and limitations have been discussed in the [thesis report](http://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-289444)


Data Analytics is done on the raw data obtained from the simulator and exponential weighted moving average is performed and differences between the training performance of two reward structures are shown , and results of which are in the Inference_from_training.ipynb file and the results of key parameters plotted for the two reward structures for increasing amount of bias and variance are shown in the Thesis_Testing_plots.ipynb file



Inference_from_training.ipynb
Thesis_Testing_plots.ipynb

