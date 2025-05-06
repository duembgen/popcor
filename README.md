# POPR: Polynomial OPtimization for Robotics

A toolbox for setting up, solving and certifying polynomial optimization problems in robotics. 

## Purpose

This toolbox includes methods to formulate and solve polynomial optimization problems in robotics. 
The focus of this toolbox is on creating **tight** semidefinite relaxations, which means that we can
replace difficult (often NP-hard) optimization problems by easier-to-solve convex problems, which
in many cases can actually provide the globally optimal solution of the original problem. 

The toolbox allows to run the *AutoTight* and *AutoTemplate* algorithms on problems of your choice. 
These algorithms are described in detail in [this paper](https://arxiv.org/abs/2308.05783/) [1].
For a high-level overview of how this works, please refer to the **Quick start guide**. 

## Dependencies

Besides the automatically installed dependencies when using the above instructions, you need to also have a valid MOSEK license in order to use this repository. If you are an academic, you can get a license for free [here](https://www.mosek.com/license/request/?i=acp).

## Credits

The design of this toolbox, in particular the documentation, is inspired by the nice work of the creators of [PEPit](https://github.com/PerformanceEstimation/PEPit?tab=readme-ov-file).
