# POPR: Polynomial OPtimization for Robotics

A toolbox for setting up and solving, to global optimality, polynomial optimization problems in robotics. 

## Purpose

This toolbox includes methods to formulate and solve polynomial optimization problems in robotics. 
The focus of this toolbox is on creating **tight** semidefinite relaxations, which means that we can
replace difficult (often NP-hard) optimization problems by easier-to-solve convex problems, which
in many cases can actually provide the globally optimal solution of the original problem. 

The toolbox allows to run the *AutoTight* and *AutoTemplate* algorithms on problems of your choice. 
These algorithms are described in detail in [this paper](https://arxiv.org/abs/2308.05783/) [1].
For a high-level overview of how this works, please refer to the **Quick start guide**. 

## What this tool can and cannot do

This tool can be seen as a bridge between the mathematical problem formulation and the algebraic modeling languages (AMLs). AMLs take as an input the *mathematical problem* written by a user, and convert it to a standard form [`[1]`](https://arxiv.org/pdf/2002.03447). However, coming up with the *mathematical problem* is sometimes a challenge in itself.

In certifiable optimization, for example, we typically solve an SDP that was derived as the rank relaxation or Lagrangian dual of an original, non-convex problem. Finding the correct SDP can be however quite difficult! For example, one might have to add many constraints to the SDP that are redundant for the original non-convex problem but change the solution of the relaxation, making it more "tight". 

Currently, this tool is focused on finding redundant constraints. There are however numerous extensions that a powerful "bridge" should also include and that are topics of ongoing research. For example, interesting features to this toolbox would be:

- Problem conditioning: applying a coordinate transform to improve the convergence properties of the problem.
- Data-driven modeling: deriving the correct problem formulation directly from data.

## Dependencies

Besides the automatically installed dependencies when using the above instructions, you need to also have a valid MOSEK license in order to use this repository. If you are an academic, you can get a license for free [here](https://www.mosek.com/license/request/?i=acp).

## Credits

The design of this toolbox, in particular the documentation, is inspired by the nice work of the creators of [PEPit](https://github.com/PerformanceEstimation/PEPit?tab=readme-ov-file).
