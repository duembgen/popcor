# Related Software

## Other Tools

- [TS-SOS](https://wangjie212.github.io/TSSOS/dev/) is the most similar tool to ours. It takes as input a multivariate polynomial optimization problem with arbitrary polynomial cost, equality, and inequality constraints. It then formulates a hierarchy of SDP relaxations, possibly exploiting correlative and term sparsity `[1][2][5]`.

- [SOSTOOLS](https://github.com/oxfordcontrol/SOSTOOLS), [Globtipoli](https://homepages.laas.fr/henrion/software/gloptipoly3/): Matlab toolboxes with similar features as TS-SOS, but no sparsity exploitation.

## Algebraic Modeling Languages

### Commercial

- General Algebraic Modeling Language [GAMS](https://www.gams.com/49/docs/UG_Tutorial.html), created in 1976 by the World Bank: high-level language, allowing to define sets, parameters, variables, equations and then calling a number of solvers. Underlying principle: set-based declarative algebraic modeling. Can choose between many solvers such as Baron, Xpress, Gurobi. Since 2023, also available in Python ([GAMSPy](https://gamspy.readthedocs.io/en/latest/)). Supposedly very good at [handling sparsity](https://www.gams.com/blog/2023/07/performance-in-optimization-models-a-comparative-analysis-of-gams-pyomo-gurobipy-and-jump/). The key in GAMS is that it is using relational algebra, which originates from database theory, to automatically infer what variable combinations to include in the problem; rather than giving that freedom to the user.

- [AMPL](https://ampl.com/): AMPL is a powerful and flexible algebraic modeling language for large-scale optimization problems. It supports a wide range of solvers and is widely used in both academia and industry for formulating and solving mathematical programming models.

- [AIMMS](https://www.aimms.com/): AIMMS is a commercial modeling system that allows users to build optimization models and applications with a graphical interface. It is particularly known for its ease of use in decision support and supply chain optimization.

### Open-source

- [CVX](http://cvxr.com/cvx/): CVX is a modeling framework for convex optimization problems, designed for use with MATLAB. It allows users to express optimization problems in a natural, readable way and interfaces with several solvers.

- [CVXPY](https://www.cvxpy.org/): CVXPY is a Python library for convex optimization, supporting a wide variety of problem types and solvers. It is widely used in research and industry for its flexibility and ease of use.

- [Yalmip](https://yalmip.github.io/): YALMIP is a MATLAB toolbox for modeling and solving optimization problems. It is particularly strong in semidefinite programming and is popular for control and systems applications.

- [Pyomo](https://pyomo.readthedocs.io/en/stable/) (Python): developed in 2021 by Michael L. Bynum et al., Sandia National Laboratories. Pyomo is a Python-based, open-source optimization modeling language that supports a wide range of problem types, including linear, nonlinear, and mixed-integer programming.

- [JuMP 1.0](https://jump.dev/JuMP.jl/stable/) (Julia): released in 2022 by Miles Lubin et al., Google Research. JuMP is a domain-specific modeling language for mathematical optimization embedded in Julia, known for its speed and flexibility. Original version [JuMP 0.12](https://github.com/jump-dev/JuMP.jl/releases/tag/v0.12.0) released by Dunning, Huchette, Lubin (2016). Main improvement is the transition to [MathOptInterface](https://arxiv.org/pdf/2002.03447) by Benoit Legat (UCLouvain).

- [PuLP](https://github.com/coin-or/pulp) (Python): LPs and MILPs only. PuLP is an open-source linear programming modeler written in Python. It is easy to use and interfaces with many solvers, making it suitable for educational purposes and small to medium-sized optimization problems.

## Surveys / Studies

- [Study](https://skoge.folk.ntnu.no/prost/proceedings/focapo-cpc-2023/Oral%20Talks/56_Oral.pdf) comparing Pyomo and JuMP, from CMU. This study provides a detailed comparison of modeling capabilities, performance, and user experience between two leading open-source algebraic modeling languages.

## What this tool can and cannot do

This tool can be seen as a bridge between the mathematical problem formulation and the algebraic modeling languages (AMLs). AMLs take as an input the *mathematical problem* written by a user, and convert it to a standard form [`[1]`](https://arxiv.org/pdf/2002.03447). One of the most well-known AMLs is for example shipped with [CVXPY](https://www.cvxpy.org/). However, coming up with the *mathematical problem* is sometimes a challenge in itself.

In certifiable optimization, for example, we typically solve an SDP that was derived as the rank relaxation or Lagrangian dual of an original, non-convex problem. Finding the correct SDP can be however quite difficult! For example, one might have to add many constraints to the SDP that are redundant for the original non-convex problem but change the solution of the relaxation, making it more "tight". 

Currently, this tool is focused on tightening relaxations by finding redundant constraints. There are however numerous extensions that a powerful "bridge" should also include and that are topics of ongoing research. For example, interesting features to this toolbox would be:

- Problem conditioning: applying a coordinate transform to improve the convergence properties of the problem.
- Data-driven modeling: deriving the correct problem formulation directly from data.


## Citations

[1] https://wangjie212.github.io/jiewang/research/tssos.pdf
[2] https://arxiv.org/abs/2103.00915
[6] https://github.com/oxfordcontrol/SOSTOOLS
[7] https://www.cds.caltech.edu/sostools/
[8] https://arxiv.org/abs/1912.08899
