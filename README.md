[![Python Package using Conda](https://github.com/duembgen/popr/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/duembgen/popr/actions/workflows/python-package-conda.yml)
[![Documentation build](https://github.com/duembgen/popr/actions/workflows/documentation.yml/badge.svg)](https://github.com/duembgen/popr/actions/workflows/documentation.yml)
[![Documentation deploy](https://github.com/duembgen/popcor/actions/workflows/static.yml/badge.svg)](https://duembgen.github.io/popcor)

![](./docs/source/_static/overview.png)

> [!WARNING]  
> This package is currently in pre-release state. You are welcome to try it out, and please get in touch via github or e-mail if you have any questions, suggestions or issues.

.. start-doc

# POPCOR: Polynomial Optimization for Certifiably Optimal Robotics

A toolbox for setting up and solving polynomial optimization problems in robotics with certifiable optimality.

## Purpose

This toolbox includes methods to formulate and solve polynomial optimization problems in robotics. 
The focus of this toolbox is on creating **tight** semidefinite relaxations, which means that we can
replace difficult (often NP-hard) optimization problems by easier-to-solve convex problems. In doing so, we can identify globally optimal solution of the original problem. 

The toolbox allows to run the *AutoTight* and *AutoTemplate* algorithms on problems of your choice. 
These algorithms are described in detail in [this paper](https://arxiv.org/abs/2308.05783/). For a high-level overview of how this works, please refer to the **Quick start guide**. 

If you use this toolbox, please cite our corresponding paper:
```bibtex
@article{autotight2024,
  author={Dümbgen, Frederike and Holmes, Connor and Agro, Ben and Barfoot, Timothy D.}, 
  title={{Toward Globally Optimal State Estimation Using Automatically Tightened Semidefinite Relaxations}},
  journal={IEEE Transactions on Robotics}, 
  volume={40},
  pages={4338-4358},
  year={2024},
  doi={10.1109/TRO.2024.3454570}
}
```

## Who this tool is for

This tool is created for engineers (in particular, roboticists) who want to test if an optimization problem they need to solve could be amenable to certifiable optimization. For many problems, it has been shown that random initialization can lead to convergence to poor minima, while the same problems admit a so-called **convex relaxation** which allows to solve it to global optimality (or certify local solutions). This tool is meant to help you try quickly if your current problem also belongs to this family of problems that can be solved to certifiable global optimality. 

## Dependencies

Besides the automatically installed dependencies when using the above instructions, you need to also have a valid MOSEK license in order to use this repository. If you are an academic, you can get a license for free [here](https://www.mosek.com/license/request/?i=acp).

## Credits

The design of this toolbox, in particular the documentation, is inspired by the nice work of the creators of [PEPit](https://github.com/PerformanceEstimation/PEPit?tab=readme-ov-file).

.. end-doc
