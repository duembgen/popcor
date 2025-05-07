Quick start guide
=================

This toolbox includes methods to formulate and solve polynomial optimization problems in robotics. 
The focus of this toolbox is on creating **tight** semidefinite relaxations, which means that we can
replace difficult (often NP-hard) optimization problems by easier-to-solve convex problems, which
in many cases can actually provide the globally optimal solution of the original problem. 

The toolbox allows to run the *AutoTight* and *AutoTemplate* algorithms on problems of your choice. 
These algorithms are described in detail in `this paper <https://arxiv.org/abs/2308.05783/>`_ [1].


Background
----------

We start with an optimization problem written in (QCQP) form:

.. math::

  \begin{align} q^\star =&\min_{x} x^\top Q x  \\ 
  & \text{s.t. } (\forall i): x^\top A_i x = b_i
  \end{align} 

with cost matrix :math:`Q`, known constraint matrices :math:`A_i`, and :math:`b_0=1`, :math:`b_{i}=0, i>0`. Many maximum-a-posteriori or maximum-likelihood estimation problems can be written as such, for example `range-only localization <https://arxiv.org/abs/2209.04266/>`_ and `range-aided SLAM <https://arxiv.org/abs/2302.11614/>`_, (`matrix-weighted <https://arxiv.org/abs/2308.07275>`_) `SLAM <https://arxiv.org/abs/1612.07386/>`_, and `outlier-robust estimation <https://ieeexplore.ieee.org/abstract/document/9785843/>`_. The same is true for many control and planning problems, for example the `inverted pendulum <https://arxiv.org/abs/2406.05846>`_ and other classical dynamical systems, and even contact-rich problems such as `slider-pusher planning problems <https://arxiv.org/abs/2402.10312>`_. 

Algorithms
----------

The main tools that this toolbox provides are the following two classes, implemented in `this paper <https://arxiv.org/abs/2308.05783/>`_.

AutoTight Method
^^^^^^^^^^^^^^^^

*AutoTight* finds all possible additional constraints matrices :math:`A_r` which are also automatically satisfied by solutions of (QCQP), called **redundant constraints** and checks if the SDP (rank-)relaxation of the QCQP is cost and/or rank-tight after adding them. The rank relaxation is given by:

.. math::
  \begin{align} p^\star = &\min_{X} \langle Q, X \rangle  \\ 
  & \text{s.t. } (\forall i): \langle A_i, X \rangle = b_i \\
  & \text{s.t. } (\forall r): \langle A_r, X \rangle = 0 
  \end{align} 

In this context, cost-tight means that strong duality holds (:math:`p^\star = q^\star`) while rank-tight means that we even have :math:`\text{rank}(X)=1`.
If successful, the output is a set of constraints that leads to a tight SDP relaxation of the original problem, which can be used to solve the problem to global optimality (if we have rank tightness) or certify given solutions (if we have cost tightness). 

More information on how to use AutoTight can be found in :class:`popr.auto_tight.AutoTight`.

AutoTemplate Method
^^^^^^^^^^^^^^^^^^^

*AutoTemplate* follows the same principle as *AutoTight*, but its output are templates rather than constraints. These templates can be seen as "parametrized" versions of the constraints matrices, and can be applied to new problem instances of any size without having to learn the constraints again from scratch. 

More information on how to use AutoTemplate can be found :ref:`here <AutoTemplate>`.

Installation
------------

The tool can be installed by running from a terminal:

.. code-block:: bash

  git clone --recurse-submodules git@github.com:utiasASRL/popr
  cd popr
  conda env create -f environment.yml

Basic Usage
-----------

For the standard usage, the user first needs to define a custom **Lifter** class.
This class should inherit from :ref:`StateLifter`. A basic skeleton of such a 
Lifter class is provided in :ref:`ExampleLifter`. The main purpose of this class is 
that it provides all basic operations related to the problem formulation, such as:
- to sample feasible states (:meth:`popr.lifters.StateLifter.sample_theta`),
- to get the lifted vector (:meth:`popr.lifters.StateLifter.get_x`),

For most functionality you also need to define functions to
- to get the cost matrix (:py:meth:`popr.lifters.StateLifter.get_Q`), 
- to get known constraint matrices (:meth:`popr.lifters.StateLifter.get_A_known`, :meth:`popr.lifters.StateLifter.get_B_known`).

Many example lifters are provided, you can find them under :ref:`Examples`.

Some basic sanity checks
~~~~~~~~~~~~~~~~~~~~~~~~
The following code snippet shows some basic operations (and useful sanity checks) for the example
lifter class :class:`popr.examples.Poly4Lifter`.

.. testcode::

    from popr.examples import Poly4Lifter

    lifter = Poly4Lifter()

    Q = lifter.get_Q()

    # theta corresponds to the ground truth; in this case, the global minimum. 
    x = lifter.get_x(lifter.theta)
    cost_optimum = float(x.T @ Q @ x)

    # the cost at any other randomly sampled point has to be larger. 
    for i in range(10):
        theta_random = lifter.sample_theta()
        x_random = lifter.get_x(theta_random)
        assert float(x_random.T @ Q @ x_random) > cost_optimum


Solving simple SDPs
~~~~~~~~~~~~~~~~~~~

The following code snippet shows how you can already use this simple lifter to find the global
optimum of this polynomial, by solving an SDP.

.. testcode::

    from cert_tools.sdp_solvers import solve_sdp
    from cert_tools.linalg_tools import rank_project
    from popr.examples import Poly4Lifter

    lifter = Poly4Lifter()

    # the cost matrix
    Q = lifter.get_Q()

    # the equality constraints
    A_known = lifter.get_A_known()

    # the homogenization constraint
    A_0 = lifter.get_A0()

    X, info = solve_sdp(Q, [(A_0, 1.0)] + [(A_i, 0.0) for A_i in A_known])

    # if X is rank one, the global optimum can be found in element X_10 of the matrix.
    theta_pick = X[1, 0] 
    assert abs(theta_pick - lifter.theta) < 1e-5

    # We can also first extract the rank-1 estimate (X=xx') and then extract theta.
    x, info_rank = rank_project(X, p=1)
    theta_round = x[1]

    assert abs(theta_round - lifter.theta) < 1e-5


References
----------

`[1] F. Dümbgen, C. Holmes, B. Agro and T. Barfoot, "Toward Globally Optimal State Estimation Using Automatically Tightened Semidefinite Relaxations," in IEEE Transactions on Robotics, vol. 40, pp. 4338-4358, 2024, doi: 10.1109/TRO.2024.3454570. <https://arxiv.org/abs/2308.05783>`_
