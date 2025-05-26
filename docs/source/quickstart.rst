Quick Start Guide
=================

Installation
------------

POPR can be installed by running from a terminal:

.. code-block:: bash

   git clone --recurse-submodules git@github.com:duembgen/popr
   cd popr
   conda env create -f environment.yml



Problem Formulation
-------------------

We start with polynomial optimization problems (POPs) of the form:

.. math::

  \begin{align} q^\star =&\min_{\theta} f(\theta)   \\ 
   \text{s.t. } &g(\theta) = 0 \\
                &h(\theta) \geq 0
  \end{align} 

where :math:`f,g,\text{ and } h` are polynomial functions, and both :math:`g` and :math:`h` can be vector-valued.  Many maximum-a-posteriori or maximum-likelihood estimation problems can be formulated as such, for example `range-only localization <https://arxiv.org/abs/2209.04266/>`_ and `range-aided SLAM <https://arxiv.org/abs/2302.11614/>`_, (`matrix-weighted <https://arxiv.org/abs/2308.07275>`_) `SLAM <https://arxiv.org/abs/1612.07386/>`_, and `outlier-robust estimation <https://ieeexplore.ieee.org/abstract/document/9785843/>`_. The same is true for many control and planning problems, for example the `inverted pendulum <https://arxiv.org/abs/2406.05846>`_ and other classical dynamical systems, and even contact-rich problems such as `slider-pusher planning problems <https://arxiv.org/abs/2402.10312>`_.

Any POP can be equivalently written in the following QCQP form:

.. math::

  \begin{align} q^\star =&\min_{x} x^\top Q x  \\ 
   \text{s.t. } &(\forall i): x^\top A_i x = b_i \\
                &(\forall j): x^\top B_j x \geq 0
  \end{align} 

with cost matrix :math:`Q`, known constraint matrices :math:`A_i,B_j`. 
Note that

- We always include the so-called homogenization variable, which enables to write linear and constant terms as quadratics. By convention, we set the first element of :math:`x` to one, and we use :math:`b_0=1, A_0` to encorce this constraint. 
- All inequality and some equality constraints correspond to the constraints from the original POP. 
- Some additional equality constraints correspond to new substitution variables that need to be added to formulate the problem as a quadratic. 

.. warning:: 
   Note that while inequality constraints can be added to the problem formulation, there is no implementation yet to add find and add redundant inequality constraints to the relaxation. 

For the standard usage, the user first needs to define a custom **Lifter** class which essentially contains all elements related to the QCQP problem formulation.
This class should inherit from :ref:`StateLifter`. A basic skeleton of such a 
Lifter class is provided in :ref:`Example for AutoTight`. The main purpose of this class is 
that it provides all basic operations related to the problem formulation, such as:

- to sample feasible states (:meth:`popr.lifters.StateLifter.sample_theta`),
- to get the lifted vector (:meth:`popr.lifters.StateLifter.get_x`),

For a bit more advanced functionality (for example for the :ref:`SDP Relaxation` in the next section), you also need to define functions such as

- get the cost matrix (:py:meth:`popr.lifters.StateLifter.get_Q`), 
- get known constraint matrices (:meth:`popr.lifters.StateLifter.get_A_known`, :meth:`popr.lifters.StateLifter.get_B_known`).

Many example lifters are provided, you can find them under :ref:`Examples`.

**Example: instantiating and using lifter**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following code snippet shows some basic operations (and useful sanity checks) for the example
lifter class :class:`popr.examples.Poly4Lifter`. Note that this and all following examples can be found
in the file :file:`../../tests/test_quickstart.py`.

.. literalinclude:: ../../tests/test_quickstart.py
   :language: python
   :lines: 9-23
   :dedent: 4


SDP Relaxation
--------------

It is straightforward to derive a convex relaxation of the original QCQP, using the reformulation :math:`x^\top Qx=\langle x, Qx\rangle = \langle Q, xx^\top \rangle`, where :math:`\langle \cdot, \cdot \rangle` denotes the trace inner product. Then introducing :math:`X:=xx^\top` and relaxing its rank, we obtain the following convex relaxation, in the form of an SDP: 

.. math::
  \begin{align} p^\star = &\min_{X \succeq 0} \langle Q, X \rangle  \\ 
   \text{s.t. } &(\forall i): \langle A_i, X \rangle = b_i \\
                 &(\forall j): \langle B_j, X \rangle \geq 0 
  \end{align} 


**Example: solving the QCQP using rank relaxation**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following code snippet shows how you can use the simple lifter from earlier to find the global
optimum of the nonconvex polynomial problem, by solving an SDP.

.. literalinclude:: ../../tests/test_quickstart.py
   :language: python
   :lines: 28-55
   :dedent: 4



AutoTight Method
----------------

**AutoTight** is used to find all possible constraints matrices :math:`A_r`, which are also automatically satisfied by solutions of the QCQP. They are also called **redundant constraints** because they do not change the feasible set of the original problem, but when adding those constraints to the SDP (rank-)relaxation, they often improve tightness. Denoting by :math:`A_r` the redundant constraints, we can solve the following SDP: 

.. math::
  \begin{align} p_r^\star = &\min_{X \succeq 0} \langle Q, X \rangle  \\ 
   \text{s.t. } &(\forall i): \langle A_i, X \rangle = b_i \\
                 &(\forall r): \langle A_r, X \rangle = 0 \\
                 &(\forall j): \langle B_j, X \rangle \geq 0 
  \end{align} 

We use the term **cost-tight** to say that strong duality holds (:math:`p_r^\star = q^\star`) while by rank-tight we denote the fact that the SDP solver returns a rank-one solution.
If successful, the output is a set of constraints that leads to a tight SDP relaxation of the original problem, which can be used to solve the problem to global optimality (if we have rank tightness) or certify given solutions (if we have cost tightness). 

More information on how to use AutoTight can be found :ref:`here <AutoTight>` and a simple example is given next. 

**Example: tightening the SDP relaxation using AutoTight.**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../tests/test_quickstart.py
   :language: python
   :lines: 60-100
   :dedent: 4


AutoTemplate Method
-------------------

*AutoTemplate* follows the same principle as *AutoTight*, but its output are templates rather than constraint matrices. These templates can be seen as "parametrized" versions of the constraint matrices, and can be applied to new problem instances of any size without having to learn the constraints again from scratch. 

More information on how to use AutoTemplate can be found :ref:`here <AutoTemplate>` and a simple example is given next. 

.. literalinclude:: ../../tests/test_quickstart.py
   :language: python
   :lines: 105-135
   :dedent: 4


References
----------

`[1] F. Dümbgen, C. Holmes, B. Agro and T. Barfoot, "Toward Globally Optimal State Estimation Using Automatically Tightened Semidefinite Relaxations," in IEEE Transactions on Robotics, vol. 40, pp. 4338-4358, 2024, doi: 10.1109/TRO.2024.3454570. <https://arxiv.org/abs/2308.05783>`_
