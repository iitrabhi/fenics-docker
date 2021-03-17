# Cantilever beam problem with different solvers

This folder contains the python script for a cantilever beam problem. Different solvers    are tested for the same problem. Check this link ("https://www.mcs.anl.gov/petsc/documentation/linearsolvertable.html") to know more about different solvers available which we can directly use for easy and efficient computation.  

Before selecting the type of solver you should have a prior knowledge about the type and properties of matrix involved in the computation, and also about the different solvers available. 

> Direct solver is equal to a Iterative solver with 1 number of iterations.

In the case of a cantilever beam problem, the marixes involved are symmetric in nature. I have tested for iterative solvers to solve the problem. I am mentioning specifically about two solvers i.e., Conjugate gradient (cg) method and Generalized minimal residual method (germs). It is observed that when cg method is used the solution is obtained with only 39 number of linear solver iterations where as in the case of gmres solver it took almost 560 number of linear solver iterations. When we compare the time of computation, there was almost 10 times reduction in the overall computational time. 

 So it is very important to design the solver as per your problem type. 

