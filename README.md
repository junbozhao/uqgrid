UQGrid
======
Uncertainty quantification for the electrical grid.

This is a work-in-progress package to perform uncertainty quantification-type computations in the power grid. Right now, it is possible to integrate a DAE and obtain first and second-order sensitivities with respect to a parameter.

In the /bin/ directory there are two examples: one with the IEEE nine bus test case and another with the New England test case. In these examples we compute sensitivities with respect to the parameters of the ZIP loads. There's also a notebook where the first-order sensitivities are verified with finite differences.

For any questions, please email:
Author: D. Adrian Maldonado (maldonadod AT anl.gov)
Institution: Argonne National Laboratory

Acknowledgements
----------------
This material is based upon work supported by the U.S. Department of Energy, Office of Science, under contract number DE-AC02-06CH11357.
