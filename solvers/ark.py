# TODO: Implement a 5th order ARK additive implicit-explicit solver. It should only accept linear ODE systems defined by a jac_fn because these are the ones
# for which we can avoid an iterative nonlinear solve within each step. It should allow for either the splitting of jac_fn upfront via a jac_fn = (f,g) interface
# or a single jac_fn which is dynamically split using Gershgorin's circle theorem and subsequent reodering of the jacobian matrix.
