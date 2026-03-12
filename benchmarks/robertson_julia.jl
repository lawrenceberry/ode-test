"""
Julia GPU ensemble benchmark for Robertson equation.

Usage:
    julia robertson_julia.jl [N]

N: number of trajectories (default 2)

Runs with Float64 only.
Outputs JSON to stdout with timing and results.
"""

using OrdinaryDiffEq, StaticArrays, LinearAlgebra, JSON, Random, CUDA, DiffEqGPU

N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2
const T = Float64

# Paper-style Robertson ODE form (conservative):
#   dy1/dt = -k1*y1 + k3*y2*y3
#   dy2/dt =  k1*y1 - k2*y2^2 - k3*y2*y3
#   dy3/dt =  k2*y2^2
function robertson_ode(u, p, t)
    y1, y2, y3 = u
    k1, k2, k3 = p
    return @SVector [
        -k1 * y1 + k3 * y2 * y3,
         k1 * y1 - k2 * y2^2 - k3 * y2 * y3,
         k2 * y2^2,
    ]
end

params = @SVector [T(0.04), T(3e7), T(1e4)]
u0 = @SVector [one(T), zero(T), zero(T)]
tspan = (zero(T), T(1e5))
prob = ODEProblem{false}(robertson_ode, u0, tspan, params)

# Generate perturbed parameters (±10%)
function make_perturbed_params(N::Int)
    base = [T(0.04), T(3e7), T(1e4)]
    rng = MersenneTwister(42)
    return [SVector{3,T}(base .* (one(T) .+ T(0.1) .* (2 .* rand(rng, T, 3) .- one(T)))) for _ in 1:N]
end

params_list = make_perturbed_params(N)
prob_func = (prob, i, repeat) -> remake(prob, p=params_list[i])
monteprob = EnsembleProblem(prob, prob_func=prob_func, safetycopy=false)

backend = CUDA.CUDABackend()

reltol = T(1e-6)
abstol = T(1e-8)
dt0 = T(1e-3)

function run_solve(monteprob, backend, N, dt0, abstol, reltol)
    return solve(monteprob, GPURosenbrock23(), EnsembleGPUKernel(backend),
                 trajectories=N, adaptive=true, dt=dt0, abstol=abstol, reltol=reltol,
                 save_everystep=false)
end

# Warmup (compile GPU kernel)
warmup_monte = EnsembleProblem(prob, prob_func=(p,i,r)->p, safetycopy=false)
run_solve(warmup_monte, backend, 2, dt0, abstol, reltol)
CUDA.synchronize()

# Benchmark
CUDA.synchronize()
t_start = time()
sol = run_solve(monteprob, backend, N, dt0, abstol, reltol)
CUDA.synchronize()
elapsed = time() - t_start

finals = [[s.u[end][j] for j in 1:3] for s in sol.u]
conservations = [sum(s.u[end]) for s in sol.u]
result = Dict(
    "solver" => "GPURosenbrock23",
    "backend" => "CUDA",
    "reltol" => reltol,
    "abstol" => abstol,
    "elapsed_seconds" => elapsed,
    "n_trajectories" => N,
    "y_finals" => finals,
    "conservations" => conservations,
    "converged" => sol.converged,
)
println(JSON.json(result))
