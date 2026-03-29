"""
Julia GPU ensemble benchmark for 35 coupled van der Pol oscillators (70D).

Reproduces the benchmark flow from GPUODEBenchmarks:
- DiffEqGPU.vectorized_solve (fixed dt)
- DiffEqGPU.vectorized_asolve (adaptive dt)

Usage:
    julia benchmarks/vdp_julia.jl [N] [rtol] [atol]

N: number of trajectories (default 8192)
rtol: relative tolerance (default 1e-6)
atol: absolute tolerance (default 1e-8)

Outputs JSON to stdout with minimum benchmark times in milliseconds.
"""

using DiffEqGPU, StaticArrays
using CUDA
using JSON
using Random

numberOfParameters = length(ARGS) >= 1 ? parse(Int64, ARGS[1]) : 8192
const RTOL = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1e-6
const ATOL = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1e-8

const SOLVER = GPURosenbrock23()
const SOLVER_NAME = "GPURosenbrock23"

const N_OSC = 35
const N_VARS = 2 * N_OSC  # 70

# Damping coefficients matching Python: 10^(-0.3 + 2.3*i/(N_OSC-1)) for i=0..34
const MU = ntuple(Val(N_OSC)) do j
    10.0^(-0.3 + 2.3 * (j - 1) / (N_OSC - 1))
end

function vdp(u, p, t)
    s = p[1]
    vals = ntuple(Val(N_VARS)) do k
        i = div(k - 1, 2) + 1  # oscillator index (1-based)
        if isodd(k)
            # dx_i/dt = v_i
            u[k + 1]
        else
            # dv_i/dt = s * mu_i * (1 - x_i^2) * v_i - x_i
            x = u[k - 1]
            v = u[k]
            s * MU[i] * (1.0 - x * x) * v - x
        end
    end
    return SVector{N_VARS}(vals)
end

# Initial conditions: x=2, v=0 for each oscillator
u0 = SVector{N_VARS}(ntuple(k -> isodd(k) ? 2.0 : 0.0, Val(N_VARS)))
tspan = (0.0, 10.0)
p0 = SVector{1}(1.0)

vdpProblem = DiffEqGPU.ODEProblem(vdp, u0, tspan, p0)

rng = MersenneTwister(42)
parameterList = [SVector{1,Float64}(1.0 + 0.1 * (2.0 * rand(rng) - 1.0)) for _ in 1:numberOfParameters]

prob_func = (prob, i, repeat) -> DiffEqGPU.remake(prob, p = parameterList[i])
ensembleProb = DiffEqGPU.EnsembleProblem(vdpProblem, prob_func = prob_func)

function min_time_ms(f; repeats = 5)
    best = Inf
    for _ in 1:repeats
        CUDA.synchronize()
        t0 = time_ns()
        f()
        CUDA.synchronize()
        dt_ms = (time_ns() - t0) / 1e6
        best = min(best, dt_ms)
    end
    return best
end

fixed_min_ms = 0.0
adaptive_min_ms = 0.0

# Strict paper path: build per-trajectory problems and use vectorized APIs.
I = 1:numberOfParameters
if ensembleProb.safetycopy
    probs = map(I) do i
        p = ensembleProb.prob_func(deepcopy(ensembleProb.prob), i, 1)
        convert(DiffEqGPU.ImmutableODEProblem, p)
    end
else
    probs = map(I) do i
        p = ensembleProb.prob_func(ensembleProb.prob, i, 1)
        convert(DiffEqGPU.ImmutableODEProblem, p)
    end
end

probs = cu(probs)
fixed_dt = (tspan[2] - tspan[1]) / 1000

fixed_min_ms = min_time_ms() do
    CUDA.@sync DiffEqGPU.vectorized_solve(probs, ensembleProb.prob,
                                          SOLVER;
                                          save_everystep = false,
                                          dt = fixed_dt)
end
adaptive_min_ms = min_time_ms() do
    CUDA.@sync DiffEqGPU.vectorized_asolve(probs, ensembleProb.prob,
                                           SOLVER;
                                           dt = 1e-3,
                                           reltol = RTOL,
                                           abstol = ATOL)
end

result = Dict(
    "solver" => SOLVER_NAME,
    "n_trajectories" => numberOfParameters,
    "n_vars" => N_VARS,
    "n_oscillators" => N_OSC,
    "fixed_dt" => Dict(
        "dt" => fixed_dt,
        "min_time_ms" => fixed_min_ms,
    ),
    "adaptive_dt" => Dict(
        "dt" => 1e-3,
        "reltol" => RTOL,
        "abstol" => ATOL,
        "min_time_ms" => adaptive_min_ms,
    ),
)
println(JSON.json(result))
