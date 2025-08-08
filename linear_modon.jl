using FourierFlows, NetCDF, Bessels, Roots, CUDA
using LinearAlgebra: mul!, ldiv!
using Random: seed!
seed!(1)

# Define modon parameters:

U = 1.0
a = 1.0
β = 1.0 #0
R = 1.0 #Inf

# Define numerical parameters:

Nx, Ny = 1024, 1024
Lx, Ly = 10.24, 10.24
nν = 2
T, Ns = 1000, 1000		                # stop time and number of saves
savename = "nonlinear_modon_test_1024_2"	                # filename for NetCDF data file
dev = GPU()			                    # device, CPU() or GPU() (GPU is much faster)
stepper = "FilteredRK4"		          # timestepping method, e.g. "RK4", "LSRK54" or "FilteredRK4"
aliased_fraction = 0		            # fraction of wavenumbers zeroed out in dealiasing
nonlinear = true
κ₁, κ₂ = 2.5π, 5π

ν = 0.0*((Lx/Nx)^2+(Ly/Ny)^2)^nν
Δt = 0.5*((Lx/Nx)+(Ly/Ny)) / (5*U)
Nt = ceil(T / Δt)

# Helper functions:

to_CPU(f) = device_array(CPU())(f)
to_dev(f) = device_array(dev)(f)
fstring(num) = string(round(num, sigdigits=8))
istring(num) = string(Int(num))

# Create grid:

grid = TwoDGrid(dev; nx=Nx, ny=Ny, Lx, Ly)
x, y = gridpoints(grid)
r, θ = to_CPU(sqrt.(x.^2 .+ y.^2)), to_CPU(atan.(y, x))

# Define modon fields:

p = sqrt(β/U + 1/R^2)

J₁(x)  = besselj(1, x)
K₁(x)  = besselk(1, x)
J₁′(x) = (besselj(0, x) - besselj(2, x)) / 2
K₁′(x) = (-besselk(0, x) - besselk(2, x)) / 2

if p == 0

    K = 3.83170597020751231561443589 / a
    A = -U * a^2
    B = 2 * U / (K * J₁′(K * a))
    
    Ψᵢ(r) = B * J₁(K * r) - U * r
    Ψₒ(r) = A / r
    Qᵢ(r) = -K^2 * B * J₁(K * r)
    Qₒ(r) = 0
    
    Ψᵢ′(r) = B * J₁(K * r) - U * r
    Ψₒ′(r) = A / r
    Qᵢ′(r) = -K^2 * B * J₁(K * r)
    Qₒ′(r) = 0

else

    f(x) = x * J₁′(x) - (1 + x^2 / (p^2 * a^2)) * J₁(x) + x^2 * J₁(x) * K₁′(p * a) / (p * a * K₁(p * a))
    K′ = find_zero(f, 3.83170597020751231561443589)
    K = a * sqrt(K′^2 + 1/R^2)
    
    A = -U * a / K₁(p * a)
    B = p^2 * U * a / (K′^2 * J₁(K′ * a))
    
    Ψᵢ(r) = B * J₁(K′ * r) - U * (K′^2 + p^2) / K′^2 * r
    Ψₒ(r) = A * K₁(p * r)
    Qᵢ(r) = -K^2 / a^2 * B * J₁(K′ * r) + (U * p^2 * K^2 / (a^2 * K′^2) - β) * r;
    Qₒ(r) = β / U * A * K₁(p * r);
    
    Ψᵢ′(r) = B * K′ * J₁′(K′ * r) - U * (K′^2 + p^2) / K′^2
    Ψₒ′(r) = A * p * K₁′(p * r)
    Qᵢ′(r) = -K^2 / a^2 * B * K′ * J₁′(K′ * r) + (U * p^2 * K^2 / (a^2 * K′^2) - β);
    Qₒ′(r) = β / U * A * p * K₁′(p * r);

end

ψ = @. (Ψᵢ(r) * (r < a) + Ψₒ(r) * (r >= a)) * sin(θ)
q = @. (Qᵢ(r) * (r < a) + Qₒ(r) * (r >= a)) * sin(θ)
ψx = @. ((Ψᵢ′(r) - Ψᵢ(r) / r) * (r < a) + (Ψₒ′(r) - Ψₒ(r) / r) * (r >= a)) * sin(θ) * cos(θ)
qx = @. ((Qᵢ′(r) - Qᵢ(r) / r) * (r < a) + (Qₒ′(r) - Qₒ(r) / r) * (r >= a)) * sin(θ) * cos(θ)
ψy = @. (Ψᵢ′(r) * (r < a) + Ψₒ′(r) * (r >= a)) * sin(θ)^2 + (Ψᵢ(r) * (r < a) + Ψₒ(r) * (r >= a)) / r * cos(θ)^2
qy = @. (Qᵢ′(r) * (r < a) + Qₒ′(r) * (r >= a)) * sin(θ)^2 + (Qᵢ(r) * (r < a) + Qₒ(r) * (r >= a)) / r * cos(θ)^2

ψx[isnan.(ψx)] .= 0
ψy[isnan.(ψy)] .= 0
qx[isnan.(qx)] .= 0
qy[isnan.(qy)] .= 0

# Define parameter and variables structures:

struct ParamsFiniteR{T, Aphys} <: AbstractParams
   ν :: T
  nν :: Int
   β :: T
   R :: T
   U :: T
  Ψx :: Aphys
  Ψy :: Aphys
  Qx :: Aphys
  Qy :: Aphys
end

struct ParamsInfiniteR{T, Aphys} <: AbstractParams
   ν :: T
  nν :: Int
   β :: T
   U :: T
  Ψx :: Aphys
  Ψy :: Aphys
  Qx :: Aphys
  Qy :: Aphys
end

struct Vars{Aphys, Atrans} <: AbstractVars
   q :: Aphys
   ψ :: Aphys
   u :: Aphys
   v :: Aphys
  qh :: Atrans
  ψh :: Atrans
  uh :: Atrans
  vh :: Atrans
end

function Vars(grid)
  
  Dev = typeof(grid.device)
  T = eltype(grid)

  @devzeros Dev T (grid.nx, grid.ny) q ψ u v
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) qh ψh uh vh

  return Vars(q, ψ, u, v, qh, ψh, uh, vh)
end

# Define linear and nonlinear terms:

function calcN!(N, sol, t, clock, vars, params, grid)

  dealias!(sol, grid)

  @. vars.qh = sol

  streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)
  
  @. vars.uh = -im * grid.l  * vars.ψh
  @. vars.vh =  im * grid.kr * vars.ψh

  ldiv!(vars.u, grid.rfftplan, vars.uh)
  ldiv!(vars.v, grid.rfftplan, vars.vh)

  uQx, vQy = vars.u, vars.v
  uQxh, vQyh = vars.uh, vars.vh

  @. uQx *= params.Qx
  @. vQy *= params.Qy

  mul!(uQxh, grid.rfftplan, uQx)
  mul!(vQyh, grid.rfftplan, vQy)
  
  @. N = - uQxh - vQyh

  qx, qy = vars.u, vars.v
  qxh, qyh = vars.uh, vars.vh

  @. qxh = im * grid.kr * vars.qh
  @. qyh = im * grid.l  * vars.qh

  ldiv!(qx, grid.rfftplan, qxh)
  ldiv!(qy, grid.rfftplan, qyh)

  JΨq, JΨqh = vars.ψ, vars.ψh

  @. JΨq = qx * params.Ψy - qy * params.Ψx

  mul!(JΨqh, grid.rfftplan, JΨq)
  
  @. N += JΨqh

  return nothing
end

function calcNnonlin!(N, sol, t, clock, vars, params, grid)

  calcN!(N, sol, t, clock, vars, params, grid)

  @. vars.qh = sol

  streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)

  @. vars.uh = -im * grid.l  * vars.ψh
  @. vars.vh =  im * grid.kr * vars.ψh

  ldiv!(vars.q, grid.rfftplan, vars.qh)
  ldiv!(vars.u, grid.rfftplan, vars.uh)
  ldiv!(vars.v, grid.rfftplan, vars.vh)

  uq, vq = vars.u, vars.v                   # use vars.u, vars.v as scratch variable
  @. uq *= vars.q                           # uq
  @. vq *= vars.q                           # vq

  uqh, vqh = vars.uh, vars.vh               # use vars.uh, vars.vh as scratch variable
  mul!(uqh, grid.rfftplan, uq)              # \hat{uq}
  mul!(vqh, grid.rfftplan, vq)              # \hat{vq}

  @. N += -im * grid.kr * uqh - im * grid.l * vqh  # - ∂[uq]/∂x - ∂[vq]/∂y

  return nothing
end

function Equation(params::ParamsInfiniteR, grid)
 
  L = @. im * params.U * grid.kr - params.ν * grid.Krsq^params.nν + im * params.β * grid.kr * grid.invKrsq

  CUDA.@allowscalar L[1, 1] = 0

  calcNfunc! = nonlinear ? calcNnonlin! : calcN!

  return FourierFlows.Equation(L, calcNfunc!, grid)
end

function Equation(params::ParamsFiniteR, grid)
 
  L = @. im * params.U * grid.kr - params.ν * grid.Krsq^params.nν + im * params.β * grid.kr / (grid.Krsq + 1 / params.R^2) 

  CUDA.@allowscalar L[1, 1] = 0

  calcNfunc! = nonlinear ? calcNnonlin! : calcN!

  return FourierFlows.Equation(L, calcNfunc!, grid)
end

function updatevars!(sol, vars, params, grid)
  dealias!(sol, grid)

  @. vars.qh = sol
  streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)
  @. vars.uh = -im * grid.l  * vars.ψh
  @. vars.vh =  im * grid.kr * vars.ψh

  ldiv!(vars.q, grid.rfftplan, deepcopy(vars.qh))
  ldiv!(vars.ψ, grid.rfftplan, deepcopy(vars.ψh))
  ldiv!(vars.u, grid.rfftplan, deepcopy(vars.uh))
  ldiv!(vars.v, grid.rfftplan, deepcopy(vars.vh))

  return nothing
end

function streamfunctionfrompv!(ψh, qh, params::ParamsFiniteR, grid)
  @. ψh =  - qh / (grid.Krsq + 1 / params.R^2)
  return nothing
end

function streamfunctionfrompv!(ψh, qh, params::ParamsInfiniteR, grid)
  @. ψh =  - qh * grid.invKrsq
  return nothing
end

updatevars!(prob) = updatevars!(prob.sol, prob.vars, prob.params, prob.grid)

# Build problem:

  params = R == Inf ? ParamsInfiniteR(ν, nν, β, U, to_dev(ψx), to_dev(ψy), to_dev(qx), to_dev(qy)) :
                      ParamsFiniteR(ν, nν, β, R, U, to_dev(ψx), to_dev(ψy), to_dev(qx), to_dev(qy))
    vars = Vars(grid)
equation = Equation(params, grid)

    prob = FourierFlows.Problem(equation, stepper, Δt, grid, vars, params)

# Create initial condition:

κ = @.sqrt(prob.grid.Krsq)
prob.sol .= 1e-6 * Nx * device_array(dev)(exp.(im*2π*randn(Int(Nx/2+1), Int(Nx)))).*(κ.>κ₁).*(κ.<κ₂)
updatevars!(prob)

# Define output saves:

filename = savename * ".h5"

if isfile(filename); rm(filename); end

nccreate(filename, "psi", "x", grid.x, "y", grid.y, "t", LinRange(0,T,Ns+1))
nccreate(filename, "q", "x", grid.x, "y", grid.y, "t", LinRange(0,T,Ns+1))
ncputatt(filename," ", Dict("R" => R, "U" => U, "a" => a, "b" => β))

function save_field_data(problem, grid, filename, i, iter)
  ψ, q = reshape(to_CPU(problem.vars.ψ),(Nx, Ny, 1)), reshape(to_CPU(problem.vars.q),(Nx, Ny, 1))

  ncwrite(ψ, filename, "psi", start = [1, 1, i+1], count = [Nx, Ny, 1])
  ncwrite(q, filename, "q", start = [1, 1, i+1], count = [Nx, Ny, 1])

  println("Iteration: " * istring(iter) * ", t = " * fstring(problem.clock.t))

  return nothing
end

save_field_data(prob, prob.grid, filename, 0, 0)	# initial save

# Run simulation:

I = Int(Nt/Ns)

for i1 in 1:ceil(Nt/I)

  stepforward!(prob, I)		# evolve problem in time
  updatevars!(prob)

  if maximum(isnan.(prob.sol)); @warn "NaN detected."; end

  save_field_data(prob, prob.grid, filename, i1, i1*I)

end