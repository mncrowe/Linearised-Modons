using GeophysicalFlows, NetCDF, Bessels, Roots, CUDA
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
T, Ns = 500, 100		                # stop time and number of saves
savename = "full_modon_test_1024"	                # filename for NetCDF data file
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

end

ψ = @. (Ψᵢ(r) * (r < a) + Ψₒ(r) * (r >= a)) * sin(θ)
q = @. (Qᵢ(r) * (r < a) + Qₒ(r) * (r >= a)) * sin(θ)

ψ[isnan.(ψ)] .= 0
q[isnan.(q)] .= 0

# Build problem:

prob = SingleLayerQG.Problem(dev;
                             nx=Nx,
                             ny=Ny,
                             Lx,
                             Ly,
                             β,
                             U=-U,
                             deformation_radius=R,
			     dt=Δt,
                             stepper,
                             aliased_fraction,
                             ν,
                             nν)

# Create initial condition:

κ = @.sqrt(prob.grid.Krsq)
q₀ = to_dev(q) .+ 1e-6 * Nx * irfft(to_dev(exp.(im*2π*randn(Int(Nx/2+1), Int(Nx)))).*(κ.>κ₁).*(κ.<κ₂), Nx)

SingleLayerQG.set_q!(prob, q₀)

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
  SingleLayerQG.updatevars!(prob)

  if maximum(isnan.(prob.sol)); @warn "NaN detected."; end

  save_field_data(prob, prob.grid, filename, i1, i1*I)

end