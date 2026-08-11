
include(joinpath(@__DIR__, "..", "src", "common.jl"))

using OrdinaryDiffEqOperatorSplitting
using SciMLLogging, DiffEqBase, SciMLIterators
using SciMLOperators, LinearSolve
using OrdinaryDiffEqSDIRK, OrdinaryDiffEqLowOrderRK, OrdinaryDiffEqTsit5
using LinearAlgebra

# Helper for the reaction evaluation
struct ReactionFunction{MType}
    material::MType
end
function (f::ReactionFunction)(du, u, p, t)
    num_individual_reaction_dofs = length(u) ÷ 2
    reactant_matrix = reshape(u, (2, num_individual_reaction_dofs))
    rate_matrix = reshape(du, (2, num_individual_reaction_dofs))
    for i in 1:num_individual_reaction_dofs
        r₁ = reactant_matrix[1, i]
        r₂ = reactant_matrix[2, i]
        duᵢ = reaction_kernel((r₁, r₂), f.material, t)
        rate_matrix[1, i] = duᵢ[1]
        rate_matrix[2, i] = duᵢ[2]
    end
    return
end

function gray_scott_on_sphere_os(material::GrayScottMaterial, alg, Δt₀::Real, Δtvis::Real, T::Real, refinements::Integer, adaptive)
    ## We start by setting up grid, dof handler and the matrices for the heat problem.
    grid = create_embedded_sphere(refinements)

    ## Next we are creating our element assembly helper for surface elements.
    ip = Lagrange{RefTriangle, 1}()
    qr = QuadratureRule{RefTriangle}(2)
    cellvalues = CellValues(qr, ip, ip^3)

    ## We have two options to add the reactants to the dof handler, which will give us slightly
    ## different resulting dof distributions:
    ## A) We can add a scalar-valued interpolation for each reactant.
    ## B) We can add one vectorized interpolation whose dimension is the number of reactants.
    ## In this tutorial we opt for B, because the dofs are distributed per cell entity -- or
    ## to be specific for this tutorial, we use an isoparametric concept such that the nodes
    ## of our grid and the nodes of our solution approximation coincide. This way we can
    ## simply reshape the solution vector u into a matrix where the inner index
    ## corresponds to the index of the reactant. Note that we will still use the scalar
    ## interpolation for the assembly procedure.
    dh = DofHandler(grid)
    add!(dh, :reactants, ip^2)
    close!(dh)

    ## We can save some memory by telling the sparsity pattern that the matrices are not coupled.
    M = allocate_matrix(dh; coupling = [true false; false true])
    D = allocate_matrix(dh; coupling = [true false; false true])
    assemble_matrices!(M, D, cellvalues, dh, material)
    D .*= -1

    u₀ = zeros(ndofs(dh))
    setup_initial_conditions!(u₀, cellvalues, dh)

    ## And prepare output for visualization. The whole time series is stored in
    ## a single temporal VTKHDF file, with the grid written only once.
    vtkhdf = VTKHDFGridFile("reactive-surface-os.vtkhdf", dh; temporal = true)

    ## Now we setup the SciML functions and problem.
    ## The diffusion right hand side is passed as a MatrixOperator, not as a closure:
    ## `islinearfunction` in OrdinaryDiffEqDifferentiation only recognizes the problem as
    ## linear if the right hand side itself is a linear SciMLOperator, and only then is the
    ## W-matrix kept (and its factorization reused) across steps of constant Δt.
    diffusion_function = ODEFunction(MatrixOperator(D); mass_matrix = M)
    reaction_function = ODEFunction(ReactionFunction(material))
    f = GenericSplitFunction((diffusion_function, reaction_function), (1:ndofs(dh), 1:ndofs(dh)))
    prob = OperatorSplittingProblem(f, u₀, (0.0, T))
    verbose = DEVerbosity(SciMLLogging.Minimal())
    adaptive = TreeOption(f, adaptive)
    integrator = init(prob, alg, dt = Δt₀; save_everystep = false, adaptive, verbose, abstol = TreeOption(f, 4e-4), reltol = TreeOption(f, 1e-3)) #, verbose = verbose)
    ## This is now the main solve loop.
    for (uₜ, t) in TimeChoiceIterator(integrator, 0.0:Δtvis:T)
        @info t, integrator.dt, norm(uₜ)
        write_timestep(vtkhdf, t) do vtk
            write_solution(vtk, dh, uₜ)
        end
    end
    close(vtkhdf)
    return integrator
end

## This parametrization gives the spot pattern shown in the gif above.
material = GrayScottMaterial(0.00016, 0.00008, 0.06, 0.062)

# Warmup to precompile the code
gray_scott_on_sphere_os(material, LieTrotterGodunov((ImplicitEuler(linsolve = LinearSolve.UMFPACKFactorization()), Euler())), 10.0, 10.0, 10.0, 0, false)
gray_scott_on_sphere_os(material, PalindromicPairLieTrotterGodunov((SDIRK2(linsolve = LinearSolve.UMFPACKFactorization()), Heun())), 10.0, 10.0, 10.0, 1, true)

# Now the real runs
@time gray_scott_on_sphere_os(material, LieTrotterGodunov((ImplicitEuler(linsolve = LinearSolve.UMFPACKFactorization()), Euler())), 1.0, 10.0, 32000.0, 3, false)
# about 170.626542 seconds / 3 minutes
@time gray_scott_on_sphere_os(material, PalindromicPairLieTrotterGodunov((SDIRK2(linsolve = LinearSolve.UMFPACKFactorization()), Heun())), 2.2, 10.0, 32000.0, 3, false)
# about 273.011054 seconds / 4.5 minutes
@time gray_scott_on_sphere_os(material, PalindromicPairLieTrotterGodunov((SDIRK2(linsolve = LinearSolve.UMFPACKFactorization()), Heun())), 0.1, 10.0, 32000.0, 3, true)
# about an hour :)
