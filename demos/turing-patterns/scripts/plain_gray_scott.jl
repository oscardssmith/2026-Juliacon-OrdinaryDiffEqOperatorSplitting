
include(joinpath(@__DIR__, "..", "src", "common.jl"))

function gray_scott_on_sphere_plain(material::GrayScottMaterial, Δt::Real, T::Real, refinements::Integer)
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

    ## Now we setup buffers for the time dependent solution and fill the initial condition.
    uₜ = zeros(ndofs(dh))
    uₜ₋₁ = ones(ndofs(dh))
    setup_initial_conditions!(uₜ₋₁, cellvalues, dh)

    ## And prepare output for visualization. The whole time series is stored in
    ## a single temporal VTKHDF file, with the grid written only once.
    vtkhdf = VTKHDFGridFile("reactive-surface.vtkhdf", dh; temporal = true)
    write_timestep(vtkhdf, 0.0) do vtk
        write_solution(vtk, dh, uₜ₋₁)
    end

    ## This is now the main solve loop.
    ## Since the heat problem is linear and has no time dependent parameters, we precompute the
    ## decomposition of the system matrix to speed up the linear system solver.
    @time begin
        W = M + Δt .* D
        cholW = cholesky(W)
        for (iₜ, t) in enumerate(Δt:Δt:T)
            @info t
            ## First we solve the heat problem
            uₜ .= cholW \ (M * uₜ₋₁)

            ## Then we solve the point-wise reaction problem with the solution of
            ## the heat problem as initial guess. 2 is the number of reactants.
            num_individual_reaction_dofs = ndofs(dh) ÷ 2
            rvₜ = reshape(uₜ, (2, num_individual_reaction_dofs))
            for i in 1:num_individual_reaction_dofs
                r₁ = rvₜ[1, i]
                r₂ = rvₜ[2, i]
                du = reaction_kernel((r₁, r₂), material, t)
                rvₜ[1, i] += Δt * du[1]
                rvₜ[2, i] += Δt * du[2]
            end

            ## The solution is then stored every 10th step to the VTKHDF file for
            ## later visualization purposes.
            if (iₜ % 10) == 0
                write_timestep(vtkhdf, t) do vtk
                    write_solution(vtk, dh, uₜ)
                end
            end

            ## Finally we rotate the solution to initialize the next timestep.
            uₜ₋₁ .= uₜ
        end
        close(vtkhdf)
    end
    return uₜ
end


## This parametrization gives the spot pattern shown in the gif above.
material = GrayScottMaterial(0.00016, 0.00008, 0.06, 0.062)
gray_scott_on_sphere_plain(material, 10.0, 10.0, 0) # Warmup
gray_scott_on_sphere_plain(material, 10.0, 32000.0, 3)
