using Ferrite, FerriteGmsh
using BlockArrays, SparseArrays, LinearAlgebra, VTKHDF
using Random

# Adopted from Ferrite.jl Reactive Surface tutorial

struct GrayScottMaterial{T}
    D₁::T
    D₂::T
    F::T
    k::T
end

function reaction_kernel(u, p::GrayScottMaterial, t)
    r₁, r₂ = u
    return (
        (-r₁ * r₂^2 + p.F * (1 - r₁)),
        (r₁ * r₂^2 - r₂ * (p.F + p.k))
    )
end

function assemble_element_mass!(Me::Matrix, cellvalues::CellValues)
    n_basefuncs = getnbasefunctions(cellvalues)
    num_reactants = 2
    r₁range = 1:num_reactants:(num_reactants * n_basefuncs)
    r₂range = 2:num_reactants:(num_reactants * n_basefuncs)
    Me₁ = @view Me[r₁range, r₁range]
    Me₂ = @view Me[r₂range, r₂range]
    ## Loop over quadrature points
    for q_point in 1:getnquadpoints(cellvalues)
        ## Get the quadrature weight
        dΩ = getdetJdV(cellvalues, q_point)
        ## Loop over test shape functions
        for i in 1:n_basefuncs
            δuᵢ = shape_value(cellvalues, q_point, i)
            ## Loop over trial shape functions
            for j in 1:n_basefuncs
                δuⱼ = shape_value(cellvalues, q_point, j)
                ## Add contribution to Ke
                Me₁[i, j] += (δuᵢ * δuⱼ) * dΩ
                Me₂[i, j] += (δuᵢ * δuⱼ) * dΩ
            end
        end
    end
    return nothing
end

function assemble_element_diffusion!(De::Matrix, cellvalues::CellValues, material::GrayScottMaterial)
    n_basefuncs = getnbasefunctions(cellvalues)
    D₁ = material.D₁
    D₂ = material.D₂
    ## The diffusion between the reactions is not coupled, so we get a blocked-strided matrix.
    num_reactants = 2
    r₁range = 1:num_reactants:(num_reactants * n_basefuncs)
    r₂range = 2:num_reactants:(num_reactants * n_basefuncs)
    De₁ = @view De[r₁range, r₁range]
    De₂ = @view De[r₂range, r₂range]
    ## Loop over quadrature points
    for q_point in 1:getnquadpoints(cellvalues)
        ## Get the quadrature weight
        dΩ = getdetJdV(cellvalues, q_point)
        ## Loop over test shape functions
        for i in 1:n_basefuncs
            ∇δuᵢ = shape_gradient(cellvalues, q_point, i)
            ## Loop over trial shape functions
            for j in 1:n_basefuncs
                ∇δuⱼ = shape_gradient(cellvalues, q_point, j)
                ## Add contribution to Ke
                De₁[i, j] += D₁ * (∇δuᵢ ⋅ ∇δuⱼ) * dΩ
                De₂[i, j] += D₂ * (∇δuᵢ ⋅ ∇δuⱼ) * dΩ
            end
        end
    end
    return nothing
end

function assemble_matrices!(M::SparseMatrixCSC, D::SparseMatrixCSC, cellvalues::CellValues, dh::DofHandler, material::GrayScottMaterial)
    n_basefuncs = getnbasefunctions(cellvalues)

    ## Allocate the element stiffness matrix and element force vector
    Me = zeros(2 * n_basefuncs, 2 * n_basefuncs)
    De = zeros(2 * n_basefuncs, 2 * n_basefuncs)

    ## Create an assembler
    M_assembler = start_assemble(M)
    D_assembler = start_assemble(D)
    ## Loop over all cells
    for cell in CellIterator(dh)
        ## Reinitialize cellvalues for this cell
        Ferrite.reinit!(cellvalues, cell)
        fill!(Me, 0)
        fill!(De, 0)
        ## Compute element contribution
        assemble_element_mass!(Me, cellvalues)
        assemble!(M_assembler, celldofs(cell), Me)

        assemble_element_diffusion!(De, cellvalues, material)
        assemble!(D_assembler, celldofs(cell), De)
    end
    return nothing
end;

function setup_initial_conditions!(
    u₀::Vector, cellvalues::CellValues, dh::DofHandler,
    nseeds::Int = 1; radius = 0.15, rng = MersenneTwister(7)
)
    u₀ .= 1.0
    u₀[2:2:end] .= 0.0
    centers = [normalize(Vec{3}(randn(rng, 3))) for _ in 1:nseeds]

    n_basefuncs = getnbasefunctions(cellvalues)
    for cell in CellIterator(dh)
        Ferrite.reinit!(cellvalues, cell)
        coords = getcoordinates(cell)
        rv₀ₑ = reshape(@view(u₀[celldofs(cell)]), (2, n_basefuncs))
        for i in 1:n_basefuncs
            x = normalize(coords[i])
            if any(c -> acos(clamp(x ⋅ c, -1, 1)) < radius, centers)
                rv₀ₑ[1, i] = 0.5
                rv₀ₑ[2, i] = 0.25
            end
        end
    end
    u₀ .+= 0.01 * rand(ndofs(dh))
    return
end

function create_embedded_sphere(refinements)
    gmsh.initialize()

    ## Add a unit sphere in 3D space
    gmsh.model.occ.addSphere(0.0, 0.0, 0.0, 1.0)
    gmsh.model.occ.synchronize()

    ## Generate nodes and surface elements only, hence we need to pass 2 into generate
    gmsh.model.mesh.generate(2)

    ## To get good solution quality refine the elements several times
    for _ in 1:refinements
        gmsh.model.mesh.refine()
    end

    ## Now we create a Ferrite grid out of it. Note that we also call toelements
    ## with our surface element dimension to obtain these.
    nodes = tonodes()
    elements, _ = toelements(2)
    gmsh.finalize()
    return Grid(elements, nodes)
end
