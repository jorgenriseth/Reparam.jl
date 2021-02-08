struct FourierVectorBasisFunction <: BasisFunction
    k::Int
    l::Int
    label::String
    direction::String

    function FourierVectorBasisFunction(k::Int, l::Int, label::String, direction::String)
        @assert (label in ["sin", "sin-sin", "sin-cos"]) "label must be of type 'sin', 'sin-sin' or 'sin-cos'."
        @assert (direction in ["x", "y"]) "direction must be 'x' or 'y'."
        new(k, l, label, direction)
    end
end


function (v::FourierVectorBasisFunction)(x)
    if v.direction == "x"
        if v.label == "sin"
            return [sqrt(2) * sin(π * v.k * x[1]), 0.]
        elseif v.label =="sin-sin"
            return [2 * sin(π * v.k * x[1]) * sin(2π * v.l * x[2]), 0.]
        else
            return [2 * sin(π * v.k * x[1]) * cos(2π * v.l * x[2]), 0.]
        end
    else 
        if v.label == "sin"
            return [0., sqrt(2) * sin(π * v.k * x[2])]
        elseif v.label =="sin-sin"
            return [0., 2 * sin(π * v.k * x[2]) * sin(2π * v.l * x[1])]
        else
            return [0., 2 * sin(π * v.k * x[2]) * cos(2π * v.l * x[1])]
        end
    end
    return error("Invalid configuration in basisfunction")
end


function FourierVectorBasis(N)
    basis = FourierVectorBasisFunction[]
    for k in 1:N
        push!(basis, FourierVectorBasisFunction(k, 1, "sin", "x"))
        push!(basis, FourierVectorBasisFunction(k, 1, "sin", "y"))
        for l in 1:N
            push!(basis, FourierVectorBasisFunction(k, l, "sin", "x"))
            push!(basis, FourierVectorBasisFunction(k, l, "sin", "y"))
            push!(basis, FourierVectorBasisFunction(k, l, "sin-sin", "x"))
            push!(basis, FourierVectorBasisFunction(k, l, "sin-sin", "y"))
            push!(basis, FourierVectorBasisFunction(k, l, "sin-cos", "x"))
            push!(basis, FourierVectorBasisFunction(k, l, "sin-cos", "y"))
        end
    end
    return basis
end