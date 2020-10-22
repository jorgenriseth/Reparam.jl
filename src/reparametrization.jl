using LinearAlgebra: ⋅

function l2_gradient(q, r)
    # Get the derivatives of the curves q, r, and compute the gradient of the cost function
    qdt(t) = ForwardDiff.derivative(q, t)
    rdt(t) = ForwardDiff.derivative(r, t)
    function (t)
        return r(t)⋅qdt(t) - q(t)⋅rdt(t)  # /diff_norm
    end
end

function accumulate_composition(funcvec)
    n = length(funcvec)
    out = Vector{Function}(undef, n)
    out[1] = funcvec[1]
    for i in 2:n
        out[i] = out[i-1] ∘ funcvec[i]
    end
    return out
end


function relative_error(f1, f0)
    return abs((f1 - f0) / f0)
end


function printheader(;width=82)
    println(" " * "-"^width)
    @printf "| %-4s | %-16s | %-16s | %-16s | %-16s | %-16s | \n" "Iter" "Error" "εmax" "εi" "||dγ||" "|f_i - f_{i-1}|/ |fi|"
    println("-"^width)
end

function printfirst(error)
    @printf "| %4d | %.10e | %-16s | %-16s | %-16s |\n" 0 error "" "" ""
end

function printline(iter, error, max_step, step, gradnorm, relerror)
    @printf "| %4d | %.10e | %.10e | %.10e | %.10e | %.10e |\n" iter error max_step step gradnorm relerror
end

function printfooter(;width=82)
    println(" " * "-"^width)
end
