using Test
using Reparam
using ForwardDiff: derivative

function circle()
    γopt(t) = 0.5 * log(20t+1) / log(21) + 0.25 * (1 + tanh(20(t-0.5)) / tanh(10))
    s1(t) = [cos(2π*t), sin(2π*t)] # Circle
    s2(t) = s1(γopt(t))

    # Q-transforms (Want to reparametrize s1 to match s2)
    r = Qmap(s1)
    q = Qmap(s2)

    myintegrator = Reparam.GaussLegendre(128)
    proj = FourierProjector(10, I=myintegrator)
    @time res = reparametrize(q, r, proj, maxiter=200, I=myintegrator, lsconfig=BacktrackConfig(c=0.8, verbose=true), α_step=0.4, rtol=1e-6, gtol=1e-6);
    return
end

@test circle() == nothing  # Check that code runs without errors
