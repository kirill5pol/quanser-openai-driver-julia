import DifferentialEquations
de = DifferentialEquations
include("definitions.jl")


function diff_forward_model_ode!(du, u, p, t)
    # u = [θ, α, θ̇, α̇]
    # du = [θ̇, α̇, θ̈, α̈]
    # p = (action, phys_params)
    # t = (0, dt)
    Vm, pp = p
    tau = -(pp.km * (Vm - pp.km * u[3])) / pp.Rm  # torque

    du[1] = u[3]
    du[2] = u[4]
    # From Rotary Pendulum Workbook
    du[3] = (-pp.Lp*pp.Lr*pp.mp*(-8.0*pp.Dp*u[4] + pp.Lp^2*pp.mp*u[3]^2*sin(2.0*u[2]) + 4.0*pp.Lp*pp.g*pp.mp*sin(u[2]))*cos(u[2]) + (4.0*pp.Jp + pp.Lp^2*pp.mp)*(4.0*pp.Dr*u[3] + pp.Lp^2*u[4]*pp.mp*u[3]*sin(2.0*u[2]) + 2.0*pp.Lp*pp.Lr*u[4]^2*pp.mp*sin(u[2]) - 4.0*tau))/(4.0*pp.Lp^2*pp.Lr^2*pp.mp^2*cos(u[2])^2 - (4.0*pp.Jp + pp.Lp^2*pp.mp)*(4.0*pp.Jr + pp.Lp^2*pp.mp*sin(u[2])^2 + 4.0*pp.Lr^2*pp.mp))
    du[4] = (2.0*pp.Lp*pp.Lr*pp.mp*(4.0*pp.Dr*u[3] + pp.Lp^2*u[4]*pp.mp*u[3]*sin(2.0*u[2]) + 2.0*pp.Lp*pp.Lr*u[4]^2*pp.mp*sin(u[2]) - 4.0*tau)*cos(u[2]) - 0.5*(4.0*pp.Jr + pp.Lp^2*pp.mp*sin(u[2])^2 + 4.0*pp.Lr^2*pp.mp)*(-8.0*pp.Dp*u[4] + pp.Lp^2*pp.mp*u[3]^2*sin(2.0*u[2]) + 4.0*pp.Lp*pp.g*pp.mp*sin(u[2])))/(4.0*pp.Lp^2*pp.Lr^2*pp.mp^2*cos(u[2])^2 - (4.0*pp.Jp + pp.Lp^2*pp.mp)*(4.0*pp.Jr + pp.Lp^2*pp.mp*sin(u[2])^2 + 4.0*pp.Lr^2*pp.mp))
    
    return du
end


function forward_model_ode(
    θ::Float64, α::Float64, θ̇::Float64, α̇::Float64,
    Vm::Float64,
    dt::Float64,
    p::PhysicalParameters,
    integration_steps::Int
)
    u0 = [θ, α, θ̇, α̇]
    p = (Vm,  PhysicalParameters())
    tspan = (0, dt)

    prob = de.ODEProblem(diff_forward_model_ode!,u0,tspan,p)
    θ, α, θ̇, α̇ = de.solve(prob)[end]

    θ = rem2pi(θ, RoundNearest)
    α = rem2pi(α, RoundNearest)

    θ, α, θ̇, α̇
end


function forward_model_ode!(
    state::Vector{Float64},
    action::Vector{Float64},
    dt::Float64,
    p::PhysicalParameters,
    integration_steps::Int
)
    u0 = state
    p = (action[1],  PhysicalParameters())
    tspan = (0, dt)

    prob = de.ODEProblem(diff_forward_model_ode!,u0,tspan,p)
    state = de.solve(prob)[end]

    state[1] = rem2pi(state[1], RoundNearest)
    state[2] = rem2pi(state[2], RoundNearest)

    state
end


function forward_model_euler(
    θ::Float64, α::Float64, θ̇::Float64, α̇::Float64,
    Vm::Float64,
    dt::Float64,
    p::PhysicalParameters,
    integration_steps::Int
)
    dt /= integration_steps
    for step in range(1, length=integration_steps)
        tau = -(p.km * (Vm - p.km * θ̇)) / p.Rm  # torque

        # From Rotary Pendulum Workbook
        θ̈::Float64 = (-p.Lp*p.Lr*p.mp*(-8.0*p.Dp*α̇ + p.Lp^2*p.mp*θ̇^2*sin(2.0*α) + 4.0*p.Lp*p.g*p.mp*sin(α))*cos(α) + (4.0*p.Jp + p.Lp^2*p.mp)*(4.0*p.Dr*θ̇ + p.Lp^2*α̇*p.mp*θ̇*sin(2.0*α) + 2.0*p.Lp*p.Lr*α̇^2*p.mp*sin(α) - 4.0*tau))/(4.0*p.Lp^2*p.Lr^2*p.mp^2*cos(α)^2 - (4.0*p.Jp + p.Lp^2*p.mp)*(4.0*p.Jr + p.Lp^2*p.mp*sin(α)^2 + 4.0*p.Lr^2*p.mp))
        α̈::Float64 = (2.0*p.Lp*p.Lr*p.mp*(4.0*p.Dr*θ̇ + p.Lp^2*α̇*p.mp*θ̇*sin(2.0*α) + 2.0*p.Lp*p.Lr*α̇^2*p.mp*sin(α) - 4.0*tau)*cos(α) - 0.5*(4.0*p.Jr + p.Lp^2*p.mp*sin(α)^2 + 4.0*p.Lr^2*p.mp)*(-8.0*p.Dp*α̇ + p.Lp^2*p.mp*θ̇^2*sin(2.0*α) + 4.0*p.Lp*p.g*p.mp*sin(α)))/(4.0*p.Lp^2*p.Lr^2*p.mp^2*cos(α)^2 - (4.0*p.Jp + p.Lp^2*p.mp)*(4.0*p.Jr + p.Lp^2*p.mp*sin(α)^2 + 4.0*p.Lr^2*p.mp))

        θ̇ += θ̈ * dt
        α̇ += α̈ * dt

        θ += θ̇ * dt
        α += α̇ * dt

        θ = rem2pi(θ, RoundNearest)
        α = rem2pi(α, RoundNearest)
    end

    θ, α, θ̇, α̇
end


function forward_model_euler(
    state::Vector{Float64},
    action::Vector{Float64},
    dt::Float64,
    p::PhysicalParameters,
    integration_steps::Int
)
    θ, α, θ̇, α̇ = state
    Vm = action[1]
    θ, α, θ̇, α̇ = forward_model_euler(θ, α, θ̇, α̇, Vm, dt, p, integration_steps)
    return [θ, α, θ̇, α̇]
end


mutable struct QubeSimulator
    state::Vector{Float64}
    δt::Float64
    forward_model::Function
    physparams::PhysicalParameters
    integration_steps::Int
    max_voltage::Float64

    # function QubeSimulator(
        # forward_model::Function; 
        # frequency::Float64=250, integration_steps::Int=1, max_voltage::Float64=18.0)
        # δt = 1.0 / frequency
        # physparams = PhysicalParameters()
        # integration_steps = integration_steps
        # max_voltage = max_voltage
        # state = [0, 0, 0, 0] + randn(4) * 0.01
        # new(state, δt, forward_model, physparams, integration_steps, max_voltage)
    # end

    function QubeSimulator()
        frequency = 250
        δt = 1.0 / frequency
        max_voltage = 18.0
        integration_steps = 1
        physparams = PhysicalParameters()
        forward_model = forward_model_ode!

        state = [0, 0, 0, 0] + randn(4) * 0.01
        new(state, δt, forward_model, physparams, integration_steps, max_voltage)
    end
end

function step!(sim::QubeSimulator, action::Vector{Float64})
    action = clamp.(action, -sim.max_voltage, sim.max_voltage)
    sim.state = sim.forward_model(
        sim.state, action, sim.δt, sim.physparams, sim.integration_steps
    )

    return sim.state
end

function reset_physparams!(sim::QubeSimulator)
    sim.physparams = SampledPhysicalParameters()
end

function reset_up!(sim::QubeSimulator)
    sim.state = [0, 0, 0, 0] + randn(4) * 0.01
    return sim.state
end

function reset_down!(sim::QubeSimulator)
    sim.state = [0, π, 0, 0] + randn(4) * 0.01
    return sim.state
end
