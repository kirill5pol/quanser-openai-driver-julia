

"State with values of θ, α, θ̇, α̇"
mutable struct State
    θ::Float64
    α::Float64
    θ̇::Float64
    α̇::Float64

    function State()
        new(0, 0, 0, 0)
    end
    function State(θ, α, θ̇, α̇)
        new(θ, α, θ̇, α̇)
    end
end

"State to be easier to learn (sin/cos of angles are <= 1) with values of cos(θ), 
sin(θ), cos(α), sin(α), θ̇, α̇"
mutable struct StateTrig
    cosθ::Float64
    sinθ::Float64
    cosα::Float64
    sinα::Float64
    θ̇::Float64
    α̇::Float64

    function StateTrig()
        new(1, 0, 1, 0, 0, 0)
    end
    function StateTrig(θ, α, θ̇, α̇)
        new(cos(θ), sin(θ), cos(α), sin(α), θ̇, α̇)
    end
    function StateTrig(cosθ, sinθ, cosα, sinα, θ̇, α̇)
        new(cosθ, sinθ, cosα, sinα, θ̇, α̇)
    end
end


function _convert_state(state::State)
    return state
end

function _convert_state(state::StateTrig)
    # Get the angles
    cosθ, sinθ, cosα, sinα, θ̇, α̇ = (
        state.cosθ, state.sinθ, state.cosα, state.sinα, state.θ̇, state.α̇
    )
    θ = atan(sinθ, cosθ)
    α = atan(sinα, cosα)
    return state
end


function _convert_state(state::Vector{Float64})
    if length(state) == 4
        return state
    elseif length(state) == 6
        # Get the angles
        cosθ, sinθ, cosα, sinα, θ̇, α̇ = state
        θ = atan(sinθ, cosθ)
        α = atan(sinα, cosα)
        return [θ, α, θ̇, α̇]
    else
        println("BAD STATE!!!")
    end
end


# No input
function zero_policy(state::Vector{Float64}; kwargs...)::Vector{Float64}
    return [0.0]
end


# Constant input
function constant_policy(state::Vector{Float64}; kwargs...)::Vector{Float64}
    return [3.0]
end


# Rand input
function random_policy(state::Vector{Float64}; kwargs...)::Vector{Float64}
    return [randn()]
end


# Square wave, switch every 85 ms
function square_wave_policy(state::Vector{Float64}, step; frequency=250, kwargs...)::Vector{Float64}
    steps_until_85ms = int(85 * (frequency / 300))
    state = _convert_state(state)
    # Switch between positive and negative every 85 ms
    mod_170ms = step % (2 * steps_until_85ms)
    if mod_170ms < steps_until_85ms
        action = 3.0
    else
        action = -3.0
    end
    return [action]
end


# Flip policy
function energy_control_policy(state::Vector{Float64}; kwargs...)::Vector{Float64}
    state = _convert_state(state)
    # Run energy-based control to flip up the pendulum
    theta, alpha, theta_dot, alpha_dot = state
    # alpha_dot += alpha_dot + 1e-15

    # """Implements a energy based swing-up controller"""
    mu = 50.0  # in m/s/J
    ref_energy = 30.0 / 1000.0  # Er in joules

    # TODO: Which one is correct?
    max_u = 6  # Max action is 6m/s^2
    # max_u = 0.85  # Max action is 6m/s^2

    # System parameters
    jp = 3.3282e-5
    lp = 0.129
    lr = 0.085
    mp = 0.024
    mr = 0.095
    rm = 8.4
    g = 9.81
    kt = 0.042

    pend_torque = (1 / 2) * mp * g * lp * (1 + cos(alpha))
    energy = pend_torque + (jp / 2.0) * alpha_dot * alpha_dot

    u = mu * (energy - ref_energy) * sign(-1 * cos(alpha) * alpha_dot)
    # u = clamp.(u, -max_u, max_u)

    torque = (mr * lr) * u
    voltage = (rm / kt) * torque
    println(voltage)
    voltage = clamp.(voltage, -3.0, 3.0)
    return [-voltage]
end


# Hold policy
function pd_control_policy(state::Vector{Float64}; kwargs...)::Vector{Float64}
    state = _convert_state(state)
    theta, alpha, theta_dot, alpha_dot = state
    # multiply by proportional and derivative gains
    kp_theta = -2.0
    kp_alpha = 35.0
    kd_theta = -1.5
    kd_alpha = 3.0

    # If pendulum is within 20 degrees of upright, enable balance control, else zero
    if abs(alpha) <= (20.0 * π / 180.0)
        action = (
            theta * kp_theta
            + alpha * kp_alpha
            + theta_dot * kd_theta
            + alpha_dot * kd_alpha
        )
    else
        action = 0.0
    end
    action = clamp.(action, -3.0, 3.0)
    return [action]
end


# Flip and Hold
function flip_and_hold_policy(state::Vector{Float64}; kwargs...)::Vector{Float64}
    state = _convert_state(state)
    theta, alpha, theta_dot, alpha_dot = state

    # If pendulum is within 20 degrees of upright, enable balance control
    if abs(alpha) <= (20.0 * π / 180.0)
        action = pd_control_policy(state)
        println("pd_control_policy")
    else
        action = energy_control_policy(state)
        println("energy_control_policy")

    end
    return action
end


# Square wave instead of energy controller flip and hold
function square_wave_flip_and_hold_policy(state::Vector{Float64}; kwargs...)::Vector{Float64}
    state = _convert_state(state)
    theta, alpha, theta_dot, alpha_dot = state

    # If pendulum is within 20 degrees of upright, enable balance control
    if abs(alpha) <= (20.0 * π / 180.0)
        action = pd_control_policy(state)
    else
        action = square_wave_policy(state, kwargs=kwargs)
    end
    return action
end


function dampen_policy(state::Vector{Float64}; kwargs...)::Vector{Float64}
    state = _convert_state(state)
    theta, alpha, theta_dot, alpha_dot = state

    if (abs(alpha) > (20.0 * π / 180.0)) && (abs(theta) < (π / 4))
        kp_theta = -2
        kp_alpha = 35
        kd_theta = -1.5
        kd_alpha = 3
        if alpha >= 0
            action = (
                -theta * kp_theta
                + (π - alpha) * kp_alpha
                + -theta_dot * kd_theta
                + -alpha_dot * kd_alpha
            )
        else
            action = (
                -theta * kp_theta
                + (-π - alpha) * kp_alpha
                + -theta_dot * kd_theta
                + -alpha_dot * kd_alpha
            )
        end
    else
        action = 0
    end

    action = clamp.(action, -3.0, 3.0)
    return [action]
end


function dampen_alt_policy(state::Vector{Float64}; kwargs...)::Vector{Float64}
    state = _convert_state(state)
    theta, alpha, theta_dot, alpha_dot = state

    if abs(alpha) > (20.0 * π / 180.0)  # && abs(theta) < (π / 4)
        kp_theta, kp_alpha, kd_theta, kd_alpha = [-1.99, 41.7, -1.81, 3.11]

        if alpha >= 0
            action = (
                -theta * kp_theta
                + (π - alpha) * kp_alpha
                + -theta_dot * kd_theta
                + -alpha_dot * kd_alpha
            )
        else
            action = (
                -theta * kp_theta
                + (-π - alpha) * kp_alpha
                + -theta_dot * kd_theta
                + -alpha_dot * kd_alpha
            )
        end
    else
        action = 0
    end

    action = clamp.(action, -3.0, 3.0)
    return [action]
end
