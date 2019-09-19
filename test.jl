include("QubeEnv.jl")
include("controls.jl")
using Printf

function test_env(policy::Function; verbose=false, play=true)
    qube = QubeEnv()
    state = reset!(qube)
    qube.state[2] += π/2
    action = policy(state)
    for i in 1:2750
        state, reward, done, info = step!(qube, action)
        action = policy(state)

        if play
            render(qube)
        end
        if done
            reset!(qube)
        end
        if verbose
            @printf("θ=%.4f, α=%.4f, θ̇=%.4f, α̇=%.4f", state[1], state[2], state[3], state[4])
        end
    end
end


function main()
    policies = Dict(
        "none" => zero_policy,
        "zero" => zero_policy,
        "const" => constant_policy,
        "rand" => random_policy,
        "random" => random_policy,
        "sw" => square_wave_policy,
        "energy" => energy_control_policy,
        "energy" => energy_control_policy,
        "pd" => pd_control_policy,
        "hold" => pd_control_policy,
        "flip" => flip_and_hold_policy,
        "sw-hold" => square_wave_flip_and_hold_policy,
        "damp" => dampen_policy,
    )
    policy = policies[ARGS[1]]

    # begin_down = nothing
    # if length(ARGS) > 1
    #     begin_down = ARGS[2]
    # end
    test_env(policy)#, begin_down=begin_down)
end

main()