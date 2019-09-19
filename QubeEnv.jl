# using .Space: Box
include("definitions.jl")
include("qube_simulator.jl")
using Defer
import PyCall


const ACT_MAX = 3.0

mutable struct QubeEnv # <: AbstractEnv
    state::Vector{Float64}
    # observation_space::Box
    # action_space::Box
    frequency::Float64
    max_episode_steps::Int
    episode_steps::Int
    isdone::Bool
    qube::QubeSimulator
    renderer::PyCall.PyObject
end

function QubeEnv(;frequency::Int=250, batch_size::Int=2048)
    state = [0.0, 0.0, 0.0, 0.0]
    # observation_space = Box(-OBS_MAX, OBS_MAX)
    # action_space = Box(-ACT_MAX, ACT_MAX)
    frequency = frequency

    # Ensures that samples in episode are the same as batch size
    # Reset every batch_size steps (2048 ~= 8.192 seconds)
    max_episode_steps = batch_size
    episode_steps = 0
    isdone = true

    # Open the Qube
    # TODO: Check assumption: ODE integration should be ~ once per ms
    # integration_steps = Int(ceil(1000 / frequency))
    qube = QubeSimulator(
        # forward_model=forward_model_euler,
        # frequency=frequency,
        # integration_steps=integration_steps,
        # max_voltage=MAX_MOTOR_VOLTAGE,
    )

    # Create renderer python object
    # renderer = PyCall.py"QubeRenderer"(state[1], state[2], frequency)
    renderer = PyCall.py"Renderer"()

    return QubeEnv(
        state,
        # observation_space,
        # action_space,
        frequency,
        max_episode_steps,
        episode_steps,
        isdone,
        qube,
        renderer
    )
end

function step!(env::QubeEnv, action::Vector{Float64})
    _step!(env, action)
    state = _get_state(env)
    reward = reward_fn(env, state, action)

    env.episode_steps += 1
    done = false
    done |= env.episode_steps % env.max_episode_steps == 0
    done |= abs(env.state[1]) > (180 * π / 180) # abs(θ) > 180°
    env.isdone = done

    info = Dict("θ"=>env.state[1], "α"=>env.state[2], "θ̇"=>env.state[3], "α̇"=>env.state[4])
    return state, reward, env.isdone, info
end

function reset!(env::QubeEnv)
    env.episode_steps = 1
    action = [0.0]
    step!(env, action)
    return _get_state(env)
end

function render(env::QubeEnv)
    θ = env.state[1]
    α = env.state[2]
    PyCall.py"render"(env.renderer, θ, α)
end

function close(env::QubeEnv)
    PyCall.py"close"(env.renderer)
end

function _step!(env::QubeEnv, action::Vector{Float64})
    action = clamp.(action, -ACT_MAX, ACT_MAX)
    env.state = step!(env.qube, action)
end

function _get_state(env::QubeEnv)
    θ, α, θ̇, α̇ = env.state
    return [cos(θ), sin(θ), cos(α), sin(α), θ̇, α̇]
end

function reward_fn(env::QubeEnv, state::Vector{Float64}, action::Vector{Float64})
    if length(state) == 6
        # Get the angles
        cosθ, sinθ, cosα, sinα, θ̇, α̇ = state
        θ = atan(sinθ, cosθ)
        α = atan(sinα, cosα)
    else
        θ, α, θ̇, α̇ = state
    end

    if abs(α) < (20 * π / 180) && abs(θ) < (90 * π / 180)
        # Encourage α=0, θ=0
        # return 100 * (1 - 0.5 * (abs(α) + abs(θ))) + 1
        return 1 * (1 - 0.5 * (abs(α) + abs(θ)))
    else
        # return 1 - 0.5 * (abs(α) + abs(θ))
        return 0
    end
end

PyCall.py"""
import vpython as vp
import numpy as np

class QubeRenderer:
    def __init__(self, θ, α, frequency):
        self.frequency = frequency
        vp.scene.width, vp.scene.height = 1000, 600
        vp.scene.range = 0.25
        vp.scene.title = 'A rotary pendulum'
        base_w, base_h, base_d = 0.102, 0.101, 0.102  # width, height, & len of base
        rotor_d, rotor_h = 0.0202, 0.01  # height, diameter of the rotor platform
        rotary_top_l, rotary_top_d = (
            0.032,
            0.012,
        )  # height, diameter of the rotary top
        arm_l, arm_d = 0.085, 0.00325  # height, diameter of the arm
        self._pendulum_l, self._pendulum_d = (
            0.129,
            0.00475,
        )  # height, diameter of the pendulum
        arm_origin = vp.vec(0, 0, 0)
        self._rotary_top_origin = vp.vec(0, 0, -rotary_top_l / 2)
        rotor_origin = arm_origin - vp.vec(0, rotor_h + rotary_top_d / 2 - 0.0035, 0)
        base_origin = rotor_origin - vp.vec(0, base_h / 2, 0)
        def pendulum_origin(θ, arm_l=arm_l):
            x = arm_l * np.sin(θ)
            y = 0
            z = arm_l * np.cos(θ)
            return vp.vec(x, y, z)
        self._pendulum_origin = pendulum_origin
        def pendulum_axis(θ):
            x = np.sin(θ)
            y = 0
            z = np.cos(θ)
            return vp.vec(x, y, z)
        self._pendulum_axis = pendulum_axis
        base = vp.box(
            pos=base_origin,
            size=vp.vec(base_w, base_h, base_d),
            color=vp.vec(0.45, 0.45, 0.45),
        )
        rotor = vp.cylinder(
            pos=rotor_origin,
            axis=vp.vec(0, 1, 0),
            size=vp.vec(rotor_h, rotor_d, rotor_d),
            color=vp.color.yellow,
        )
        self._rotary_top = vp.cylinder(
            pos=self._rotary_top_origin,
            axis=vp.vec(0, 0, 1),
            size=vp.vec(rotary_top_l, rotary_top_d, rotary_top_d),
            color=vp.color.red,
        )
        self._rotary_top.rotate(angle=θ, axis=vp.vec(0, 1, 0), origin=arm_origin)
        self._arm = vp.cylinder(
            pos=arm_origin,
            axis=vp.vec(0, 0, 1),
            size=vp.vec(arm_l, arm_d, arm_d),
            color=vp.vec(0.7, 0.7, 0.7),
        )
        self._arm.rotate(angle=θ, axis=vp.vec(0, 1, 0), origin=arm_origin)
        self._pendulum = vp.cylinder(
            pos=self._pendulum_origin(θ),
            axis=vp.vec(0, 1, 0),
            size=vp.vec(self._pendulum_l, self._pendulum_d, self._pendulum_d),
            color=vp.color.red,
        )
        self._pendulum.rotate(
            angle=α, axis=self._pendulum_axis(θ), origin=self._pendulum_origin(θ)
        )
        self.θ, self.α = θ, α

def render(obj, θ, α):
    Δθ = θ - obj.θ
    obj.θ = θ
    obj._arm.rotate(angle=Δθ, axis=vp.vec(0, 1, 0), origin=vp.vec(0, 0, 0))
    obj._rotary_top.rotate(angle=Δθ, axis=vp.vec(0, 1, 0), origin=vp.vec(0, 0, 0))
    obj._pendulum.origin = obj._pendulum_origin(θ)
    obj._pendulum.pos = obj._pendulum_origin(θ)
    obj._pendulum.axis = vp.vec(0, 1, 0)
    obj._pendulum.size = vp.vec(
        obj._pendulum_l, obj._pendulum_d, obj._pendulum_d
    )
    obj._pendulum.rotate(
        angle=α, axis=obj._pendulum_axis(θ), origin=obj._pendulum_origin(θ)
    )
    vp.rate(obj.frequency)

def close(obj, *args, **kwargs):
    pass


#==============================================================================#
from gym.envs.classic_control import rendering

class Renderer():
    def __init__(self):
        self._viewer = None

def render(obj, theta, alpha, mode="human"):
    if obj._viewer is None:
        width, height = (640, 240)
        obj._viewer = rendering.Viewer(width, height)
        l, r, t, b = (2, -2, 0, 100)
        theta_poly = rendering.make_polygon([(l, b), (l, t), (r, t), (r, b)])
        l, r, t, b = (2, -2, 0, 100)
        alpha_poly = rendering.make_polygon([(l, b), (l, t), (r, t), (r, b)])
        theta_circle = rendering.make_circle(radius=100, res=64, filled=False)
        theta_circle.set_color(0.5, 0.5, 0.5)  # Theta is grey
        alpha_circle = rendering.make_circle(radius=100, res=64, filled=False)
        alpha_circle.set_color(0.8, 0.0, 0.0)  # Alpha is red
        theta_origin = (width / 2 - 150, height / 2)
        alpha_origin = (width / 2 + 150, height / 2)
        obj._theta_tx = rendering.Transform(translation=theta_origin)
        obj._alpha_tx = rendering.Transform(translation=alpha_origin)
        theta_poly.add_attr(obj._theta_tx)
        alpha_poly.add_attr(obj._alpha_tx)
        theta_circle.add_attr(obj._theta_tx)
        alpha_circle.add_attr(obj._alpha_tx)
        obj._viewer.add_geom(theta_poly)
        obj._viewer.add_geom(alpha_poly)
        obj._viewer.add_geom(theta_circle)
        obj._viewer.add_geom(alpha_circle)
    obj._theta_tx.set_rotation(theta + 3.1415926535897)
    obj._alpha_tx.set_rotation(alpha)
    return obj._viewer.render(return_rgb_array=mode == "rgb_array")

def close(obj):
    if obj._viewer:
        obj._viewer.close()
#==============================================================================#



##### Test the renderer in Python
# r = QubeRenderer(0, 0, 500)
# t,a = 3.14, 3.14
# render(r, t, a)
# for i in range(100000):
#     t = (t + 0.01) % 6.283185307179586
#     a = (a + 0.01) % 6.283185307179586
#     render(r, t, a)    
# close(r)

##### Test the renderer in Julia
# let
#     freq = 500
#     r = PyCall.py"QubeRenderer"(0, 0, 500)
#     t, a = 3.14, 3.14
#     PyCall.py"render"(r, t, a)
#     for i in 1:1000
#         t = (t + 0.01) % 2π
#         a = (a + 0.01) % 2π
#         PyCall.py"render"(r, t, a)
#     end
#     PyCall.py"close"(r)
# end
"""
