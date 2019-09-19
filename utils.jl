import PyCall
import PyPlot
plt = PyPlot

import("definitions.jl")

PyCall.py"""

import matplotlib.pyplot as plt
import numpy as np

def plot_results(hists, labels, colors=None):
    state_dims = ['Theta', 'Alpha', 'Theta dot', 'Alpha dot', 'Action']
    
    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True)
    for i, ax in enumerate((ax1, ax2, ax3, ax4, ax5)):
        if colors is None:
            for hist, label in zip(hists, labels):
                ax.plot(hist[:,i], label=label)
        else:
            for hist, label, color in zip(hists, labels, colors):
                ax.plot(hist[:,i], label=label, color=color)
        ax.set_ylabel(state_dims[i])
        ax.legend()
    plt.show()
"""


function plot_results(hists, labels, colors=Nothing)
    state_dims = ["Theta", "Alpha", "Theta dot", "Alpha dot", "Action"]
    n_axes = length(state_dims)
    
    f, fig_axes = plt.subplots(n_axes, 1, sharex=true)
    for (i, ax) in enumerate(fig_axes)
        if colors == Nothing
            for (hist, label) in zip(hists, labels)
                ax.plot(hist[:,i], label=label)
            end
        else
            for (hist, label, color) in zip(hists, labels, colors)
                ax.plot(hist[:,i], label=label, color=color)
            end
        end
        ax.set_ylabel(state_dims[i])
        ax.legend()
    end
    plt.show()
end


function run_sim(nsteps=10000, plot=false, verbose=false)
    θ, α, θ̇, α̇ = 0.1, -0.01, 0.01, 0.01
    Vm = 0.0
    dt = 1.0 / 1000
    i_step = 3
    # nsteps = 1000000

    state_hist = Matrix{Float}(undef, 4, nsteps)
    p = PhysicalParameters()

    for t in 1:nsteps
        θ, α, θ̇, α̇ = forward_model_euler(θ, α, θ̇, α̇, Vm, dt, p, i_step)
        state_hist[:, t] .= θ, α, θ̇, α̇
        if verbose println([θ, α, θ̇, α̇] * (180/π)); end
    end

    if plot plot_results([state_hist'], ["First run"]); end

end