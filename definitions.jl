
function normal(μ::Float64, σ::Float64)
    return σ * randn() + μ
end

struct PhysicalParameters
    # Motor
    Rm::Float64  # Resistance
    kt::Float64  # Current-torque (N-m/A)
    km::Float64  # Back-emf constant (V-s/rad)

    # Rotary Arm
    mr::Float64  # Mass (kg)
    Lr::Float64  # Total length (m)
    Jr::Float64  # Moment of inertia about pivot (kg-m^2)
    Dr::Float64  # Equivalent viscous damping coefficient (N-m-s/rad)

    # Pendulum Link
    mp::Float64  # Mass (kg)
    Lp::Float64  # Total length (m)
    Jp::Float64  # Moment of inertia about pivot (kg-m^2)
    Dp::Float64  # Equivalent viscous damping coefficient (N-m-s/rad)

    g::Float64  # Gravity constant
end

function PhysicalParameters()
    Rm = 8.4
    kt = 0.042
    km = 0.042
    mr = 0.095
    Lr = 0.085
    Dr = 0.00027
    mp = 0.024
    Lp = 0.129
    Dp = 0.00005
    g  = 9.81

    Jr = mr * Lr ^ 2 / 12
    Jp = mp * Lp ^ 2 / 12
    return PhysicalParameters(Rm, kt, km, mr, Lr, Jr, Dr, mp, Lp, Jp, Dp, g)
end

function SampledPhysicalParameters()
    Rm = 8.4
    kt = 0.042
    km = 0.042
    mr = 0.095
    Lr = 0.085
    Dr = 0.00027
    mp = 0.024
    Lp = 0.129
    Dp = 0.00005
    g = 9.81

    Rm = normal(Rm, 0.05 * Rm)
    kt = normal(kt, 0.05 * kt)
    km = normal(km, 0.05 * km)
    mr = normal(mr, 0.05 * mr)
    Lr = normal(Lr, 0.05 * Lr)
    Dr = normal(Dr, 0.05 * Dr)
    mp = normal(mp, 0.05 * mp)
    Lp = normal(Lp, 0.05 * Lp)
    Dp = normal(Dp, 0.05 * Dp)
    g =  normal(g,  0.05 * g)

    Jr = mr * Lr ^ 2 / 12
    Jp = mp * Lp ^ 2 / 12

    return PhysicalParameters(Rm, kt, km, mr, Lr, Jr, Dr, mp, Lp, Jp, Dp, g)
end