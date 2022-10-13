"""
    Element{TF}

Composite type that defines a beam element's properties

# Fields
 - `L`: Beam element length
 - `x`: Beam element location
 - `compliance`: Beam element compliance matrix
 - `mass`: Beam element mass matrix
 - `Cab`: Transformation matrix from the undeformed beam element frame to the body frame
 - `mu`: Beam element damping coefficients
"""
struct Element{TF}
    L::TF
    x::SVector{3,TF}
    compliance::SMatrix{6,6,TF,36}
    mass::SMatrix{6,6,TF,36}
    Cab::SMatrix{3,3,TF,9}
    mu::SVector{6,TF}
end

"""
    Element(L, x, compliance, mass, Cab, mu)

Construct a beam element

# Arguments
- `L`: Length of the beam element
- `x`: Location of the beam element (the center of the beam element)
- `compliance`: Beam element compliance matrix
- `mass`: Beam element mass matrix
- `Cab`: Transformation matrix from the undeformed beam element frame to the body frame
- `mu`: Beam element damping coefficients
"""
function Element(L, x, compliance, mass, Cab, mu)
    TF = promote_type(typeof(L), eltype(x), eltype(compliance), eltype(mass), eltype(Cab), eltype(mu))
    return Element{TF}(L, x, compliance, mass, Cab, mu)
end
"""
    Joint

Composite type that defines a joint. A joint ties DoFs between two nodes.

# Fields
 - `pt1`: Point 1 to be joined
 - `pt2`: Point 2 to be joined
 - `frame`: Transformation matrix from body frame to the joint frame
 - `jt_flag`: Vector of flags defining whether each of the 6 DOFs are joined.
 - `ux`: Float64, x distance between points in joint frame
 - `uy`: Float64, y distance between points in joint frame
 - `uz`: Float64, z distance between points in joint frame
 - `rx`: Float64, x angle between points in joint frame
 - `ry`: Float64, y angle between points in joint frame
 - `rz`: Float64, z angle between points in joint frame
"""
struct Joint
    pt1::Integer
    pt2::Integer
    frame::SMatrix{3,3,Float64,9}
    jt_flag::SVector{6, Bool}
    ux::Float64
    uy::Float64
    uz::Float64
    rx::Float64
    ry::Float64
    rz::Float64
end
"""
    Joint(pt1, pt2; kwargs...)

Construct a joint between pt1 and pt2. A joint defines specific translational/rotational
DoFs to be equivalent, or have some prescribed distance/angle between them 
e.g. a joint rotation. Typically, points are initially disconnected and coincident,
although this is not enforced. The default frame for defining equivalent DoFs is the
body frame. User-specified joint frames can also be defined.

# Arguments
 - `pt1`: Point ID 1 to be tied.
 - `pt2`: Point ID 2 to be tied.

# Keyword Arguments
 - `frame`: 3 x 3 transformation matrix from undeformed joint frame to body frame. 
        Defaults to the Identity matrix i.e. joint frame == body frame.
 - `ux`: Prescribed x distance between points in joint frame, 0 = coincident.
 - `uy`: Prescribed y distance between points in joint frame, 0 = coincident.
 - `uz`: Prescribed z distance between points in joint frame, 0 = coincident.
 - `rx`: Prescribed x angle between points in joint frame, 0 = coincident.
 - `ry`: Prescribed y angle between points in joint frame, 0 = coincident.
 - `rz`: Prescribed z angle between points in joint frame, 0 = coincident.
"""
function Joint(pt1, pt2;
    frame = I3,
    ux = nothing,
    uy = nothing,
    uz = nothing,
    rx = nothing,
    ry = nothing,
    rz = nothing)
    if isnothing(ux)
        ux = NaN
        xf = false
    else
        xf = true
    end
    if isnothing(uy)
        uy = NaN
        yf = false
    else
        yf = true
    end
    if isnothing(uz)
        uz = NaN
        zf = false
    else
        zf = true
    end
    if isnothing(rx)
        rx = NaN
        rxf = false
    else
        rxf = true
    end
    if isnothing(ry)
        ry = NaN
        ryf = false
    else
        ryf = true
    end
    if isnothing(rz)
        rz = NaN
        rzf = false
    else
        rzf = true
    end
    jt_flag = SVector(xf, yf, zf, rxf, ryf, rzf)

    return Joint(pt1, pt2, frame, jt_flag, ux, uy, uz, rx, ry, rz)
end


"""
    Assembly{TF, TP<:AbstractVector{<:AbstractVector{TF}},
        TC<:AbstractVector{<:Integer}, TE<:AbstractVector{Element{TF}}}

Composite type that defines an assembly of connected nonlinear beam elements.

# Fields
 - `points`: Array of all beam element endpoints
 - `start`: Array containing point index where each beam element starts
 - `stop`: Array containing point index where each beam element stops
 - `elements`: Array containing beam element definitions (see [`Element`](@ref))
"""
struct Assembly{TF, TP<:AbstractVector{<:AbstractVector{TF}}, TC<:AbstractVector{<:Integer}, TE<:AbstractVector{Element{TF}}}
    points::TP
    start::TC
    stop::TC
    elements::TE
end
Base.eltype(::Assembly{TF, TP, TC, TE}) where {TF, TP, TC, TE} = TF

"""
    Assembly(points, start, stop; kwargs...)

Construct an assembly of connected nonlinear beam elements.  Beam lengths and midpoints 
may be manually specified in case beam elements are curved rather than straight.

# Arguments
 - `points`: Array of all beam element endpoints
 - `start`: Array containing point indices where each beam element starts
 - `stop`: Array containing point indices where each beam element stops

# Keyword Arguments
 - `stiffness`: Array of (6 x 6) stiffness matrices for each beam element, 
        acts as an alternative to providing `compliance`
 - `compliance`: Array of (6 x 6) compliance matrices for each beam element, 
        defaults to `zeros(6,6)` for each beam element
 - `mass`: Array of (6 x 6) mass matrices for each beam element, 
        defaults to `zeros(6,6)` for each beam element
 - `frames`: Array of (3 x 3) tranformation matrices for each beam element.
        Transforms from the local undeformed beam frame to the global frame) and defaults
        to the identity matrix for each beam element
 - `lengths`: Array containing the length of each beam, defaults to the distance between 
        beam endpoints
 - `midpoints`: Array containing the midpoint of each beam element, defaults to the average 
        of the beam element endpoints
"""
function Assembly(points, start, stop;
    stiffness = nothing,
    compliance = nothing,
    mass = nothing,
    frames = nothing,
    damping = nothing,
    lengths = norm.(points[stop] - points[start]),
    midpoints = (points[stop] + points[start])/2)

    nelem = length(start)

    if isnothing(compliance)
        if isnothing(stiffness)
            compliance = fill((@SMatrix zeros(6,6)), nelem)
        else
            compliance = [(@MMatrix zeros(eltype(eltype(stiffness)), 6,6)) for i=1:nelem] #can't use fill because it copies the reference. Need a different value for every i.
            for i = 1:nelem
                filled_cols = findall(vec(mapslices(col -> any(row -> !isapprox(row, 0), col), stiffness[i], dims = 1)))
                compliance[i][filled_cols,filled_cols] .= inv(Matrix(stiffness[i][filled_cols, filled_cols]))
            end
            compliance = SMatrix.(compliance)
        end
    end

    if isnothing(mass)
        mass = fill((@SMatrix zeros(6,6)), nelem)
    end

    if isnothing(frames)
        frames = fill(I3, nelem)
    end

    if isnothing(damping)
        damping = fill(0.01*(@SVector ones(6)), nelem)
    end

    elements = Element.(lengths, midpoints, compliance, mass, frames, damping)

    return Assembly(SVector{3}.(points), promote(start, stop)..., elements)
end

"""
    discretize_beam(L, start, discretization; frame, curvature)

Discretize a beam of length `L` located at `start` according to the discretization provided 
in `discretization`

Return the lengths, endpoints, midpoints, and reference frame of the beam elements.

# Arguments
 - `L`: Beam length
 - `start`: Beam starting point
 - `discretization`: Number of beam elements, or the normalized endpoints of each beam 
        element, with values ranging from 0 to 1.

# Keyword Arguments
 - `frame`: Reference frame at the start of the beam element, represented by a 3x3
         transformation matrix from the undeformed local frame to the body frame.
 - `curvature`: curvature vector
"""
function discretize_beam(L::Number, start::AbstractVector, discretization::Integer; 
    frame=I3, curvature=(@SVector zeros(3)))
  
    return discretize_beam(L, start, range(0, 1, length=discretization+1); 
        frame=frame, curvature=curvature)
end

function discretize_beam(L::Number, start::AbstractVector, discretization; 
    frame=I3, curvature=(@SVector zeros(3)))

    r1 = SVector{3}(start)
    Cab = SMatrix{3,3}(frame)
    k = SVector{3}(curvature)

    # discretize beam
    sp = L*discretization
    sm = (sp[1:end-1] .+ sp[2:end])/2
    ΔL = sp[2:end] .- sp[1:end-1]

    # precompute some curvature quantities
    kkt = k*k'
    ktilde = tilde(k)
    kn = sqrt(k'*k)

    if iszero(curvature)
        triads = fill(Cab, length(sm))
        xp = [r1 + s*Cab*e1 for s in sp]
        xm = [r1 + s*Cab*e1 for s in sm]
    else
        triads = curve_triad.(Ref(Cab), Ref(kkt), Ref(ktilde), Ref(kn), sm)
        xp = curve_coordinates.(Ref(r1), Ref(Cab), Ref(kkt), Ref(ktilde), Ref(kn), sp)
        xm = curve_coordinates.(Ref(r1), Ref(Cab), Ref(kkt), Ref(ktilde), Ref(kn), sm)
    end

    return ΔL, xp, xm, triads
end

"""
    discretize_beam(start, stop, discretization; frame, curvature)

Discretize a beam from `start` to `stop` according to the discretization provided in 
`discretization`.

Return the lengths, endpoints, midpoints, and reference frame of the beam elements.

# Arguments
 - `start`: Beam starting point
 - `stop`: Beam ending point
 - `discretization`: Number of beam elements, or the normalized endpoints of each beam 
        element, with values ranging from 0 to 1.

# Keyword Arguments
 - `frame`: Reference frame at the start of the beam element, represented by a 3x3
        transformation matrix from the undeformed local frame to the body frame.
 - `curvature`: curvature vector
"""
function discretize_beam(start::AbstractVector, stop::AbstractVector, 
    discretization::Integer; frame=I3, curvature=(@SVector zeros(3)))
    
    return discretize_beam(start, stop, range(0, 1, length=discretization+1); 
        frame=frame, curvature=curvature)
end

function discretize_beam(start::AbstractVector, stop::AbstractVector, discretization; 
    frame=I3, curvature=(@SVector zeros(3)))

    # compute curve length
    L = curve_length(start, stop, curvature)

    # discretize beam
    ΔL, xp, xm, triads = discretize_beam(L, start, discretization; frame=frame, curvature=curvature)

    return ΔL, xp, xm, triads
end

"""
    curve_length(start, stop, curvature)

Calculate the length of a curve given its endpoints and its curvature vector.
"""
function curve_length(start, stop, curvature)

    r1 = SVector{3}(start)
    r2 = SVector{3}(stop)
    k = SVector{3}(curvature)

    # precomputed quantities
    kn = sqrt(k'*k)
    r = r2 - r1
    rn2 = r'*r
    rn = sqrt(rn2)

    # compute curve length
    if iszero(kn) || (iszero(k[2]) && iszero(k[3]))
        # the beam is straight or twisted, but not curved
        ΔL = rn
    elseif iszero(k[1])
        # the beam is curved, but not twisted
        ΔL = 2*asin(kn*rn/2)/kn
    else 
        # the beam is curved and twisted
        f = (L) -> (2*(kn^2-k[1]^2)*(1-cos(kn*L)) + k[1]^2*(kn*L)^2)/kn^4 - rn2
        ΔL = Roots.find_zero(f, (rn, rn*kn/k), Roots.Brent())
    end

    return ΔL
end

"""
    curve_triad(Cab, k, s)
    curve_triad(Cab, kkt, ktilde, kn, s)

Return the transformation matrix at `s` along the length of the beam given
the curvature vector `k` and the initial transformation matrix `Cab`.
"""
@inline curve_triad(Cab, k, s) = curve_triad(Cab, k*k', tilde(k), sqrt(k'*k), s)
@inline curve_triad(Cab, kkt, ktilde, kn, s) = SMatrix{3,3}(Cab*((I - kkt/kn^2)*cos(kn*s) +
    ktilde/kn*sin(kn*s) + kkt/kn^2))

"""
    curve_coordiantes(r, Cab, k, s)
    curve_coordinates(r, Cab, kkt, ktilde, kn, s)

Return the coordinates at `s` along the length of the beam given the starting
point `r`, initial transformation matrix `Cab`, and curvature vector `k`.
"""
@inline curve_coordinates(r, Cab, k, s) = curve_coordinates(r, Cab, k*k', tilde(k), sqrt(k'*k), s)
@inline curve_coordinates(r, Cab, kkt, ktilde, kn, s) = r + SVector{3}(Cab*((I/kn - kkt/kn^3)*sin(kn*s) + ktilde/kn^2*(1-cos(kn*s)) + kkt/kn^2*s)*e1)
