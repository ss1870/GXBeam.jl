using GXBeam, LinearAlgebra, CSV, DelimitedFiles, MAT, BlockDiagonals, Revise
using Statistics

print("\n\n")
# Beam 1 is initially disconnected from beam 2/3
L_b1 = 4 
r_b1 = [0, 0, 0]
nelem_b1 = 100
lengths_b1, xp_b1, xm_b1, Cab_b1 = discretize_beam(L_b1, r_b1, nelem_b1)

# Beam 2 and 3 are essentially one beam
L_b2 = 4 
r_b2 = [4, 0, 0]
nelem_b2 = 100
lengths_b2, xp_b2, xm_b2, Cab_b2 = discretize_beam(L_b2, r_b2, nelem_b2)
L_b3 = 1 
r_b3 = [8, 0, 0]
nelem_b3 = 40
lengths_b3, xp_b3, xm_b3, Cab_b3 = discretize_beam(L_b3, r_b3, nelem_b3)

# Concatenate beam data
nelem = nelem_b1 + nelem_b2 + nelem_b3
points = vcat(xp_b1, xp_b2[1:end], xp_b3[2:end])
start = vcat(1:nelem_b1, (nelem_b1 + 2):nelem_b1 + nelem_b2 + nelem_b3 + 1)
stop = vcat(2:nelem_b1 + 1, (nelem_b1 + 3):nelem_b1 + nelem_b2 + nelem_b3 + 2)
lengths = vcat(lengths_b1, lengths_b2, lengths_b3)
midpoints = vcat(xm_b1, xm_b2, xm_b3)
Cab = vcat(Cab_b1, Cab_b2, Cab_b3)


h = w = 0.5 # 
E = 210e9 # Young's Modulus
ν = 0.3
G = E/(2*(1+ν))
# cross section properties
A = h*w
Ay = A
Az = A
Iyy = w*h^3/12
Izz = w^3*h/12
J = Iyy + Izz

# compliance matrix for each beam element
compliance = fill(Diagonal([1/(E*A), 1/(G*Ay), 1/(G*Az), 1/(G*J), 1/(E*Iyy), 1/(E*Izz)]), nelem)

# create the assembly
assembly = Assembly(points, start, stop, compliance=compliance, frames=Cab, lengths=lengths, midpoints=midpoints)

# Add loads/boundary conditions
# Uz, thetaX, thetaY all constrained to ensure 2D
prescribed_conditions = Dict{Int64,Any}()
for i in 1:maximum(stop)
    if i == 1
        # cantilever LHS
        prescribed_conditions[i] = PrescribedConditions(ux=0, uy=0, uz=0, theta_x=0, theta_y=0, theta_z=0)
    elseif i == nelem_b1/2+1
        # Fy 10kN
        prescribed_conditions[i] = PrescribedConditions(uz=0, Fy=-10e3, theta_x=0, theta_y=0)
    elseif i == nelem_b1+2+nelem_b2
        # Simply supported
        prescribed_conditions[i] = PrescribedConditions(ux=0, uy=0, uz=0, theta_x=0, theta_y=0)
    elseif i == maximum(stop)
        # tip forces
        prescribed_conditions[i] = PrescribedConditions(Fy=-5e3, Fx=-10e3, uz=0, theta_x=0, theta_y=0)
    else
        prescribed_conditions[i] = PrescribedConditions(uz=0, theta_x=0, theta_y=0)
    end
end

# Add distributed load
distributed_loads = Dict()
for ielem in (nelem_b1+1):(nelem_b1+nelem_b2)
    distributed_loads[ielem] = DistributedLoads(assembly, ielem; fy = (s) -> -5e3)
end

# Add joint to connect beam 1 and 2/3 in translation only
frame = [0 -1 0;
         1  0 0;
         0  0 1]
frame = [cos(pi) -sin(pi) 0;
         sin(pi)  cos(pi) 0;
         0          0     1]         
joints = [
    Joint(nelem_b1 + 1, nelem_b1 + 2; frame=inv(frame), ux=true, uy=true)
]

# perform static analysis
system, converged = static_analysis(assembly, 
                    prescribed_conditions=prescribed_conditions,
                    distributed_loads=distributed_loads,
                    joints=joints,
                    show_trace=true,
                    linear=true)
show(converged)
state = AssemblyState(system, assembly, prescribed_conditions=prescribed_conditions)

# Print point displacements and external forces/moments
for i in 1:size(state.points)[1]
    print(state.points[i].u)
    print(state.points[i].theta*180/pi)
    print(state.points[i].F)
    print(state.points[i].M)
    print("\n")
end
# Get element internal reactions
Fi = [[state.elements[i].Fi; state.elements[i].Mi] for i in 1:size(state.elements)[1]]
for i in 1:size(state.elements)[1]
    print(state.elements[i].Fi)
    print(state.elements[i].Mi)
    print("\n")
end

# Check reactions
print(isapprox(state.points[1].F, [0, 18750, 0], atol=0.1))
print(isapprox(state.points[1].M, [0, 0, 55000], atol=0.1))
print(isapprox(state.points[nelem_b1+nelem_b2+2].F, [10000, 16250, 0], atol=0.1))
print(isapprox(state.points[nelem_b1+nelem_b2+2].M, [0, 0, 0], atol=0.1))

# Check internal reactions
print(isapprox(mean(getindex.(Fi[1:convert(Int64, nelem_b1/2)],2)), -18750, atol=0.1))
print(isapprox(mean(getindex.(Fi[convert(Int64, nelem_b1/2)+1:nelem_b1],2)), -8750, atol=0.1))
print(isapprox(mean(getindex.(Fi[nelem_b1+nelem_b2+1:nelem],2)), -5000, atol=0.1))
