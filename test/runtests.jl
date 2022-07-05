using CosmoWeb
using Test
using NPZ
using StaticArrays
using FortranFiles
using UnicodePlots
using Statistics
using CSV
using Profile
#pyplot()

@testset "CosmoWeb.jl" begin
    # Write your tests here.
end

function read_fortran_field(filename)
    f = FortranFile(filename)
    shape = read(f, (Int32, 3))
    @show shape
    #delta = [read(f, (Float32, shape[1], shape[1])) for _ in 1:2048]
    delta = Array{Float32, 3}(undef, Tuple(shape))
    delta_buffer = Array{Float32, 2}(undef, Tuple(shape)[1:2])
    for i in 1:2048
        read(f, delta_buffer)
        view(delta, :,:,i) .= delta_buffer 

           
    end
    delta
    #cat([view(a, :,:, CartesianIndex()) for a in delta]..., dims=3)
    #reinterpret(reshape, Float32, delta)
    
end

test_fn = "/home2/dfelipe/projects/ir_models/data/dmdens_hr.npy"
hr_name = "/data5/UNITSIM/1Gpc_4096/fixedAmp_InvPhase_001/DM_DENS/dmdens_cic_128.dat"

@time delta = read_fortran_field(hr_name)
#@time delta = npzread(test_fn)
heatmap(log10.(dropdims(mean(delta,dims=1), dims=1) .+ 1.), aspect=1)
#savefig(p1, "/home2/dfelipe/codes/CosmoWeb/test/test.png")
println("Rebinning field...")

@time delta = CosmoWeb._rebin_field(delta, (8, 8, 8))


#@time delta = CosmoWeb.rebin_field(delta, (4, 4, 4))
#@time delta = NNlib.meanpool(delta, (4,))
println("Normalizing field...")
delta ./= mean(delta)
delta .-= 1.
heatmap(dropdims(mean(delta,dims=1), dims=1))
p2 = heatmap(dropdims(mean(delta,dims=1), dims=1), aspect=1)

println("Computing Pk of field...")

@time k_edges, pk, _, n_modes = CosmoWeb.powspec_fundamental(delta, SVector{3}(1f3,1f3,1f3), Float32(pi * size(delta,1) / 1000.))

CSV.write("/home2/dfelipe/codes/CosmoWeb/test/test.csv", (k=k_edges, pk0=pk[1], pk2=pk[2], nmodes=n_modes))
#p1 = plot(k_edges, k_edges.*pk[1], xaxis=:log)
#@show p1
#plot!(k_edges, k_edges.*pk[2], xaxis=:log)
#savefig(p1, "/home2/dfelipe/codes/CosmoWeb/test/test.png")

#p = plot(p1, p2, layout=(1,3), aspect=3)
#savefig(p, "/home2/dfelipe/codes/CosmoWeb/test/test.png")

#exit()
#@time eigenvalues = CosmoWeb.compute_tidal_field(delta, SVector{3}(1000., 1000., 1000.), 10.)

