using CosmoWeb
using Test
using NPZ
using StaticArrays
using FortranFiles
using Plots
using Statistics
using CSV
using Profile
using Zygote
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
end

test_fn = "/home2/dfelipe/projects/ir_models/data/dmdens_hr.npy"
hr_name = "/data5/UNITSIM/1Gpc_4096/fixedAmp_InvPhase_001/DM_DENS/dmdens_cic_128.dat"

@time delta = read_fortran_field(hr_name);
println("Rebinning field...")
@time delta = CosmoWeb._rebin_field(delta, (4, 4, 4));
println("Normalizing field...")
delta ./= mean(delta);
delta .-= 1.;
#field_size = 128
#delta = randn(Float32, (field_size,field_size,field_size))
box_size = SVector{3}(1f3,1f3,1f3)

p = heatmap(log.(dropdims(mean(delta, dims=1), dims=1) .+ 2.), title="dm")
savefig(p, "/home2/dfelipe/codes/CosmoWeb/test/test.png")


println("Computing Pk of field...")
@time k_edges, pk, _, n_modes = CosmoWeb.powspec_fundamental(delta, box_size, Float32(pi * size(delta,1) / 1000.))
p2 = plot(k_edges[2:end], k_edges[2:end].*pk[1,2:end], xscale=:log10)
fin_plot = plot(p, p2, dpi=200, size=(1000,1000), layout = (2,2))
savefig(fin_plot, "/home2/dfelipe/codes/CosmoWeb/test/test.png")
CSV.write("/home2/dfelipe/codes/CosmoWeb/test/test.csv", (k=k_edges, pk0=pk[1,:], pk2=pk[2,:], nmodes=n_modes))


@time invariants = CosmoWeb.compute_tidal_invariants(delta, box_size, 5f0);
pinv = [heatmap(dropdims(mean(-invariants[i], dims=1), dims=1), title="I$i") for i in 1:6]
fin_plot = plot(p, pinv..., p2, dpi=200, size=(1000,700), layout = (3,3))
savefig(fin_plot, "/home2/dfelipe/codes/CosmoWeb/test/test.png")

