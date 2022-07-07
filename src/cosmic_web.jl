using FFTW
using StaticArrays
using Statistics
using LinearAlgebra
FFTW.set_provider!("mkl")
FFTW.set_num_threads(32)


function rebin_field(field::Array{<:AbstractFloat, 3}, kernel_size::Tuple{<:Int, <:Int, <:Int})

    init_shape = size(field)
    new_shape = [Int64(floor(init_shape[i] / kernel_size[i])) for i in 1:3]
    output = zeros(Float32,Tuple(new_shape)...)
    Rin = CartesianIndices(field)
    Rout = CartesianIndices(output)
    Rkernel = CartesianIndices(Tuple(1:k for k in kernel_size))
    for Iout in Rout

        for kk in 1:kernel_size[3]
            for jj in 1:kernel_size[2]
                for ii in 1:kernel_size[1]
                    i, j, k = Tuple(Iout)
                    field_id_i = (kernel_size[1] * i+ii-2)  
                    field_id_i = field_id_i > init_shape[1] ? field_id_i - init_shape[1] : field_id_i
                    field_id_j = (kernel_size[2] * j+jj-2)  
                    field_id_j = field_id_j > init_shape[2] ? field_id_j - init_shape[2] : field_id_j
                    field_id_k = (kernel_size[3] * k+kk-2)  
                    field_id_k = field_id_k > init_shape[3] ? field_id_k - init_shape[3] : field_id_k
                    @inbounds output[i,j,k] += field[field_id_i, field_id_j, field_id_k]
                end
            end
        end
        @inbounds output[Iout] /= *(kernel_size...)
    end
    
    output
end

function _rebin_field(field::Array{<:AbstractFloat, 3}, kernel_size::Tuple{<:Int, <:Int, <:Int})
    init_shape = size(field)
    new_shape = [Int64(floor(init_shape[i] / kernel_size[i])) for i in 1:3]
    dropdims(sum(reshape(field, kernel_size[1], new_shape[1], kernel_size[2], new_shape[2], kernel_size[3], new_shape[3]), dims=(1, 3, 5)), dims=(1, 3, 5))
    #dropdims(sum(reshape(field, new_shape[1], kernel_size[1], new_shape[2], kernel_size[2], new_shape[3], kernel_size[3]), dims=(2, 4, 6)), dims=(2, 4, 6))
end

function tidal_field_ab!(a::Int, b::Int, ks::Vector{<:AbstractVector{T}}, delta_k::Array{<:Complex{T}, 3}, smoothing::T, tidal_comp::Array{<:Complex{T}, 3}) where T<:AbstractFloat
    R = CartesianIndices(delta_k)
    for I in R
        i, j, k = Tuple(I)
        ksq = ks[1][i]^2 + ks[2][j]^2 + ks[3][k]^2
        if ksq > 0.
            tidal_comp[I] = - ks[a][I[a]] * ks[b][I[b]] * delta_k[I] * exp(-0.5 * 2 * pi * ksq * smoothing^2) / ksq
        else
            tidal_comp[I] = 0.
        end
    end
end


function compute_tidal_field(delta::Array{<:T, 3}, box_size::SVector{3, T}, smoothing::T) where T<:AbstractFloat

    n_bins = size(delta)
    @show n_bins

    init_k = rfft(delta)
    init_k[1,1,1] = 0.
    irfft_plan = plan_irfft(init_k, size(delta,1))

    kx::Vector{T} = rfftfreq(n_bins[1], T(n_bins[1] / box_size[1] .* 2 * pi))
    ky::Vector{T} = fftfreq(n_bins[2], T(n_bins[2] / box_size[2] .* 2 * pi))
    kz::Vector{T} = ky
    

    ks = [kx, ky, kz]

    k_space_buffer = similar(init_k)
    tidal_field = [Array{Float32, 3}(undef, size(delta)) for i=1:6]
    
    
    #
    # 1,1 element is index 1
    tidal_field_ab!(1, 1, ks, init_k, smoothing, k_space_buffer)
    mul!(tidal_field[1], irfft_plan, k_space_buffer)
    # 2,1 element is index 2
    tidal_field_ab!(2, 1, ks, init_k, smoothing, k_space_buffer)
    mul!(tidal_field[2], irfft_plan, k_space_buffer)
    # 3,1 element is index 3
    tidal_field_ab!(3, 1, ks, init_k, smoothing, k_space_buffer)
    mul!(tidal_field[3], irfft_plan, k_space_buffer)
    # 2,2 element is index 4
    tidal_field_ab!(2, 2, ks, init_k, smoothing, k_space_buffer)
    mul!(tidal_field[4], irfft_plan, k_space_buffer)
    # 3,2 element is index 5
    tidal_field_ab!(3, 2, ks, init_k, smoothing, k_space_buffer)
    mul!(tidal_field[5], irfft_plan, k_space_buffer)
    # 3.3 element is index 6 
    tidal_field_ab!(3, 3, ks, init_k, smoothing, k_space_buffer)
    mul!(tidal_field[6], irfft_plan, k_space_buffer)

    
    tidal_field

end


function compute_tidal_eigenvalues(delta::Array{<:T, 3}, box_size::SVector{3, T}, smoothing::T) where T<:AbstractFloat

    tidal_field = compute_tidal_field(delta, box_size, smoothing)

    matrix_buffer = Matrix{Float32}(undef, (3,3))
    vector_buffer = Vector{Float32}(undef, 3)
    eigenvalues = [Array{Float32, 3}(undef, size(delta)) for i=1:3]
    for I in CartesianIndices(delta)
        i, j, k = Tuple(I)
        matrix_buffer[1,1] = tidal_field[1][i,j,k]
        matrix_buffer[2,1] = tidal_field[2][i,j,k]
        matrix_buffer[3,1] = tidal_field[3][i,j,k]
        matrix_buffer[2,2] = tidal_field[4][i,j,k]
        matrix_buffer[3,2] = tidal_field[5][i,j,k]
        matrix_buffer[3,3] = tidal_field[6][i,j,k]
        vector_buffer .= eigvals!(Hermitian(matrix_buffer,:L))

        eigenvalues[1][i,j,k] = vector_buffer[3]
        eigenvalues[2][i,j,k] = vector_buffer[2]
        eigenvalues[3][i,j,k] = vector_buffer[1]
    end

    eigenvalues
end


function compute_tidal_invariants(delta::Array{<:T, 3}, box_size::SVector{3, T}, smoothing::T) where T<:AbstractFloat

    eigenvalues = compute_tidal_eigenvalues(delta, box_size, smoothing)
    invariants = [zeros(T, size(delta)) for _ in 1:5]

    for I in CartesianIndices(delta)
        i,j,k = Tuple(I)
        invariants[1][I] = eigenvalues[1][I] + eigenvalues[2][I] + eigenvalues[3][I]
        invariants[2][I] = eigenvalues[1][I] * eigenvalues[1][I] + eigenvalues[3][I] * eigenvalues[2][I] * eigenvalues[3][I]
        invariants[3][I] = eigenvalues[1][I] * eigenvalues[2][I] * eigenvalues[3][I]
        invariants[4][I] = eigenvalues[1][I]^2 + eigenvalues[2][I]^2 + eigenvalues[3][I]^2
        invariants[5][I] = eigenvalues[1][I]^3 + eigenvalues[2][I]^3 + eigenvalues[3][I]^3
        invariants[6][I] = invariants[1][I] * invariants[2][I]
    end
    invariants
end


