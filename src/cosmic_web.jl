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
end


function compute_tidal_field(delta::Array{<:AbstractFloat, 3}, box_size::SVector{3}, smoothing::AbstractFloat)
    n_bins = size(delta)
    @show n_bins

    init_k = rfft(delta)
    init_k[1,1,1] = 0.

    kx = rfftfreq(n_bins[1], Float32(n_bins[1] / box_size[1] .* 2 * pi))
    ky = fftfreq(n_bins[2], Float32(n_bins[2] / box_size[2] .* 2 * pi))
    kz = ky
    

    ks = [kx, ky, kz]

    
    
    Threads.@threads for i in 1:length(kx)
        for j in 1:length(ky)
            for k in 1:length(kz)
                ksq = kx[i]^2 + ky[j]^2 + kz[k]^2
                init_k[i,j,k] *= exp(-0.5 * 2 * pi * ksq * smoothing^2)
                if ksq > 0.
                    init_k[i,j,k] /= ksq
                else
                    init_k[i,j,k] = 0.
                end
    
                #tidal_field[k,j,i, 1, 1] = - kx[i] * kx[i] * init_k[k,j,i]
                #tidal_field[k,j,i, 1, 2] = - kx[i] * ky[j] * init_k[k,j,i]
                #tidal_field[k,j,i, 1, 3] = - kx[i] * kz[k] * init_k[k,j,i]
#
                #tidal_field[k,j,i, 2, 1] = tidal_field[k,j,i, 1, 2]
                #tidal_field[k,j,i, 2, 2] = - ky[j] * ky[j] * init_k[k,j,i]
                #tidal_field[k,j,i, 2, 3] = - ky[j] * kz[k] * init_k[k,j,i]
#
                #tidal_field[k,j,i, 3, 1] = tidal_field[k,j,i, 1, 3]
                #tidal_field[k,j,i, 3, 2] = tidal_field[k,j,i, 2, 3]
                #tidal_field[k,j,i, 3, 3] = - kz[k] * kz[k] * init_k[k,j,i]
            end
        end
    end

    k_space_buffer = similar(init_k)
    tidal_field = [Array{Float32, 3}(undef, size(delta)) for i=1:6]
    
    newaxis = [CartesianIndex()]
    #
    # 1,1 element is index 1
    k_space_buffer .= [- kx[i] * kx[i] * init_k[i,j,k] for i=1:size(init_k,1), j=1:size(init_k,2), k=1:size(init_k,3)]
    tidal_field[1] .= irfft(k_space_buffer, size(delta, 1))
    # 2,1 element is index 2
    k_space_buffer .= [- ky[j] * kx[i] * init_k[i,j,k] for i=1:size(init_k,1), j=1:size(init_k,2), k=1:size(init_k,3)]
    tidal_field[2] .= irfft(k_space_buffer, size(delta, 1))
    # 3,1 element is index 3
    k_space_buffer .= [- kz[k] * kx[i] * init_k[i,j,k] for i=1:size(init_k,1), j=1:size(init_k,2), k=1:size(init_k,3)]
    tidal_field[3] .= irfft(k_space_buffer, size(delta, 1))
    # 2,2 element is index 4
    k_space_buffer .= [- ky[j] * ky[j] * init_k[i,j,k] for i=1:size(init_k,1), j=1:size(init_k,2), k=1:size(init_k,3)]
    tidal_field[4] .= irfft(k_space_buffer, size(delta, 1))
    # 3,2 element is index 5
    k_space_buffer .= [- kz[k] * ky[j] * init_k[i,j,k] for i=1:size(init_k,1), j=1:size(init_k,2), k=1:size(init_k,3)]
    tidal_field[5] .= irfft(k_space_buffer, size(delta, 1))
    # 3.3 element is index 6 
    k_space_buffer .= [- kz[k] * kz[k] * init_k[i,j,k] for i=1:size(init_k,1), j=1:size(init_k,2), k=1:size(init_k,3)]
    tidal_field[6] .= irfft(k_space_buffer, size(delta, 1))

    
    matrix_buffer = Matrix{Float32}(undef, (3,3))
    vector_buffer = Vector{Float32}(undef, 3)
    eigenvalues = [Array{Float32, 3}(undef, size(delta)) for i=1:3]
    Threads.@threads for i in 1:size(delta, 2)
        for j in 1:size(delta, 2)
            for k in 1:size(delta,3)
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
        end
    end

                        


    
    

    eigenvalues

end

function compute_tidal_eigvals(tidal_field::Array{<:Real})

    
end

