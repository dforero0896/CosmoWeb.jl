using FFTW
using ProgressBars

function cic_correction(x::AbstractFloat)
    x == 0. ? 1. : (x / sin(x))^2
end

function powspec_fundamental_threads(delta::Array{<:AbstractFloat, 3}, box_size::SVector{3}, k_lim::AbstractFloat)
    dims = size(delta)
    
    middle = [Int32(d / 2)  for d in dims]
    k_fund = 2. * pi / box_size[1]
    k_ny = middle * k_fund
    k_max_par = middle
    prefactor = [pi / d for d in dims]
    k_max_per = Int32.(floor.(sqrt.(middle.^2 .+ middle.^2)))
    k_max = Int32.(floor.(sqrt.(middle.^2 .+ middle.^2 .+ middle.^2)))

    delta_k = rfft(delta)
    delta_k[1,1,1] = 0.

    kx = rfftfreq(dims[1], Float32(dims[1] / box_size[1] .* 2 * pi))
    ky = fftfreq(dims[2], Float32(dims[2] / box_size[2] .* 2 * pi))
    kz = ky

    k_edges = [zeros(Float32, k_max[1] + 1) for _ in 1:Threads.nthreads()]
    pk = [[zeros(Float32, k_max[1] + 1) for _ in 1:3] for _ in 1:Threads.nthreads()]
    pk_phase = [zeros(Float32, k_max[1] + 1) for _ in 1:Threads.nthreads()]
    n_modes = [zeros(Int32, k_max[1] + 1) for _ in 1:Threads.nthreads()]
    R = CartesianIndices(delta_k)
        
    Threads.@threads for I in R
        kxx, kyy, kzz = Tuple(I) 
        
        #if (kx[kxx] / k_fund == 0.) || ((kx[kxx] == middle[1] * k_fund) && (dims[1] % 2 == 0) )
        #    if ky[kyy] < 0
        #        continue
        #    elseif (ky[kyy] / k_fund == 0.) || ((ky[kyy] == middle[2] * k_fund) && (dims[2] % 2 == 0))
        #        if kz[kzz] < 0
        #            continue
        #        end
        #    end
        #end

        
        cic_corr = *([cic_correction(prefactor[1] * k_ / k_fund) for k_ in (kx[kxx], ky[kyy], kz[kzz])]...)
        k_norm = sqrt(kx[kxx]^2 + ky[kyy]^2 + kz[kzz]^2)
        
        k_index = Int32(round(k_norm / k_fund))
        
        k_par = kz[kzz]
        k_per = Int32(round(sqrt(kx[kxx]*kx[kxx] + ky[kyy]*ky[kyy])))

        mu = k_norm == 0. ? 0. : k_par / k_norm
        musq = mu^2
        k_par = k_par < 0. ? -k_par : k_par
        #delta_k[kxx,kyy,kzz] *= cic_correction_x * cic_correction_y * cic_correction_z
        delta_k_sq = abs2(delta_k[kxx,kyy,kzz] * cic_corr)
        phase = angle(delta_k[kxx,kyy,kzz] * cic_corr)

        k_edges[Threads.threadid()][k_index + 1] += k_norm
        pk[Threads.threadid()][1][k_index + 1] += delta_k_sq
        pk[Threads.threadid()][2][k_index + 1] += (delta_k_sq * (3. * musq - 1.) / 2.)
        pk[Threads.threadid()][3][k_index + 1] += (delta_k_sq * (35. * musq^2 - 30. * musq + 3.) / 8.)
        pk_phase[Threads.threadid()][k_index + 1] += phase^2
        n_modes[Threads.threadid()][k_index + 1] += 1
    end
    
    for i in 2:Threads.nthreads()
        k_edges[1] .+= k_edges[i]
        pk[1][1] .+= pk[i][1]
        pk[1][2] .+= pk[i][2]
        pk[1][3] .+= pk[i][3]
        pk_phase[1] .+= pk_phase[i]
        n_modes[1] .+= n_modes[i]
    end

    k_edges = k_edges[1]
    pk = [pk[1][i] for i in 1:3]
    pk_phase = pk_phase[1]
    n_modes = n_modes[1]

    units_factor = (box_size[1] / dims[1]^2)^3
    for i in 1:length(k_edges)
        
        k_edges[i] *= k_fund / n_modes[i]
        pk[1][i] *= 1. / n_modes[i] * units_factor
        pk[2][i] *= 5. / n_modes[i] * units_factor
        pk[3][i] *= 9. / n_modes[i] * units_factor
        pk_phase[i] *= units_factor / n_modes[i]
    end
    return [k for k in k_edges if k<k_lim], [[pk[j][i] for i in 1:length(k_edges) if k_edges[i]<k_lim] for j in 1:3], [pk_phase[i] for i in 1:length(k_edges) if k_edges[i]<k_lim], [n_modes[i] for i in 1:length(k_edges) if k_edges[i]<k_lim] 

end

function mode_count_fundamental(delta_k::Array{<:Complex{T}, 3}, dims::Tuple{Int,Int,Int}, box_size::SVector{3,T}) where T <:AbstractFloat
    middle = [Int32(d / 2)  for d in dims]
    k_fund = 2. * pi / maximum(box_size)
    k_ny = middle .* k_fund
    prefactor = [pi / d for d in dims]
    k_max = Int32.(floor.(sqrt.(middle.^2 .+ middle.^2 .+ middle.^2)))
    k_edges = zeros(Float32, k_max[1] + 1)
    pk = [zeros(Float32, k_max[1] + 1) for _ in 1:3]
    pk_phase = zeros(Float32, k_max[1] + 1)
    n_modes = zeros(Int32, k_max[1] + 1)

    R = CartesianIndices(delta_k)

    for I in ProgressBar(R)
        kxx, kyy, kzz = Tuple(I) 
        kx = (kxx - 1.) > middle[1] ? (kxx - dims[1] - 1) : (kxx - 1.)
        ky = (kyy - 1.) > middle[2] ? (kyy - dims[2] - 1) : (kyy - 1.)
        kz = (kzz - 1.) > middle[3] ? (kzz - dims[3] - 1) : (kzz - 1.)
        
        if (kx == 0.) || ((kx == middle[1] ) && (dims[2] % 2 == 0) )
            if ky < 0
                continue
            elseif (ky == 0.) || ((ky == middle[2]) && (dims[2] % 2 == 0))
                if kz < 0
                    continue
                end
            end
        end

    
        cic_corr = *([cic_correction(prefactor[1] * k_) for k_ in (kx, ky, kz)]...)
        
        k_norm = sqrt(kx^2 + ky^2 + kz^2)
        
        k_index = Int32(floor(k_norm))
        
        k_par = kz
        #k_per = Int32(round(sqrt(kx*kx + ky*ky)))

        mu = k_norm == 0. ? 0. : k_par / k_norm
        musq = mu^2
        k_par = k_par < 0. ? -k_par : k_par
        #delta_k[kxx,kyy,kzz] *= cic_correction_x * cic_correction_y * cic_correction_z
        delta_k[I] *= cic_corr
        delta_k_sq = abs2(delta_k[kxx,kyy,kzz])
        phase = angle(delta_k[kxx,kyy,kzz])

        k_edges[k_index + 1] += k_norm
        pk[1][k_index + 1] += delta_k_sq
        pk[2][k_index + 1] += (delta_k_sq * (3. * musq - 1.) / 2.)
        pk[3][k_index + 1] += (delta_k_sq * (35. * musq^2 - 30. * musq + 3.) / 8.)
        pk_phase[k_index + 1] += phase^2
        n_modes[k_index + 1] += 1
    end
    println("Done")
    units_factor = (box_size[1] / dims[1]^2)^3
    for i in 1:length(k_edges)
        k_edges[i] *= k_fund / n_modes[i]
        pk[1][i] *= 1. / n_modes[i] * units_factor
        pk[2][i] *= 5. / n_modes[i] * units_factor
        pk[3][i] *= 9. / n_modes[i] * units_factor
        pk_phase[i] *= units_factor / n_modes[i]
    end

    return k_edges, pk, pk_phase, n_modes
end

    

function powspec_fundamental(delta::Array{<:T, 3}, box_size::SVector{3,T}, k_lim::T) where T<:AbstractFloat
    dims = size(delta)
    
    
    
    
    println("Computing FFT.")
    @time delta_k = rfft(delta)
    #delta_k[1,1,1] = 0.
    println("Done")
    println("Computing Pk from FFT")
    @time k_edges, pk, pk_phase, n_modes = mode_count_fundamental(delta_k, dims, box_size)

    return [k for k in k_edges if k<k_lim], [[pk[j][i] for i in 1:length(k_edges) if k_edges[i]<k_lim] for j in 1:3], [pk_phase[i] for i in 1:length(k_edges) if k_edges[i]<k_lim], [n_modes[i] for i in 1:length(k_edges) if k_edges[i]<k_lim] 

end
