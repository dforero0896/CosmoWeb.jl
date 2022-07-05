import numpy as np
import numba
import mkl_fft._numpy_fft as fft
import time

@numba.njit(["f4(f4)", "f8(f8)"])
def cic_correction(x):
    return 1. / np.sinc(x)**2

@numba.njit(["f4[:,:](c16[:,:,:], i8[:], f8[:])", "f4[:,:](c8[:,:,:], i8[:], f8[:])"])
def pk_from_delta_k(delta_k, dims, box_size):
    prefact = [np.pi / d for d in dims]
    middle = [d // 2 for d in dims]
    k_max = np.int32(np.sqrt((middle[0]**2 + middle[1]**2 + middle[2]**2)))
    k_fund = 2 * np.pi / np.max(box_size)

    result = np.zeros((k_max + 1, 6), dtype=np.float32)

    for kxx in range(dims[0]):
        kx = (kxx-dims[0] if (kxx>middle[0]) else kxx)
        cic_corr_x = cic_correction(prefact[0] * kx)
    
        for kyy in range(dims[1]):
            ky = (kyy-dims[1] if (kyy>middle[1]) else kyy)
            cic_corr_y = cic_correction(prefact[1] * ky)
            

            for kzz in range(middle[2]+1): #kzz=[0,1,..,middle] --> kz>0
                kz = (kzz-dims[2] if (kzz>middle[2]) else kzz)
                cic_corr_z = cic_correction(prefact[2] * kz)

                #if kz==0 or (kz==middle[2] and dims[0]%2==0):
                #    if kx<0: continue
                #    elif kx==0 or (kx==middle[0] and dims[0]%2==0):
                #        if ky<0.0: continue

                k_norm = np.sqrt(kx**2 + ky**2 + kz**2)
                k_index = np.int64(k_norm)

                
                k_par = kz
                if k_norm == 0:
                    mu = 0.
                else:
                    mu = k_par / k_norm
                
                musq = mu**2
                
                if k_par<0:  k_par = -k_par

                delta_k[kxx, kyy, kzz] *= cic_corr_x * cic_corr_y * cic_corr_z
                real = delta_k[kxx, kyy, kzz].real
                imag = delta_k[kxx, kyy, kzz].imag
                delta_k_sq = real**2 + imag**2
                phase = np.arctan2(real, np.sqrt(delta_k_sq))


                result[k_index,0] += k_norm
                result[k_index,1] += delta_k_sq
                result[k_index,2] += (delta_k_sq*(3.0*musq-1.0)/2.0)
                result[k_index,3] += (delta_k_sq*(35.0*musq*musq - 30.0*musq + 3.0)/8.0)
                result[k_index,4] += (phase*phase)
                result[k_index,5] += 1.0
    for i in range(1,result.shape[0]):
        result[i,0]     = (result[i,0]/result[i,5])*k_fund
        result[i,1]  = (result[i,1]/result[i,5])*(box_size[0]/dims[0]**2) * (box_size[1]/dims[1]**2) * (box_size[2]/dims[2]**2)
        result[i,2]  = (result[i,2]*5.0/result[i,5])*(box_size[0]/dims[0]**2) * (box_size[1]/dims[1]**2) * (box_size[2]/dims[2]**2)
        result[i,3]  = (result[i,3]*9.0/result[i,5])*(box_size[0]/dims[0]**2) * (box_size[1]/dims[1]**2) * (box_size[2]/dims[2]**2)
        result[i,4] = (result[i,4]/result[i,5])*(box_size[0]/dims[0]**2) * (box_size[1]/dims[1]**2) * (box_size[2]/dims[2]**2)

    
    return result

def rebin_field(field, new_shape):
    return field.reshape(new_shape[0], field.shape[0]//new_shape[0], 
                         new_shape[1], field.shape[1]//new_shape[1], 
                         new_shape[2], field.shape[2]//new_shape[2]).sum(axis=(1, 3, 5))

def powspec_fundamental(delta, box_size, k_lim):
    dims = delta.shape

    s = time.time()
    print(f"Computing FFT", flush=True)
    delta_k = fft.rfftn(delta)
    print(f"Done in {time.time() - s} s", flush=True)
    s = time.time()
    print(f"Computing Pk", flush=True)
    results =  pk_from_delta_k(delta_k, np.asarray(dims), np.asarray(box_size))
    mask = results[:,0] < k_lim
    print(f"Done in {time.time() - s} s", flush=True)

    return results[mask]

    


if __name__ == '__main__':
    
    test_fn = "/home2/dfelipe/projects/ir_models/data/dmdens_hr.npy"
    s = time.time()
    delta = np.load(test_fn)
    print(f"Loading in {time.time() - s} s", flush=True)
    s = time.time()
    print("Rebinning", flush=True)
    delta = rebin_field(delta, (1024, 1024, 1024))
    delta /= delta.mean()
    delta -= 1.
    print(f"Rebin in {time.time() - s} s")
    box_size = (1000., 1000., 1000.)
    k_lim = np.pi * min(delta.shape) / min(box_size)

    results = powspec_fundamental(delta, box_size, k_lim)
    np.save("tests_py.npy", results)

