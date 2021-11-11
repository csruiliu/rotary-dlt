from numba import cuda
import igpu


class GPUSpecUtil:
    def __init__(self, gpu_id):
        self.gpu = cuda.select_device(gpu_id)
        self.gpu_info = igpu.get_device(gpu_id)

        ''' the amount of SM in the current GPU '''
        self.gpu_sm = getattr(self.gpu, 'MULTIPROCESSOR_COUNT')

        '''
        Compute capability of the current GPU,
        Compute capability comprises a major revision number X and a minor revision number Y and is denoted by X.Y.
        Return tuple, *(major, minor)* or *(X, Y)* indicating the supported compute capability
        '''
        self.gpu_cc = getattr(self.gpu, 'COMPUTE_CAPABILITY')

        ''' 
        Cores per SM according to compute capability
        The supported arch is up to 
        The dictionary needs to be extended as new devices become available 
        '''
        self.cores_dict = {(2, 0): 32,
                           (2, 1): 48,
                           (3, 0): 192,
                           (3, 5): 192,
                           (3, 7): 192,
                           (5, 0): 128,
                           (5, 2): 128,
                           (6, 0): 64,
                           (6, 1): 128,
                           (7, 0): 64,
                           (7, 5): 64,
                           (8, 0): 64,
                           (8, 6): 128}

    def get_cuda_cores(self):
        cores_per_sm = self.cores_dict.get(self.gpu_cc)
        total_cuda_cores = self.gpu_sm * cores_per_sm
        return total_cuda_cores

    def get_base_clock(self):
        return self.gpu_info.clocks.graphics


if __name__ == "__main__":
    gpu_id = 0
    gpu_util = GPUSpecUtil(gpu_id)
    cuda_cores = gpu_util.get_cuda_cores()
    base_clock = gpu_util.get_base_clock()
    computation_power = cuda_cores * base_clock
