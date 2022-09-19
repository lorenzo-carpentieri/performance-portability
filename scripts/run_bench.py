import os


N_RUN = 5

def run_bench(apps, image_apps, is_cuda): 
    bench = "sycl" if is_cuda == 0 else "cuda"
    for name in apps:
        os.chdir('/root/tesi/performance-portability/'+bench+'/build/'+bench+'/')
        # run all code in apps N_RUN times
        for i in range(N_RUN):
            if name in image_apps:
                os.system('./'+ name +' /root/tesi/performance-portability/img/Lenna512.png'+' /root/tesi/performance-portability/img/Lenna512_output.png'+' >> /root/tesi/performance-portability/results/'+bench+'_times.csv')
            else:
                os.system('./'+ name +' >> /root/tesi/performance-portability/results/'+bench+'_times.csv')
    
        os.system('python -c \'print "\\n"\' >> /root/tesi/performance-portability/results/'+bench+'_times.csv')

# list cuda application
cuda_apps = ["box_blur", "box_blur_local_memory", 
            "matrix_mul_cublas", "matrix_mul_tiling",
            "matrix_transpose", "nbody", 
            "reduction_binary", "reduction_tile32", 
            "sobel_filter"]

sycl_apps = ["box_blur", "box_blur_local_memory", 
            "matrix_mul_tiling",
            "matrix_transpose", "nbody", 
            "reduction_binary", "reduction_subgroup",
            "class_reducer", "sobel_filter"]

image_apps = ["box_blur", "box_blur_local_memory", "sobel_filter"]

os.system('rm -f /root/tesi/performance-portability/results/cuda_times.csv')
os.system('rm -f /root/tesi/performance-portability/results/sycl_times.csv')


is_cuda = 1
run_bench(cuda_apps,image_apps, is_cuda)
is_cuda = 0
run_bench(sycl_apps, image_apps, is_cuda)