import os
import sys
N_RUN = 5

def run_bench(apps, image_apps, is_cuda, sycl_dev): 
    bench = "sycl" if is_cuda == 0 else "cuda"

    sycl_dev_options = ('--sycl='+sycl_dev) if is_cuda==0 else ('')
    for name in apps:
        #move in build/cuda|sycl directory
        cd_command = 'cd '+ '../'+bench+'/build/'+bench+'/ && ' + './'+ name
        # run all code in apps N_RUN times
        for i in range(N_RUN):
            #executio command
            if name in image_apps:
                exe_command= ' ../../../img/Lenna512.png ../../../img/Lenna512_output.png '+ sycl_dev_options +' >> ../../../results/'+bench+'_times.csv'
            else:
                exe_command= ' ' + sycl_dev_options + ' >> ../../../results/' + bench+'_times.csv'
            
            os.system(cd_command + exe_command)
        
        os.system('python -c \'print("\\n")\' >> ../results/'+bench+'_times.csv')

# list cuda applications
cuda_apps = ["box_blur", "box_blur_local_memory", 
            "matrix_mul_cublas", "matrix_mul_tiling",
            "matrix_transpose", "nbody", 
            "reduction_binary", "reduction_tile32", 
            "sobel_filter"]
#list sycl applications
sycl_apps = ["box_blur", "box_blur_local_memory", 
            "matrix_mul_tiling",
            "matrix_transpose", "nbody", 
            "reduction_binary", "reduction_subgroup",
            "class_reducer", "sobel_filter"]

image_apps = ["box_blur", "box_blur_local_memory", "sobel_filter"]

#remove old files
os.system('rm -f ../results/cuda_times.csv')
os.system('rm -f ../results/sycl_times.csv')

#cpu or gpu
if len(sys.argv) != 2:
    print("Specify sycl dev (cpu or gpu) as command line argument")
    exit(-1)
# is_cuda = 1
# run_bench(cuda_apps,image_apps, is_cuda, '')


sycl_dev = sys.argv[1]

is_cuda = 0
run_bench(sycl_apps, image_apps, is_cuda, sycl_dev)