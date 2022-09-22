import os
import sys

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

def gen_reoports(bench, sycl_dev, apps):
    repo = 'cuda' if bench=='cuda' else 'sycl' 
    sycl_dev_options = ('--sycl=gpu') if bench!='cuda' else ('')
    for app in apps:
        if app in image_apps:
            command = 'ncu  --set full -f -o ../reports/'+bench+'_report_'+app+ ' ../'+repo+'/build/'+repo+'/'+ app + ' ../img/Lenna512.png  ../img/'+app+'_out.png'+ ' ' + sycl_dev_options 
        else:
            command = 'ncu  --set full -f -o ../reports/'+bench+'_report_'+app+ ' ../'+repo+'/build/'+repo+'/'+app + ' '+ sycl_dev_options
            
        os.system(command)

if len(sys.argv) != 3:
    print("Specify sycl impl and sycl device (cpu or gpu) as command line argument")
    exit(-1)

sycl_impl = sys.argv[1]
sycl_dev = sys.argv[2]


gen_reoports('cuda', '', cuda_apps)
gen_reoports(sycl_impl, sycl_dev,sycl_apps)
