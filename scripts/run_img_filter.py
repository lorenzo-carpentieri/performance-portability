import os
import sys
N_RUN = 5

def run_bench(image_apps, is_cuda, sycl_dev, img_size): 
    bench = "sycl" if is_cuda == 0 else "cuda"

    sycl_dev_options = ('--sycl='+sycl_dev) if is_cuda==0 else ('')
    for name in image_apps:
        #move in build/cuda|sycl directory
        cd_command = 'cd '+ '../'+bench+'/build/'+bench+'/ && ' + './'+ name
        # run all code in apps N_RUN times
        for i in range(N_RUN):
            #executio command
            exe_command= ' ../../../img/Lenna'+str(img_size)+'.png ../../../img/Lenna'+str(img_size)+'_output.png '+ sycl_dev_options +' >> ../../../results/'+bench+'_img_times.csv'
           
            os.system(cd_command + exe_command)
        
        os.system('python -c \'print "\\n"\' >> ../results/'+bench+'_img_times.csv')



image_apps = ["box_blur", "box_blur_local_memory", "sobel_filter"]

#remove old files
os.system('rm -f ../results/cuda_img_times.csv')
os.system('rm -f ../results/sycl_img_times.csv')

#cpu or gpu
if len(sys.argv) != 3:
    print("Specify sycl dev (cpu or gpu) as command line argument")
    exit(-1)


sycl_dev = sys.argv[1]
img_size = sys.argv[2]

is_cuda = 1
run_bench(image_apps, is_cuda, '', img_size)

is_cuda = 0
run_bench(image_apps, is_cuda, sycl_dev, img_size)