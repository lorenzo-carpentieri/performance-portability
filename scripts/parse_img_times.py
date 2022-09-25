import csv
import statistics as stat
 
def read_csv(name, times):
    # opening the CSV file
    with open(name, mode ='r')as file:
    
      # reading the CSV file
      csvFile = csv.reader(file, delimiter=',')
    
      # displaying the contents of the CSV file
      for lines in csvFile:
            if len(lines) == 2 :
                times.append(lines[1])


def compute_mean(times, apps):
    data = []
    app_id=0
    for idx, time in enumerate(times):
        if (idx+1) % 5 != 0:
            data.append(float(time))
        else:
            data.append(float(time))
            gm = stat.geometric_mean(data)
            m = stat.mean(data)
            dev = stat.stdev(data)
            print(""+ apps[app_id])
            print("geometric mean, "+ str(round(gm, 3)))
            print("arithmetic mean, "+ str(round(m, 3)))
            print("standar deviation, "+ str(round(dev,3))+ "\n")
            data=[]
            app_id += 1



NUM_RUN = 5
img_apps = ["box_blur", "box_blur_local_memory",  
            "sobel_filter"]


times_cuda = []
times_sycl= []           

read_csv("../results/cuda_img_times.csv", times_cuda)
read_csv("../results/sycl_img_times.csv", times_sycl)

print("\n"+ "CUDA"+ "\n")
compute_mean(times_cuda, img_apps)

print("\n"+ "SYCL"+ "\n")
compute_mean(times_sycl, img_apps)