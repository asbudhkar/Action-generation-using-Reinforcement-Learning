import numpy as np 
import csv
import matplotlib.pyplot as plt
axes = plt.gca()
#axes.set_xlim([-1,3])
#axes.set_ylim([- 1,0.5])
file1=np.loadtxt(open("out.csv", "rb"), delimiter=",", skiprows=0)
print(file1)
for mat in file1:
	i=0
	#mat=file1
	flag=True
	if(flag):
		print(mat)
		i=70
		flag=False
	#if i<=-1:
	#	i+=1
	#	continue
		
	#for i in range(0,75,3):
	 #   print(i)
	   # v=[]
	    #v = np.array([mat[i],mat[i+1]])
	    #plt.plot([mat[i]],[mat[i+1]],marker='p',markersize=3,color="red")
        #plt.plot([x], [y], marker='o', markersize=3, color="red")
        #print(v.shape)
        #print(arr1.shape)
	    #v=np.reshape(v,(1,2))
	    #arr1=np.append(arr1,v,axis=0)
	    #print(np.shape(arr1))
		plt.plot([mat[0],mat[3]],[mat[1],mat[4]],marker='o', markersize=3, color="red")
		plt.plot([mat[3],mat[6]],[mat[4],mat[7]],marker='o', markersize=3, color="red")
		plt.plot([mat[6],mat[9]],[mat[7],mat[10]],marker='o', markersize=3, color="red")
		plt.plot([mat[6],mat[12]],[mat[7],mat[13]],marker='o', markersize=3, color="red")
		plt.plot([mat[6],mat[12]],[mat[7],mat[13]],marker='o', markersize=3, color="red")
		plt.plot([mat[12],mat[15]],[mat[13],mat[16]],marker='o', markersize=3, color="red")
		plt.plot([mat[15],mat[18]],[mat[16],mat[19]],marker='o', markersize=3, color="red")
		plt.plot([mat[18],mat[21]],[mat[19],mat[22]],marker='o', markersize=3, color="red")
		plt.plot([mat[6],mat[24]],[mat[7],mat[25]],marker='o', markersize=3, color="red")
		plt.plot([mat[24],mat[27]],[mat[25],mat[28]],marker='o', markersize=3, color="red")
		plt.plot([mat[27],mat[30]],[mat[28],mat[31]],marker='o', markersize=3, color="red")
		plt.plot([mat[30],mat[33]],[mat[31],mat[34]],marker='o', markersize=3, color="red")
		
		plt.plot([mat[0],mat[36]],[mat[1],mat[37]],marker='o', markersize=3, color="red")
		plt.plot([mat[36],mat[39]],[mat[37],mat[40]],marker='o', markersize=3, color="red")
		plt.plot([mat[39],mat[42]],[mat[40],mat[43]],marker='o', markersize=3, color="red")
		plt.plot([mat[42],mat[45]],[mat[43],mat[46]],marker='o', markersize=3, color="red")
		plt.plot([mat[0],mat[48]],[mat[1],mat[49]],marker='o', markersize=3, color="red")
		plt.plot([mat[48],mat[51]],[mat[49],mat[52]],marker='o', markersize=3, color="red")
		plt.plot([mat[51],mat[54]],[mat[52],mat[55]],marker='o', markersize=3, color="red")
		plt.plot([mat[54],mat[57]],[mat[55],mat[58]],marker='o', markersize=3, color="red")
		plt.plot([mat[21],mat[63]],[mat[22],mat[64]],marker='o', markersize=3, color="red")
		plt.plot([mat[21],mat[66]],[mat[22],mat[67]],marker='o', markersize=3, color="red")
		plt.plot([mat[33],mat[69]],[mat[34],mat[70]],marker='o', markersize=3, color="red")
		plt.plot([mat[33],mat[72]],[mat[34],mat[73]],marker='o', markersize=3, color="red")
		plt.show()
		plt.close()
		plt.clf()
