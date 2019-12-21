from PIL import Image
from numpy import clip
import cv2

import numpy as np
import matplotlib.pyplot as plt
#M=int(input("enter no of associative memories"))
m=int(input('enter weight matrix1  no of rows'))
n=int(input('enter weight matrix1 no of  columns'))

r=int(input('enter threshold matrix1 no of rows'))
s=int(input('enter threshold matrix1 no of columns'))
#t=int(input("enter threshold matrix2 no of rows"))
#u=int(input("enter threshold matrix2 no of columns"))
print('enter weight1 elements')
#weights1 = [[int(input())for j in range(0,n)]for i in range(0,m)]
weights1 = [[0,1,2,3,1],[1,0,2,4,1],[2,2,0,1,3],[3,4,1,0,1],[1,1,3,1,0]]
print(weights1)


print('enter threshold matrix1')
#threshold1 = [[float(input()) for j in range (0,s)] for  i in range(0,r)] 
threshold1 = [[0.1,0.2,0.3,0.4,0.5],[0.5,0.3,0.2,0.1,0.4],[0.9,0.1,0.3,0.2,0.7],[0.8,0.7,0.6,0.1,0.2],[0.1,0.2,0.3,0.5,0.4]]
print(threshold1)

v =[]                     
img=cv2.imread('C://Users//JYOTHI PRASANNA//Desktop//3.jpg')
grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
(thresh,blackAndWhiteImage) = cv2.threshold(grey,127,255,cv2.THRESH_BINARY)
cv2.imshow('B & W',blackAndWhiteImage)
cv2.imshow('Gray',grey)
cv2.imshow('Original',img)
iar = np.asarray(blackAndWhiteImage/255)
iar = np.where(iar==0,-1,iar)
print(iar)

v.append(iar)
print(iar)
b= np.matrix([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])     

j=2**m            
for i in range(0,j):
    v.append(0)
    v[i+1] = np.sign((np.matmul(weights1,v[i]))-threshold1)
    z = v[i+1]
    z = np.sign((np.matmul(weights1,z))-threshold1)
    if(z - iar).all() == b.all():
        print("stable state")
        print("z_matrix",z)
        print("i",i+1)
    else:
        print("unstable state")    
 
iar1= np.where(iar==-1,0,iar) 
iar1=np.where(iar1==1,255,iar1)        
print(iar1)        
import matplotlib 

matplotlib.image.imsave('name.png',iar1) 

cv2.waitKey(0)
cv2.destroyAllWindows()	    