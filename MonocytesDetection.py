import cv2
import numpy as np
import matplotlib.pyplot as plt 

# -*- coding: utf-8 -*-
#def nothing(*arg):
#        pass
#        
#cv2.namedWindow('Binary')
#cv2.createTrackbar('thrs1', 'Binary',128,255, nothing)
# Se carga la imagen




img = cv2.imread('Monocytes12.jpg',cv2.COLOR_BGR2GRAY)
Z = img.reshape((-1,3))


Z = np.float32(Z)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3

# ****kmeans **** Para dar un "blur" que permita clasificar mas adelante
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
cv2.imshow('res2',res2)

gray_c = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = res2
KMEAN = img.copy()
cv2.imwrite('1kmean_blur.jpg',img)

#blur = cv2.blur(img,(7,7))
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#gray_b = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)

#************ Se pasa a grises ************
cv2.imwrite('2grayIMG_kblur.jpg',gray)

#***********Binarizacion del gray ****************
ret,binaryIMG = cv2.threshold(gray,220,255,cv2.THRESH_BINARY) #Binarizacion del gray
cv2.imshow('Binary',binaryIMG)
cv2.imwrite('3binaryIMG_blur.jpg',binaryIMG)

#********* Canny ************** Aplico Canny para detección de bordes
edges = cv2.Canny(gray_c, 600, 600, apertureSize=5, L2gradient=True)
cv2.imwrite('4Canny_EDGES.jpg',edges)
cv2.imshow('4Canny_EDGES',edges)


#********  SEPARADOS *********
separados = cv2.bitwise_or(edges,binaryIMG)
cv2.imshow('separados',separados)
cv2.imwrite('5separados.jpg',separados)

separados_inv = cv2.bitwise_not(separados)



# ****** Elemento estructurante elipsoidal (Dilatación)
kernelDILT = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))
dilateIMG = cv2.dilate(separados_inv, kernelDILT, iterations=3)
cv2.imshow('dilate',dilateIMG)
cv2.imwrite('6dilateIMG.jpg',dilateIMG)



# ****** Elemento estructurante elipsoidal (Erosion)
kernel_erosion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
erosionIMG = cv2.erode(dilateIMG, kernel_erosion, iterations=3)
cv2.imshow('erosion',erosionIMG)
cv2.imwrite('7erosionIMG.jpg',erosionIMG)



#r**************** Busqueda de contornos elipsoidales
un,contours,hierarchy = cv2.findContours(erosionIMG, 1, 2) 

ellipse_count = 0
img2 = img.copy()
for i in range(0,len(contours)):

	cnt = contours[i]
	if len(cnt) > 4:
		ellipse = cv2.fitEllipse(cnt)
		(x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
		
		
		if MA > 80 and MA < 400:	
			cv2.ellipse(img2,ellipse,(0,255,0),2)
			ellipse_count = ellipse_count + 1

cv2.imshow('detected Elipsoids',img2)
cv2.imwrite('8detectedElip.jpg',img2) 

print("Circulos encontrados: ",ellipse_count)


#*********** Conteo de area *****************

background_color = np.array([236,228, 222], np.uint8)
Blue1 = np.array([145 - 50, 76 - 50, 92 - 50], np.uint8) #BGR
#Blue = np.array([145, 76, 92], np.uint8) #BGR
Blue2= np.array([145 + 50, 76 + 50, 92 + 50], np.uint8) #BGR
# Red = np.array([134, 132, 162], np.uint8)
Red1 = np.array([134 - 50, 132 - 50, 162 - 50], np.uint8)
Red2 = np.array([134 + 50, 132 + 50, 162 + 50], np.uint8)
#Background 236,228,221

# ************* Zona azulada ************

BLUE_MASK = cv2.inRange(KMEAN,Blue1,Blue2)
cv2.imwrite('9BLUE_MASK.jpg',BLUE_MASK) 
BLUE_NUMBER = cv2.countNonZero(BLUE_MASK)

print("Pixeles azulados: ",BLUE_NUMBER)

# ************* Zona rojiza

RED_MASK = cv2.inRange(KMEAN,Red1,Red2)
cv2.imwrite('10RED_MASK.jpg',RED_MASK) 
RED_NUMBER = cv2.countNonZero(RED_MASK)

print("Pixeles rojizos: ",RED_NUMBER)

# ********** Total de foreground

TOTAL_MASK = RED_MASK + BLUE_MASK
cv2.imwrite('10TOTAL_MASK.jpg',TOTAL_MASK) 
FOREGROUND_NUMBER = cv2.countNonZero(TOTAL_MASK)

print("Total de pixeles de formas: ",FOREGROUND_NUMBER)
print("Total de pixeles: ",gray.size)
print("Porcentaje de Monicitos: %",BLUE_NUMBER*100.0/gray.size)
print("Porcentaje de Globulos rojos: %",RED_NUMBER*100.0/gray.size)
print("Porcentaje de Globulos y monocitos: %",(FOREGROUND_NUMBER*100.0)/gray.size)

while True:
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	break
#print(contours)

#print(contours[1])
