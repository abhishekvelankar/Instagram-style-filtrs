import cv2
import numpy as np
def dummy(val):
	pass

#kernels
identity_kernel = np.array([[0,0,0],[0,1,0],[0,0,0]])
sharpen_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
gaussian_kernel1 = cv2.getGaussianKernel(3, 0)
gaussian_kernel2 = cv2.getGaussianKernel(5, 0)
box_kernel = np.array([[1,1,1],[1,1,1],[1,1,1]], np.float32) / 9
#creating array of kernels
kernels = [identity_kernel, sharpen_kernel, gaussian_kernel1, gaussian_kernel2, box_kernel]
#taking image file as input
original= cv2.imread('football.jpg')

modified = original.copy()

gray_original = cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)
gray_modified = gray_original.copy()


#new window
cv2.namedWindow('APP')


#trackbar (trackbar name,window name, min value, max value, callback value(func))
cv2.createTrackbar('contrast','APP',1,20,dummy)
cv2.createTrackbar('brightness','APP',50,100,dummy)
cv2.createTrackbar('filter','APP',0,len(kernels)-1,dummy)
cv2.createTrackbar('grayscale','APP',0,1,dummy)

count = 1

#infinite loop waiting for key q. once pressed window will close
#main UI loop
while True:
	#greyscale code
	grayscale = cv2.getTrackbarPos('grayscale', 'APP')

	if grayscale == 0:
		cv2.imshow('APP',modified)
	else:
		cv2.imshow('APP',gray_modified)
	#save and quit code
	k = cv2.waitKey(1) & 0xFF
	if k == ord('q'):
		break
	elif k == ord('s'):
		if grayscale == 0:
			cv2.imwrite('out%d.png' % count,modified)
		else:
			cv2.imwrite('out%d.png' % count,gray_modified)
		count = count + 1
	
	contrast = cv2.getTrackbarPos('contrast','APP')
	brightness = cv2.getTrackbarPos('brightness','APP')
	kernel = cv2.getTrackbarPos('filter', 'APP')
	
#convolution:applying kernals to the images.
	modified = cv2.filter2D(original, -1, kernels[kernel])
	gray_modified = cv2.filter2D(gray_original,-1,kernels[kernel])


	#print kernels[kernel_value]
	modified = cv2.addWeighted(modified,contrast,np.zeros(original.shape,dtype=original.dtype),0,brightness-50)
	gray_modified = cv2.addWeighted(gray_modified,contrast,np.zeros(gray_original.shape,dtype=gray_original.dtype),0,brightness-50)


#destroy window
cv2.destroyAllWindows()
