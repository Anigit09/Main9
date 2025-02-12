**OPENCV**:-
a large open source module used to modify images (like color correction , resizing , cropping , perspective etc....) it does so by analysing a particular image as a matrix and takes in the input of that image matrix also called an src
In color correction firstly we must understand that the cv2 module reads the image using the command "cv2.imread()" in the form of BGR which can then be converted to RGB or HSV or GRAY etc.....

What does an image matrix represent , it essentially represents a matrix with rows representing y-axis and start from top while the columns represent the x-axis , the image.shape will give us the (width,height) tuple which would be the matrix size with each element containing the intensity of the color (0 being least intense and 255 being most intense)


**TASK 2**
1.CANNY EDGE DETECTION:- This is used to detect the edges in an image and it does this after the image has been smoothened using the 5X5 gaussian filter which is a 5x5 matrix that is also called as a kernel ( whose general form is a 3X3 matrix) now this kernel is convolved with 25 pixels and the avg is taken by taking their sum (as the matrix is aldready multiplied by 1/25) and the avg intensity value replaces the central pixel of the image upon iterating this process we can reduce the noise in the image.

  ![image](https://github.com/user-attachments/assets/85eabaa4-c9a6-4b8e-943e-adcb52161e05)


Next step is edge gradient calculation (Edge gradient = sqrt(Gx^2+Gy^2) and tan(theta)=Gy/Gx) this will calculate the edge gradient vector that is always perpendicular to the edge and after calculation of which we can pass it to a filter.
NON MAXIMUM SUPPRESSION:-What this does is that it checks if the value of the pixel is local maximum or not and if it is then it passes else it is supressed/set to 0,this inturn helps us make the image sharper by chosing only higher intensity pixels . The image below shows point A ,B ,C where A is on the edge while B and C are shown on the gradient vector perpendicular to the edge if A is the local maxima i.e., it has the highest intensity among all three then it passes 

  ![image](https://github.com/user-attachments/assets/4466cc20-a48c-46b9-b50f-9e03dadd3ca6)

HYSTERSIS THRESHOLD:-is another method important to us where pixel filteration is done by taking two threshold values maxval and minval if an edge is above the maxval its called a "sure-edge" any edge that is connected to the sure-edge and is above minval (even though it may not be above maxval) is also considered as an edge and passed where as any other edge below minval or in b/w min and maxval are rejected/suppressed an example to this is shown below where A is a sure-edge , C is connected to A and B is suppressed this helps remove those small pixel noises.

  ![image](https://github.com/user-attachments/assets/3bd7126c-b4d4-4780-9b49-21646a07e794)

In opencv we use the function cv2.Canny(src/img,minval,maxval,aperture_size,L2gradient) where aperture_size is the size of the kernel to be used by default it is set to 3 (i.e., 3X3 matrix) where as the L2gradient takes a boolean type True meaning it uses the above eqn to calculate the edge gradient and False meaning that G=|Gx|+|Gy| is used. By default it is set to False.

*HISTOGRAMS*:- Like we know from pandas histograms are basically a type of plot used in data handeling and it carries out the same function in image processing where Histograms are just a plot of no.of pixels(in the Y-axis) vs intensity (in the X-axis) with the intensity values ranging from [0-255].Now to find histograms we need either all the 256 values that is no.of pixels in each intensity or we can divide them into 16 intervals such that we only would end up needing 16 such values where each value is the sum of no.of pixels with the intensity in that interval range which will overall reduce the workload. However if the no.of bins are set 256 we get a more detailed plot.
The calculation of histograms can be done either through the numpy functions like hist,bins= np.histogram(img.flatten)/img.ravel(),[bins],[range])/np.bincount(img.flatten(),minlength=256(Default)) where in the latter one is 10X faster . Here the key difference between img.flatten() and img.ravel() is that img.flatten() will create a new flattened array of the original image while img.ravel() dosent create a new array but just flattens the original image.Other alternative that is almost 40X faster is using the opencv module cv2.calcHist([img],[Channel],mask,[bins],[range]) all of them must be enclosed in [] like shown and if we want histogram for the complete plot then we need to set mask=None else we specify the masked array , [bins] and [range] can be set to 256 , [0,256].
Histograms can be plotted either using matplotlib's plt.hist(img.ravel(),bins,[range]) which calculates the histogram and plots it else we can use the opencv's drawing function which would be way complex.
Application of Mask:- Masking is used to confine a certain region of image that we want we can do this by specifying our ROI for ex:- img[0:200,200:400]=mask is my ROI here.The basics steps for masking an image is to use the cv2.bitwise_and() operator that accepts two images and a mask to be applied.
If i do the following mask=np.zeros(img.shape[:2],np.uint8) this will access first two dimensions (heightXwidth) ignoring the color channel (if any) and will set it to 0, next would be assigning ROI like we did above but this time we will do mask[100:200,400:600]=255 and we will do cv2.bitwise_and(img,img,mask=mask) this will take common b/w both images and set all values where mask is 0 is set to zero in the original image and pixel values are only kept where mask has non-zero (255) as the value set.
img = cv.imread('home.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
 
mask = np.zeros(img.shape[:2], np.uint8)
mask[100:300, 100:400] = 255
masked_img = cv.bitwise_and(img,img,mask = mask)
 
hist_full = cv.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv.calcHist([img],[0],mask,[256],[0,256])
 
plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask,'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0,256])
 
plt.show()

  ![image](https://github.com/user-attachments/assets/400c5887-6b21-4fbe-b51f-5f3b9cfb6cad)

*HISTOGRAM EQUALIZATION*:- This process is used to increase a clarity of image by playing with its contrast , it mainly makes use of two variables one is the histogram itslef and the other is cdf(cumulative distribution function) which takes the cumilative sum of the histogram and stores it in a 1-D array.
So basically if an image is confined only to a specific range of intensities like a darker image or lighter image then we would want to increase the clarity of the image and this is done by streching of the histogram i.e., distribution of pixels to various intensities.
We will using the numpy's masked array concept that allows us to mask an element in the array and perform operations on the remaining ones (np.ma.masked_equal())
cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')
the following code will first create a masked array which excludes zeros of the cdf array and will only perform operation on the non-zero parts, the next line will set the range of cdf values to [0,255] (understand it in this way iam subtracting cdf array with its min element so if i subtract the minimum element in the array with itself we indirectly set it to 0 whereas if we reach the max element in the array and subtract it from the minimum element we would get 1 as we divide it by the same and 1*255=255 so if min and max are set to 0,255 then rest all values will be in b/w [0,255]) and finally we will fill all the originally masked 0's with 0 to complete the cycle.
img2=cdp[img]
plt.imshow(img2)
will finally display our output in the form of high contrast image.
In opencv however we will make use of the function cv.equalizeHist() that takes src as input and returns a high contrast image.
But sometimes due to differently lit areas we might not get a clear picture visbility i.e., Global histogram equalization doesnt work for all conditons hence will use the methond **CLAHE(Contrast Limited Adaptive Histogram Equalization)** this method divides the source image into smaller tiles (by default 8X8 ) and applies histogram equalization to each tile , but if each tile isnt noise free then the noise gets amplified hence will apply contrast limiting (like if 40 is my contrast limit then any bin with more than 40 values will be split and distributed across the image) this will reduce the noise of the image and finally we will apply bilinear interpolation to complete the process .
cv2.createCLAHE() takes 2 arguments one being the clipLimit (if it is set to 2.0 then all the bins with more than 2*average pixels will be clipped) and tileGridSize (that is set to 8X8 by default) , we also use clahe.apply() to apply the following clahe framework that we created to an source.


  ![image](https://github.com/user-attachments/assets/44e5e02f-7e19-41c7-a18c-c6e4bba93ec2)



  ![image](https://github.com/user-attachments/assets/4af48917-d987-48c7-8ac8-9e88a2abd5e3)





