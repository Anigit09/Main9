**OPENCV**:-
a large open source module used to modify images (like color correction , resizing , cropping , perspective etc....) it does so by analysing a particular image as a matrix and takes in the input of that image matrix also called an src
In color correction firstly we must understand that the cv2 module reads the image using the command "cv2.imread()" in the form of BGR which can then be converted to RGB or HSV or GRAY etc.....

What does an image matrix represent , it essentially represents a matrix with rows representing y-axis and start from top while the columns represent the x-axis , the image.shape will give us the (width,height) tuple which would be the matrix size with each element containing the intensity of the color (0 being least intense and 255 being most intense)
