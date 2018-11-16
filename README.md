# Advanced-Lane-Finding-Project
Udacity’s Self Driving car Nano-Degree project 2 || Detect the Lane Lines of Roads using python and OpenCV
# Self-Driving Car Engineer Nanodegree Program
---
**Advanced Lane Finding Project using OpenCv**

The goals / steps of this project are the following:
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle
position.

[//]: # (Image References)
[im00]:./camera_cal/calibration1.jpg "Chessboard Original Image"
[im01]:./output_images/image0.png "Chessboard calibration"
[im02]:./output_images/image1.png "Undistorted Chessboard"
[im03]:./test_images/test1.jpg "Undistorted Image"
[im04]:./output_images/image4.png "Undistorted Image"
[im05]:./output_images/plot.png "Colorspace Exploration"
[im06]:./output_images/Sobel.png "Sobel"
[im07]:./output_images/Sobel_Magnitude_Direction_Threshold.png "Sobel Magnitude & Direction"
[im08]:./output_images/Pipeline.png "Processing Pipeline for All Image"
[im09]:./output_images/image10.png "Unwarped Image"
[im10]:./output_images/image11.png 
[im11]:./output_images/image12.png
[im12]:./output_images/image13.png
[im13]:./output_images/image14.png

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first two code cells of the Jupyter notebook 
`Advanced Lane Line Project.ipynb`.  

The OpenCV functions `findChessboardCorners` and `calibrateCamera` are the backbone of the image calibration. We have many number of images of a chessboard which taken from different angles with the same camera. we have to identify the number of chessboard corners in corresponding to the location (essentially indices) of internal corners, the pixel locations of the internal chessboard corners determined by `findChessboardCorners`, are fed to `calibrateCamera` which returns camera calibration and distortion coefficients. These can then be used by the OpenCV `undistort` function to undo the effects of distortion on any image produced by the same camera. Generally, these coefficients will not change for a given camera (and lens). 

![alt text][im00]

The image below depicts the results of applying `undistort`, using the calibration and distortion coefficients, to one of the chessboard images:

![alt text][im02]

I then used the output objpoints and imgpoints to compute the camera calibration and distortion coefficients using the cv2.calibrateCamera() function. I applied this distortion correction to the test image using the cv2.undistort() function and obtained this result:

![alt text][im01]

## Pipeline (single images)
#### 1. Provide an example of a distortion-corrected image.
The image below depicts the results of applying `undistort` to one of the project dashcam images:

![alt text][im03]:
![alt text][im04]:

The effect of `undistort` is subtle, but can be perceived from the difference in shape of the hood of the car at the bottom corners of the image.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I have used various combination of color and gradient thresholds to generate a binary image where the lane lines are clearly visible. I have used gradient on x and y direction, manginutude of the gradient, direction of the gradient, and color transformation technique to get the final binary image.

![alt text][im06]

The below image shows the various channels of three different color spaces for the same image:

![alt text][im05]

Below is an example of the combination of sobel magnitude and direction thresholds:

![alt text][im07]:

Ultimately, I chose to use just the L channel of the HLS color space to isolate white lines and the B channel of the LAB colorspace to isolate yellow lines. I did not use any gradient thresholds in my pipeline. I did, however finely tune the threshold for each channel to be minimally tolerant to changes in lighting. As part of this, I chose to normalize the maximum values of the HLS L channel and the LAB B channel (presumably occupied by lane lines) to 255, since the values the lane lines span in these channels can vary depending on lighting conditions. 

Below are the results of applying the binary thresholding pipeline to various sample images:

![alt text][im08]:

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
A perspective transform maps the points in a given image to different, desired, image points with a new perspective. The perspective transform you’ll be most interested in is a bird’s-eye view transform that let’s us view a lane from above; this will be useful for calculating the lane curvature.

Opencv provide two functions getPerspectiveTransform and warpPerspective to perform this task.

![alt text][im09]

 I chose to hardcode the source and destination points in the following manner:

```
 src = np.float32([(575,448),
                  (730,448), 
                  (200,682), 
                  (1100,682)])

dst = np.float32([(450,0),
                  (img.shape[1]-450,0),
                  (450,img.shape[0]),
                  (img.shape[1]-450,img.shape[0])])                 
                  
```

### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
After applying calibration, thresholding, and a perspective transform to a road image, you should have a binary image where the lane lines stand out clearly. However, you still need to decide explicitly which pixels are part of the lines and which belong to the left line and which belong to the right line.

I first take a histogram along all the columns in the lower half of the image.
```histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)```

With this histogram I am adding up the pixel values along each column in the image. In my thresholded binary image, pixels are either 0 or 1, so the two most prominent peaks in this histogram will be good indicators of the x-position of the base of the lane lines. I can use that as a starting point for where to search for the lines. From that point, I can use a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame.

![alt text][im10]

The functions `fit_polynomial` and `fit_poly`, which identify lane lines and fit a second order polynomial to both right and left lane lines, are clearly labeled in the Jupyter notebook as "fit_polynomial" and "fit_poly".

The `fit_poly` function performs basically the same task, but alleviates much difficulty of the search process by leveraging a previous fit (from a previous video frame, for example) and only searching for lane pixels within a certain range of that fit. The image below demonstrates this - the green shaded area is the range from the previous fit, and the yellow lines and red and blue pixels are from the current image:

![alt text][im11]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

We say the curve and the circle osculate (which means “to kiss”), since the 2 curves have the same tangent and curvature at the point where they meet.

The radius of curvature of the curve at a particular point is defined as the radius of the approximating circle. This radius changes as we move along the curve. we using this line of code to calculate Radius of Curvature and Distance from Lane Center.
```
curve_radius = ((1 + (2*fit[0]*y_0*y_meters_per_pixel + fit[1])**2)**1.5) / np.absolute(2*fit[0])
```
The position of the vehicle with respect to the center of the lane is calculated with the following lines of code:
```
lane_center_position = (r_fit_x_int + l_fit_x_int) /2
center_dist = (car_position - lane_center_position) * x_meters_per_pix
```
`r_fit_x_int` and `l_fit_x_int` are the x-intercepts of the right and left fits, respectively. This requires evaluating the fit at the maximum y value (719, in this case - the bottom of the image) because the minimum y value is actually at the top (otherwise, the constant coefficient of each fit would have sufficed). The car position is the difference between these intercept points and the image midpoint (assuming that the camera is mounted at the center of the vehicle).

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the code cells titled "Draw the Detected Lane Back onto the Original Image" and "Draw Curvature Radius and Distance from Center Data onto the Original Image" in the Jupyter notebook. A polygon is generated based on plots of the left and right fits, warped back to the perspective of the original image using the inverse perspective matrix `Minv` and overlaid onto the original image. The image below is an example of the results of the `draw_lane` function:

![alt text][im12]

Below is an example of the results of the `draw_data` function, which writes text identifying the curvature radius and vehicle position data onto the original image:

![alt text][im13]
