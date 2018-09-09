## Advanced Lane Finding Project

### Udacity Self-driving car engineer nanodegree - term 1, project 5

---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration1.jpg "Original image"
[image2]: ./output_images/test_undistort.jpg "Undistorted"
[image3]: ./output_images/undistorted0.jpg "Road Transformed"
[image4]: ./output_images/thresholded0.jpg "Binary Example"
[image5]: ./output_images/thresholded_warped0.jpg "Warp Example"
[image6]: ./output_images/thresholded_warped_pixels0.jpg "Fit Pixels with window"
[image7]: ./output_images/thresholded_warped_polys0.jpg "Fit lane polys"
[image8]: ./output_images/final_image0.jpg "Final view"
[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the `01_camera_calibration.py` file. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the 9x6 chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

Original image
![alt text][image1]

Undistorted image
![alt text][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images to obtain a result like this one:
![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 68 through 91 in `02_image_generation.py`).  

I used the examples given in the coursework to convert the colour space to HLS, apply Sobel to L channel and applying a threshold to values between 20-100.  Then applying a threshold of 170-255 to the colour channel before stacking these two results to obtain a binary image like the one below:

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 110 through 131 in the file `02_image_generation.py`.  The `warper()` function takes as inputs an image (`img`) and uses fixed source (`src`) and destination (`dst`) points in the following manner:

```python
src = np.float32(
        [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
        [((img_size[0] / 6) - 10), img_size[1]],
        [(img_size[0] * 5 / 6) + 60, img_size[1]],
        [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
    dst = np.float32(
        [[(img_size[0] / 4), 0],
        [(img_size[0] / 4), img_size[1]],
        [(img_size[0] * 3 / 4), img_size[1]],
        [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

Using this process results in the following 'warped' image, giving a bird's eye view of the binary lane image from before:

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The process of fitting polynomials to the lane lines begins with finding the lane pixels (`find_lane_pixels()` from line 133 - 214 of `02_image_generation.py`).  This uses a sliding-window approach to find the likely lane pixels as peaks in the histogram of the image.  Iterating up the image in this fashion results in the lane pixels being identified as shown in the image below.
![alt text][image6]

Note that when this is used for video, subsequent frames are fitted using the `search_around_poly()` function (line 293-354 of `02_image_generation.py`) to narrow the search to within 100 pixels of the previous polynomial.  This speeds up the processing and avoids jittering due to any unexpected peak on an area of the road unlikely to represent a lane line.

Once the pixels have been identified, we can fit a polynomial to each lane line to find the best curve using `np.polyfit()`.  An example of these resulting two curves with the lane area filled is shown below:
![alt text][image7]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The curvature and position are measured in the `measure_curvature_real()` function (lines 356-379 of `02_image_generation.py`).  This is done using an estimation of pixels to metres in the x and y dimension based on knowledge of the image resolution and lane widths (30/720 in y dimension, 3.7/700 in x dimension), and applying this to the formula for radius of curvature.

The vehicle position (offset with respect to centre of the lane) is calculated by comparing the centre of the identified lane (the average of the starting position of the left and right lanes) and subtracting 640 pixels (the centre of the 1280 wide image, given the camera is mounted centrally).  Using the same pixels-to-metres x dimension value, we can convert this to a real world offset from the centre of the lane.  See the section below for example output.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Finally the `draw_lines()` function (lines 381 - 406) draws the lane lines and lane area onto the warped image, and then unwarps the image back to its original perspective using Minv.  This function also outputs the 'radius of curvature' and 'offset from centre' values for this frame:

![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4).  

Discussion of techniques used to obtain smooth performance on the video are included in the section below.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I faced two main challenges in getting acceptable performance on the video stream.

#### 1. The lane area would sometimes shift to pick up incorrect areas such as the side of the road rather than the left lane line, especially on the concrete or section

The best solution to this issue was to use the `search_around_poly()` function so that I limited my lane search only to pixels within 100px of the polynomial found in the previous frame.  This avoided the lane being found in a completely incorrect position, but the lines were still jittery.

#### 2. The lane areas were prone to jittering, especially at the beginning and end of the curves

In order to avoid sudden changes in the polynomial fitted between consecutive frames, I defined a `Line` class which would store the fitted values for the last x frames, as well as the resulting radius of curvature, and could return mean values over this many frames.  I found that a value of 10 worked well, such that when I took the mean of each fit over this many frames, the sudden jitters were eliminated but it was not so long that the curve appeared unresponsive to the road conditions.

---

Along with these challenges, there are several areas the pipeline could be improved:

#### 1.  Implement a sanity check per frame

Checking that the fit values on each new frame are within a certain threshold of the prior frame(s) would allow bad fits to be ignored completely, rather than just smoothed due to averaging.  I didn't find this necessary on the video provided, but it would likely improve the robustness of the output.

#### 2. Avoid hard-coding of perspective transforms and reliance on image/video resolution

Currently this code is extremely dependent on the footage remaining the same resolution in order for the perspective transforms and calculations for things like radius of curvature to be correct.  This would cause it to return invalid results for other resolutions or camera types.

#### 3. Reliance on a good first frame

If the lane lines are poorly detected in the very first frame, the subsequent searches within 100px offset will perform similarly poorly and degrade the quality of the video output.  The sanity checks outlined in part 1, allowing a full windowed search of the image again for a poor result would address this.

#### 4. More variety of lighting and road conditions

The generation of the binary image works well for the images and video used in this project, but a larger variety of images and video of road surfaces and more difficult situations like night, rainy reflections or headlight-lit roads would be needed to be truly robust.  In addition, vehicles ahead moving in and out of the lane could impede the view of the lane lines and cause the polynomial fits to be poor for those frames.

#### 5. Code refactoring

Given more time, refactoring this large `02_image_generation.py` file to make it easier to read and inserting some flags to activate/deactivate some of the interim image processing or line averaging (when processing many concurrent but unrelated images) would be desirable.
