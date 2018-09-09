import numpy as np
import cv2
import glob
import pickle
from moviepy.editor import VideoFileClip

# Define a Line class to hold aggregate information across frames to smooth output
class Line():
    def __init__(self, frames):
        # store each polynomial coefficient for the last x lines
        self.poly1 = []
        self.poly2 = []
        self.poly3 = []
        # store the last x curvature radii
        self.curverad = []
        # how many frames should we keep and average across
        self.frames = frames
        self.current_mean_fit = [np.array([False])]
    
    def push_new_fit(self,new_fit):
        # append each polynomial coefficient to its respective list
        # pop the first element off the list if we reach the frame limit
        self.poly1.append(new_fit[0])
        if len(self.poly1) > self.frames:
            self.poly1.pop(0)
        
        self.poly2.append(new_fit[1])
        if len(self.poly2) > self.frames:
            self.poly2.pop(0)

        self.poly3.append(new_fit[2])
        if len(self.poly3) > self.frames:
            self.poly3.pop(0)
    
    def push_new_curverad(self,new_curverad):
        # append the new radius of curvature
        # pop the first element off the list if we reach the frame limit
        self.curverad.append(new_curverad)
        if len(self.curverad) > self.frames:
            self.curverad.pop(0)

    def get_mean_fit(self):
        # return the mean of all stored fits
        meanp1 = np.mean(self.poly1)
        meanp2 = np.mean(self.poly2)
        meanp3 = np.mean(self.poly3)
        self.current_mean_fit = [meanp1,meanp2,meanp3]
        return self.current_mean_fit

    def get_mean_curverad(self):
        # return the mean of all stored curvatures
        return np.mean(self.curverad)

# Open the pickle of camera calibration data
dist_pickle = pickle.load(open("calibration_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Create our line objects, averaging over 10 frames
left_line = Line(10)
right_line = Line(10)

# Our main pipeline to transform and output images
def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    
    img = np.copy(img)

    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
   
   # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # Stack each channel (for colour output during testing)
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    # Combined binary b&w output for final
    preprocessImage = np.zeros_like(img[:,:,0])
    preprocessImage[((sxbinary == 1 ) | (s_binary == 1))] = 255

    # Warp this image to a bird's eye view
    warpedImage, MInv = warper(preprocessImage)

    # Check if this is the first frame (no current mean fit values)
    if left_line.current_mean_fit[0] == False:
        # Fit a polynomial to this bird's eye view image using a sliding window technique
        polyImage, left_fitx, right_fitx, ploty, vehicle_offset  = fit_polynomial(warpedImage)
    
    else:
        # Fit a polynomial to this bird's eye view image searching around prior windowed area
        polyImage, left_fitx, right_fitx, ploty, vehicle_offset  = search_around_poly(warpedImage, left_line.get_mean_fit(), right_line.get_mean_fit())

    # Draw our final unwarped image based on the calculated lane information
    final = draw_lines(warpedImage, left_fitx, right_fitx, ploty, MInv, img, vehicle_offset)

    return final

def warper(img):
    # Define the source and destination image points for warping a lane
    # based on points mapped to the lane on a sample unwarped image
    img_size = (img.shape[1],img.shape[0])
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

    # Warp the image into a bird's eye view, and return the Minv so we can
    # unwarp it later for final output
    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    warped = cv2.warpPerspective(img,M,img_size,flags=cv2.INTER_LINEAR)

    return warped, Minv

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

# Used for the first frame only
def fit_polynomial(binary_warped):

    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # push these new fit values to our Line objects
    left_line.push_new_fit(left_fit)
    right_line.push_new_fit(right_fit)

    # For this first frame only, set the mean to the current fit
    left_line.current_mean_fit = left_fit
    right_line.current_mean_fit = right_fit

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ### Visualization - only used in initial test image output  ###
    ### Not part of final image output for video frames         ###

    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Calculate the radius of curvature in pixels for both lane lines
    left_curverad, right_curverad, vehicle_offset = measure_curvature_real(ploty,left_fit,right_fit)

    left_line.push_new_curverad(left_curverad)
    right_line.push_new_curverad(right_curverad)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(out_img, np.int_([pts]), (255,255, 0))

    return out_img, left_fitx, right_fitx, ploty, vehicle_offset

# Used during search_around_poly for frames after the first frame
def fit_poly(img_shape, leftx, lefty, rightx, righty):
    # Fit a second order polynomial to each with np.polyfit()

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Push these new fits and recalculate the mean of the last x frames
    # to use as our overall fit for smoothing purposes
    left_line.push_new_fit(left_fit)
    right_line.push_new_fit(right_fit)

    left_fit = left_line.get_mean_fit()
    right_fit = right_line.get_mean_fit()

    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    # Calc both polynomials using ploty, left_fit and right_fit
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx, ploty

# This is used for each subsequent frame after the first one, to limit the
# window search
def search_around_poly(binary_warped, left_fit, right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Set the area of search based on activated x-values
    # within the +/- margin of our polynomial function
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    # Calculate the radius of curvature in pixels for both lane lines
    left_curverad, right_curverad, vehicle_offset = measure_curvature_real(ploty,left_line.get_mean_fit(),right_line.get_mean_fit())

    # Update curvature radii on each line object
    left_line.push_new_curverad(left_curverad)
    right_line.push_new_curverad(right_curverad)

    ### Visualization - only used in initial test image output  ###
    ### Not part of final image output for video frames         ###

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    return result, left_fitx, right_fitx, ploty, vehicle_offset

def measure_curvature_real(ploty,left_fit_cr,right_fit_cr):
    '''
    Calculates the curvature of polynomial functions in metres.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # Calculation of vehicle offset from centre of the lane based on 
    # image height of 720 and the centre of the image/camera mount being 640px
    left_lane_start = left_fit_cr[0]*720**2 + left_fit_cr[1]*720 + left_fit_cr[2]
    right_lane_start = right_fit_cr[0]*720**2 + right_fit_cr[1]*720 + right_fit_cr[2]
    centre_of_lane = (left_lane_start + right_lane_start) / 2
    vehicle_offset = (centre_of_lane - 640) * xm_per_pix

    return left_curverad, right_curverad, vehicle_offset

def draw_lines(warped, left_fitx, right_fitx, ploty, MInv, undist, vehicle_offset):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, MInv, (undist.shape[1], undist.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    # Add the text for the current mean radius of curvature & vehicle offset being used for this frame
    frame_curverad = "Radius of curvature: " + str(round((left_line.get_mean_curverad() + right_line.get_mean_curverad() / 2),2)) + "m"
    cv2.putText(result,frame_curverad,(20,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)
    
    frame_offset = "Offset from centre: " + str(round(vehicle_offset,2)) + "m"
    cv2.putText(result,frame_offset,(20,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)

    return result

# Frame processing
def process_frame(img):
    img = cv2.undistort(img,mtx,dist,None,mtx)
    result = pipeline(img)
    return result

### Test image processing used while developing pipeline ###
### Not needed for final video output below              ###
# Open test images
#images = glob.glob('test_images/test*.jpg')

# for idx, fname in enumerate(images):
#     img = cv2.imread(fname)

#     result = process_frame(img)

#     write_name = 'test_images/final_image'+str(idx)+'.jpg'
#     cv2.imwrite(write_name, result)

# Open the video and process each frame to an output video
video_output = 'project_video_output.mp4'
clip = VideoFileClip('project_video.mp4')
input_clip = clip.fl_image(process_frame)
input_clip.write_videofile(video_output, audio=False)