import time
import math
import numpy as np
import cv2
import rospy

from line_fit import line_fit, tune_fit, bird_fit, final_viz
from Line import Line
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
from skimage import morphology



class lanenet_detector():
    def __init__(self):

        self.bridge = CvBridge()
        # NOTE
        # Uncomment this line for lane detection of GEM car in Gazebo
        # self.sub_image = rospy.Subscriber('/gem/front_single_camera/front_single_camera/image_raw', Image, self.img_callback, queue_size=1)
        # Uncomment this line for lane detection of videos in rosbag

        # REMEMBER: DO NOT PUT IMAGE, PUT THE FOLDER IMAGE LIVES IN
        self.sub_image = rospy.Subscriber('zed2/zed_node/rgb/image_rect_color/', Image, self.img_callback, queue_size=1)
        print('image subscribed')
      
        self.pub_image = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
        self.pub_bird = rospy.Publisher("lane_detection/birdseye", Image, queue_size=1)
        print('publishing image')
        self.left_line = Line(n=5)
        self.right_line = Line(n=5)
        self.detected = False
        self.hist = True


    def img_callback(self, data):

        print('img callback') 
        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            cv2.imwrite("test1.png", cv_image)
        except CvBridgeError as e:
            print(e)

        raw_img = cv_image.copy()
        mask_image, bird_image = self.detection(raw_img)

        if mask_image is not None and bird_image is not None:
            # Convert an OpenCV image into a ROS image message
            out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
            out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')

            # Publish image message in ROS
            self.pub_image.publish(out_img_msg)
            self.pub_bird.publish(out_bird_msg)


    def gradient_thresh(self, img, thresh_min=25, thresh_max=100):
        """
        Apply sobel edge detection on input image in x, y direction
        """
        #1. Convert the image to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #2. Gaussian blur the image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        #3. Use cv2.Sobel() to find derievatives for both X and Y Axis
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize= 3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize= 3)

        #4. Use cv2.addWeighted() to combine the results
        grad_mag = cv2.addWeighted(np.abs(sobelx), 0.5, np.abs(sobely), 0.5, 0)

        #5. Convert each pixel to unint8, then apply threshold to get binary image
        grad_mag = np.uint8(255 * grad_mag / np.max(grad_mag))
        binary_output = np.zeros_like(grad_mag)
        binary_output[(grad_mag >= thresh_min) & (grad_mag <= thresh_max)] = 1
        
        return binary_output

    # def color_thresh(self, img, thresh=(190, 255)):
    #     """
    #     Convert RGB to HSL and threshold to binary image using S channel
    #     """
    #     #1. Convert the image from RGB to HSL
    #     converted_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    #     h_channel = converted_image[:, :, 0]
    #     s_channel = converted_image[:, :, 2]

    #     #2. Apply threshold on S channel to get binary image
    #     binary_output = np.zeros_like(s_channel)
    #     binary_output[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1

    #     #Hint: threshold on H to remove green grass
    #     h_thresh = (22, 43) # trial and error change
    #     binary_output[(h_channel >= thresh[0]) & (h_channel <= h_thresh[1])] = 0

    #     ####

    #     return binary_output

    def color_thresh(self, img, thresh=(190, 255)):
        """
        Convert RGB to HSL and threshold to binary image using S channel
        """
        #1. Convert the image from RGB to HSL
        converted_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        h_channel = converted_image[:, :, 0]
        s_channel = converted_image[:, :, 2]
        l_channel = converted_image[:, :, 1]
        binary_output = np.zeros_like(s_channel)  # Initialize binary_output to the same shape as the input channels

        # Use parentheses to group conditions properly
        # condition = (l_channel < 20) & (s_channel > 115) & ((h_channel > 15) & (h_channel < 40)) & (s_channel > 50) & ((l_channel > 64) & (l_channel < 140))
        condition = (l_channel < 75) & ((h_channel > 70) | (h_channel < 35))

        # Set the binary output based on the condition
        binary_output[condition] = 0
        binary_output[~condition] = 1

        # #2. Apply threshold on S channel to get binary image
        # binary_output = np.zeros_like(s_channel)
        # binary_output[(s_channel >= 220) & (s_channel <= 255)] = 1
        # binary_output[(s_channel < 220)] = 0
        # #Hint: threshold on H to remove green grass
        # h_thresh = (22, 30) # trial and error change
        # binary_output[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1
        # binary_output[(h_channel < h_thresh[0]) & (h_channel > h_thresh[1])] = 0

        # l_thresh = 70 # trial and error change
        # binary_output[(l_channel > l_thresh)] = 1
        # binary_output[(l_channel <= l_thresh)] = 0
        ####

        return binary_output


    def combinedBinaryImage(self, img):
        """
        Get combined binary image from color filter and sobel filter
        """
        #1. Apply sobel filter and color filter on input image
        #2. Combine the outputs
        ## Here you can use as many methods as you want.

        ## TODO
        SobelOutput = self.gradient_thresh(img)
        ColorOutput = self.color_thresh(img)
        ####

        binaryImage = np.zeros_like(SobelOutput)
        binaryImage[(ColorOutput==1)|(SobelOutput==1)] = 1
        img3 = ColorOutput.astype(np.uint8) * 255
        # Remove noise from binary image
        binaryImage = morphology.remove_small_objects(binaryImage.astype('bool'),min_size=50,connectivity=2)
        src = np.float32([[32, 399], [590, 400], [200, 265], [425, 280]])  # Given coordinates
        for point in src:
            cv2.circle(img, (int(point[0]), int(point[1])), 10, (255), -1)  # White circles with radius 10
        return binaryImage

    def perspective_transform(self, img, verbose=False):
        """
        Get bird's eye view from input image
        """

        img = img.astype(np.uint8)

        #1. Visually determine 4 source points and 4 destination points
        img_size = (img.shape[1], img.shape[0])
        # print(img_size)
        src = np.float32([[32, 399], [590, 400], [200, 265], [425, 280]]) # change these values trial and error
        # src = np.float32([[160, 290], [0, 370], [616, 370], [450, 290]]) # change these values trial and error

        # src = np.float32([[150, 275], [75, 400], [500, 400], [500, 275]]) # change these values trial and error
        dst = np.float32([[0, img_size[1]-1], [img_size[0]-1, img_size[1]-1], [0,0], [img_size[0]-1, 0]]) # change these values trial and error
        # dst = np.float32([[0, 0], [0, img_size[0]-1], [img_size[1]-1,img_size[0]-1], [img_size[1]-1, 0]])

        #2. Get M, the transform matrix, and Minv, the inverse using cv2.getPerspectiveTransform()
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)

        #3. Generate warped image in bird view using cv2.warpPerspective()
        warped_img = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

        if verbose:
            print("Perspective transform matrix (M):\n", M)
            print("Inverse perspective transform matrix (Minv):\n", Minv)

        ####

        return warped_img, M, Minv


    def detection(self, img):

        binary_img = self.combinedBinaryImage(img)
        img_birdeye, M, Minv = self.perspective_transform(binary_img)

        if not self.hist:
            # Fit lane without previous result
            ret = line_fit(img_birdeye)
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            nonzerox = ret['nonzerox']
            nonzeroy = ret['nonzeroy']
            left_lane_inds = ret['left_lane_inds']
            right_lane_inds = ret['right_lane_inds']

        else:
            # Fit lane with previous result
            if not self.detected:
                ret = line_fit(img_birdeye)

                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    left_lane_inds = ret['left_lane_inds']
                    right_lane_inds = ret['right_lane_inds']

                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)

                    self.detected = True

            else:
                left_fit = self.left_line.get_fit()
                right_fit = self.right_line.get_fit()
                ret = tune_fit(img_birdeye, left_fit, right_fit)

                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    left_lane_inds = ret['left_lane_inds']
                    right_lane_inds = ret['right_lane_inds']

                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)

                else:
                    self.detected = False

            # Annotate original image
            bird_fit_img = None
            combine_fit_img = None
            if ret is not None:
                bird_fit_img = bird_fit(img_birdeye, ret, save_file=None)
                combine_fit_img = final_viz(img, left_fit, right_fit, Minv)
            else:
                print("Unable to detect lanes")

            return combine_fit_img, bird_fit_img


if __name__ == '__main__':
    # init args
    rospy.init_node('lanenet_node', anonymous=True)
    lanenet_detector()
    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.5)
