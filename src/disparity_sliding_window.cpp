/**
    disparity_sliding_window.cpp
    Purpose: Implementation of methods needed for calculating object proposals from disparity images

    @author Julian Mueller, University of Ulm
    @version 1.1 02/03/18
*/


#include "disparity_sliding_window.h"
#include <cmath>
#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <fstream>



DisparitySlidingWindow::DisparitySlidingWindow() {

}

/**
    Constructor of DisparitySlidingWindow

    @param obj_width        Objects real world width in meters, e.g. 0.6 meters for pedestrians
    @param obj_height       Objects real world height in meters, e.g. 1.73 meters for pedestrians
    @param hyp_aspect       Aspect ratio (pixels) of the object, e.g. 2-3 for pedestrians. ratio = height/width
    @param min_hyp_width    Minnimum proposal width in pixels
    @param max_hyp_width    Maximum proposal width in pixels
    @param hyp_class_id     Class identity, you can actually ignore this
    @param max_nans         Maximum number of allowed NaNs before proposal is rejected, actually we use 6 pixels for homogeneity thresholding, if you dont want to use this feature set it to 6
    @param max_stddev       Maximum allowed standard deviation (homogeneity verification), good value for pedestrians: 1.0
    @param step_perc        Adaptive step size (percentage of object width, see paper), we propose values between 0.3 and 0.4 for accurate detection results

*/
DisparitySlidingWindow::DisparitySlidingWindow(const float &obj_width, const float &obj_height, const float &hyp_aspect, const int &min_hyp_width, const int &max_hyp_width, const int &hyp_class_id, const size_t &max_nans, const float &max_stddev, const float &step_perc, const int &class_type) :
    obj_width_(obj_width),
    obj_height_(obj_height),
    hyp_aspect_(hyp_aspect),
    min_hyp_width_(min_hyp_width),
    max_hyp_width_(max_hyp_width),
    hyp_class_id_(hyp_class_id),
    max_nans_(max_nans),
    max_stddev_(max_stddev),
    step_perc_(step_perc),
    class_type_(class_type)

{
    this->hyp_.id_ = 0;
    this->lut_adress_factor_ = 0;
    this->homogeneity_verification_method_ = class_type;
}

/**
    Destructor of DisparitySlidingWindow
*/
DisparitySlidingWindow::~DisparitySlidingWindow() {

}

/**
    initLookUpTable:            Method precalculating the size of an object in the image (width, height in pixels) with respect to the calibration and disparity range. This is cheaper than calculating the projection frame by frame.
                                If calibration changes frame by frame, please call this method before processing the file. If calibration is fixed, only call this function once.

    @param tx                   Tx = -fx' * B, where fx is the focal length in x and B is the baseline of the stereo camera. See also here: http://docs.ros.org/jade/api/sensor_msgs/html/msg/CameraInfo.html
    @param camera_matrix        Camera matrix  K of the left camera, 3x3
                                     [fx  0 cx]
                                 K = [ 0 fy cy]
                                     [ 0  0  1]
    @param distortion_matrix    Distortion Matrix D of the left camera, 5x1, D = [k1, k2, t1, t2, k3]
    @param min_disp             Minimum disparity value of the disparity image algo
    @param max_disp             Maximum disparity value of the disparity algo
    @param dist_step            Disparity step, must be a multiple of 2
    @return                     Boolean, false if disparity step is not valid, true if LUT was cauculated

*/
bool DisparitySlidingWindow::initLookUpTable(const float &tx, const cv::Mat &camera_matrix, const cv::Mat &distortion_matrix, const float &min_disp, const float &max_disp, const float &disp_step) {

    // check if inverse disp-step is a multiple of 2
    if (((int)(disp_step/1.))%2 != 0) {
        std::cout << "ERROR:\t Inverse disparity step is not a multiple of two! This lib expects fixed-point floating point disparity images with a sub-decimal precision of n bits." << std::endl;
        return false;
    }

    this->LUT_.clear();
    lut_adress_factor_ = 1./((float)disp_step);

    std::cout << "INFO:\tLUT adress factor: " << lut_adress_factor_ << std::endl;
    std::cout << "INFO:\tCalculating a hypotheses LUT with " << ((max_disp-min_disp)/disp_step)+1 << " entries." << std::endl;

    // prepare real world object points
    std::vector<cv::Point3f> object_points;
    object_points.resize(4);
    object_points[0] = cv::Point3f(0.,0.,0.);
    object_points[1] = cv::Point3f(obj_width_,0.,0.);
    object_points[2] = cv::Point3f(obj_width_,obj_height_,0.);
    object_points[3] = cv::Point3f(0.,obj_height_,0.);

    // prepare other data for cv::projectPoints
    std::vector<cv::Point2f> image_points;
    cv::Mat_<float> tvec = (cv::Mat_<float>(3,1) << 0., 0., 0.);
    cv::Mat_<float> rvec = (cv::Mat_<float>(3,1) << 0., 0., 0.);


    Rect hyp;
    for (float disparity = min_disp; disparity <= max_disp; disparity += disp_step) {

        // calculate depth in meters
        tvec.at<float>(2,0) = -1.* tx / (disparity);

        // project
        cv::projectPoints(object_points, rvec, tvec, camera_matrix, distortion_matrix, image_points);
        // TODO: this assumes that the object is at the image center, it would be a better to use the matching ray!
        //       but this would mean w'd have a sizeof(float) * (cols/step_x) * (rows/step_y) * (disp_range/disp_step) LUT!
        hyp.w_ = (image_points.at(1).x - image_points.at(0).x);    // TODO: should we std::round here?

        hyp.h_ = hyp_aspect_ * hyp.w_;
        hyp.x_ = 0;
        hyp.y_ = 0;
        hyp.classId_ = std::to_string(hyp_class_id_);
        hyp.id_ = LUT_.size();
        hyp.confidence_ = 1.0;
        LUT_.push_back(hyp);

    }

    return true;
}

/**
    generate:                   Calculates object proposals from the disparity input

    @param disparity_image      Disparity image (CV_32FC1)
    @param dst                  Debug image
    @param hyps                 Vector of proposal rectangles
    @param tx                   Tx = -fx' * B, where fx is the focal length in x and B is the baseline of the stereo camera. See also here: http://docs.ros.org/jade/api/sensor_msgs/html/msg/CameraInfo.html

*/
void DisparitySlidingWindow::generate(const cv::Mat &disparity_image, cv::Mat &dst, std::vector<Rect> &hyps, const float &tx) {

    // check if we're ready to generate
    if (LUT_.size() == 0 || lut_adress_factor_ == 0) {
        std::cout << "ERROR:\tYou must call 'calculateLookUpTable' before calling 'generate'!" << std::endl;
        return;
    }

    // check if disparity image is of type float
    if (disparity_image.type() != CV_32FC1) {
        std::cout << "ERROR:\tWe expect a float image as an input!" << std::endl;
        return;
    }

    // adaptive step sizes
    int step_x_adapt = 1;
    int step_y_adapt = 0;

    float stddev;
    size_t nan_cnt;
    float nan_point = std::numeric_limits<float>::quiet_NaN ();
    double dNaN = std::numeric_limits<double>::quiet_NaN();

    // we need this to save the step size (pixels we want to jump over)
    cv::Mat_<float> disparity_copy = disparity_image;


    dst = cv::Mat(disparity_image.rows, disparity_image.cols, disparity_image.type(), cv::Scalar(dNaN));

    // for rows
    for (int row = 0; row < disparity_image.rows; row = row + 1) {

        // for cols
        for (int col = 0; col < disparity_image.cols; col = col + step_x_adapt) {

            // check if value is valid
            if ((!(std::isnan(disparity_image.at<float>(row,col)))) && (!(std::isnan(disparity_copy.at<float>(row,col))))) {

                // Get Hyp width from Table (shift sub-decimals away: *lut_adress_factor)
                hyp_ = LUT_[((int)disparity_image.at<float>(row,col)) * lut_adress_factor_];

                // Check if proposal larger than minimum width
                if( hyp_.w_ > min_hyp_width_)
                {
                    // Step size in percentage
                    step_x_adapt = std::floor(float(hyp_.w_ * step_perc_));
                    step_y_adapt = std::floor(float(hyp_.h_ * step_perc_));

                    // Step size must be >= 1
                   if (step_x_adapt < 1) {
                        step_x_adapt = 1;
                    }

                    if (step_y_adapt < 1) {
                        step_y_adapt = 1;
                    }

                    // Remember step size in y-direction in copy image
                    else {
                        for (int z = 0; z < step_y_adapt; z++) {
                            if (row + z + 1 < disparity_copy.rows) {
                                disparity_copy.at<float>(row + z + 1,col) = nan_point;
                            }
                        }
                    }

                    // adjust x and y
                    hyp_.x_ = col - hyp_.w_/2;
                    hyp_.y_ = row - hyp_.h_/2;

                    // Check if hyp is insinde image and hyp_width is in range(min,max)
                    if (hyp_.x_ > 0 && hyp_.y_ > 0  && ((hyp_.x_ + hyp_.w_) < disparity_image.cols) && ((hyp_.y_ + hyp_.h_) < disparity_image.rows) && (hyp_.w_ > min_hyp_width_) && (hyp_.w_ < max_hyp_width_)) {
                        // Check disparity in Hyp (6 Points)
                        inspectHypothesisDepth(disparity_image(cv::Rect(hyp_.x_,hyp_.y_,hyp_.w_,hyp_.h_)), stddev, nan_cnt);
                        if (stddev < max_stddev_ && nan_cnt <= max_nans_) {
                            // TODO: can we compute a confidence with nan_cnt and stddev?
                            hyp_.id_++;
                            hyp_.dist_ = -1.* tx / (disparity_image.at<float>(row,col));
                            dst.at<float>(row,col) = disparity_image.at<float>(row,col);
                            hyps.push_back(hyp_);
                        }

                    } // end check inside image
                } //end if hyp width > min_hyp_w
            } // end if valid
        } // end for cols
    } // end for rows

}

/**
    toPythonList:               Converts std vector ty python list

    @param std::vector<T>       Vector of int, float, double, ....
    @return                     Python list

*/
template <class T>
boost::python::list toPythonList(std::vector<T> vector) {
    typename std::vector<T>::iterator iter;
    boost::python::list list;
    for (iter = vector.begin(); iter != vector.end(); ++iter) {
        list.append(*iter);
    }
    return list;
}

/**
    inspectHypothesisDepth      Verifies the disparity values inside a potential object proposal: We expect homogeneous disparity values if it is an object

    @param hyp                  Object proposal as cv rect
    @param stddev               standard deviation as return
    @param nan_count            Number of NaNs of six pixels

*/
void DisparitySlidingWindow::inspectHypothesisDepth(const cv::Mat &hyp, float &stddev, size_t &nan_count) {

    switch (this->homogeneity_verification_method_) {

        case HOMOGENEITY_VERIFICATION::TRAFFIC_LIGHT: {


            // reset output vars
            nan_count = 0;
            stddev = 0.;

            // select six points from this hyp (only save because we checked out of image before!):
            int c_x = hyp.cols/2;
            int c_y = hyp.rows/2;
            int w = hyp.cols/4;
            int h = hyp.rows/4;
            float selected_points[6] = {
                hyp.at<float>(c_y,c_x-w),
                hyp.at<float>(c_y,c_x+w),
                hyp.at<float>(c_y+h,c_x-w),
                hyp.at<float>(c_y+h,c_x+w),
                hyp.at<float>(c_y-h,c_x-w),
                hyp.at<float>(c_y-h,c_x+w)
            };

            float sum = 0;
            int cnt = 0;
            float mean;

            // first run: check nan/inf and calc sum
            for (size_t i = 0; i < 6; i++) {
                if (!(std::isnan(selected_points[i])) && std::isfinite(selected_points[i])) {
                    sum += selected_points[i];
                    ++cnt;
                } else {
                    ++nan_count;
                }
            }

            // second run: calculate the stddev
            mean = sum/(float)cnt;
            sum = 0;
            cnt = 0;
            for (size_t i = 0; i < 6; i++) {
                if (!(std::isnan(selected_points[i])) && std::isfinite(selected_points[i])) {
                    sum += std::pow(selected_points[i]-mean,2);
                    ++cnt;
                }
            }
            stddev = std::sqrt(sum/(float)cnt);
            break;
        }

        // pedestrians bounding boxes contain background disparities at boundaries: only use a small region in the middle for homogeneity calculation
        case HOMOGENEITY_VERIFICATION::PEDESTRIAN: {


            int center_x = hyp.cols/2 - hyp.cols /6;
            int center_y = hyp.rows/2 - hyp.rows/6;
            int width = hyp.cols/3;
            int height = hyp.rows/3;


            cv::Mat hyp_crop = hyp(cv::Rect(center_x, center_y, width, height));


            // reset output vars
            nan_count = 0;
            stddev = 0.;

            // select six points from this hyp (only save because we checked out of image before!):
            int c_x = hyp_crop.cols/2;
            int c_y = hyp_crop.rows/2;
            int w = hyp_crop.cols/4;
            int h = hyp_crop.rows/4;
            float selected_points[6] = {
                hyp_crop.at<float>(c_y,c_x-w),
                hyp_crop.at<float>(c_y,c_x+w),
                hyp_crop.at<float>(c_y+h,c_x-w),
                hyp_crop.at<float>(c_y+h,c_x+w),
                hyp_crop.at<float>(c_y-h,c_x-w),
                hyp_crop.at<float>(c_y-h,c_x+w)
            };

            float sum = 0;
            int cnt = 0;
            float mean;

            // first run: check nan/inf and calc sum
            for (size_t i = 0; i < 6; i++) {
                if (!(std::isnan(selected_points[i])) && std::isfinite(selected_points[i])) {
                    sum += selected_points[i];
                    ++cnt;
                } else {
                    ++nan_count;
                }
            }

            // second run: calculate the stddev
            mean = sum/(float)cnt;
            sum = 0;
            cnt = 0;
            for (size_t i = 0; i < 6; i++) {
                if (!(std::isnan(selected_points[i])) && std::isfinite(selected_points[i])) {
                    sum += std::pow(selected_points[i]-mean,2);
                    ++cnt;
                }
            }
            stddev = std::sqrt(sum/(float)cnt);
                break;
            }

        default: {
            stddev = 0;
            break;
        }
    }

}



/**
    setXXX      Member setter, self-descriptive

*/
void DisparitySlidingWindow::setHypClassId(const int &id) {
    hyp_.classId_ = id;
}

void DisparitySlidingWindow::setHypCounter(const size_t &cnt) {
    hyp_.id_ = cnt;
}

void DisparitySlidingWindow::setMaxNans(const size_t &cnt) {
    max_nans_ = cnt;
}

void DisparitySlidingWindow::setMaxStddev(const float &val) {
    max_stddev_ = val;
}

void DisparitySlidingWindow::setHypAspect(const float &aspect) {
    hyp_aspect_ = aspect;
}

void DisparitySlidingWindow::setMinHypWidth(const int &min_w) {
    min_hyp_width_ = min_w;
}

void DisparitySlidingWindow::setMaxHypWidth(const int &max_w) {
    max_hyp_width_ = max_w;
}

void DisparitySlidingWindow::setObjWidth(const float &val) {
    obj_width_ = val;
}

void DisparitySlidingWindow::setObjHeight(const float &val) {
    obj_height_ = val;
}

void DisparitySlidingWindow::setStepPercentage(const float &val) {
    step_perc_ = val;
}
