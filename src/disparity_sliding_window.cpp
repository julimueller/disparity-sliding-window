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


/**
    Constructor of DisparitySlidingWindow

    @param obj_width        Objects real world width in meters, e.g. 0.6 meters for pedestrians
    @param obj_height       Objects real world height in meters, e.g. 1.73 meters for pedestrians
    @param hyp_aspect       Aspect ratio (pixels) of the object, e.g. 2-3 for pedestrians
    @param min_hyp_width    Minnimum proposal width in pixels
    @param max_hyp_width    Maximum proposal width in pixels
    @param hyp_class_id     Class identity, you can actually ignore this
    @param max_nans         Maximum number of allowed NaNs before proposal is rejected, actually we use 6 pixels for homogeneity thresholding, if you dont want to use this feature set it to 6
    @param max_stddev       Maximum allowed standard deviation (homogeneity verification), good value for pedestrians: 1.0
    @param step_perc        Adaptive step size (percentage of object width, see paper), we propose values between 0.3 and 0.4 for accurate detection results

*/
DisparitySlidingWindow::DisparitySlidingWindow(const float &obj_width, const float &obj_height, const float &hyp_aspect, const int &min_hyp_width, const int &max_hyp_width, const int &hyp_class_id, const size_t &max_nans, const float &max_stddev, const float &step_perc, const int &class_type) :
    obj_width(obj_width),
    obj_height(obj_height),
    hyp_aspect(hyp_aspect),
    min_hyp_width(min_hyp_width),
    max_hyp_width(max_hyp_width),
    hyp_class_id(hyp_class_id),
    max_nans(max_nans),
    max_stddev(max_stddev),
    step_perc(step_perc),
    class_type(class_type)

{
    hyp.id = 0;
    lut_adress_factor = 0;
    this->homogeneity_verification_method = class_type;
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

    this->LUT.clear();
    lut_adress_factor = 1./((float)disp_step);

    std::cout << "INFO:\tLUT adress factor: " << lut_adress_factor << std::endl;
    std::cout << "INFO:\tCalculating a hypotheses LUT with " << ((max_disp-min_disp)/disp_step)+1 << " entries." << std::endl;
    //std::cout <<"camera matrix: " << camera_matrix <<"\n";
    //std::cout <<"distortion_matrix: " << distortion_matrix <<"\n";

    // prepare real world object points
    std::vector<cv::Point3f> object_points;
    object_points.resize(4);
    object_points[0] = cv::Point3f(0.,0.,0.);
    object_points[1] = cv::Point3f(obj_width,0.,0.);
    object_points[2] = cv::Point3f(obj_width,obj_height,0.);
    object_points[3] = cv::Point3f(0.,obj_height,0.);

    // prepare other data for cv::projectPoints
    std::vector<cv::Point2f> image_points;

    cv::Mat_<float> tvec = (cv::Mat_<float>(3,1) << 0., 0., 0.);
    cv::Mat_<float> rvec = (cv::Mat_<float>(3,1) << 0., 0., 0.);


    Rect hyp;
    for (float disparity = min_disp; disparity <= max_disp; disparity += disp_step) {

        // calculate depth in meters
        tvec.at<float>(2,0) = -1.* tx / (disparity);
        //std::cout << tvec.at<float>(2,0) << std::endl;

        // project
        cv::projectPoints(object_points, rvec, tvec, camera_matrix, distortion_matrix, image_points);
        // TODO: this assumes that the object is at the image center, it would be a better to use the matching ray!
        //       but this would mean w'd have a sizeof(float) * (cols/step_x) * (rows/step_y) * (disp_range/disp_step) LUT!
        hyp.w = (image_points.at(1).x - image_points.at(0).x);    // TODO: should we std::round here?

        //std::cout << "w: " << image_points.at(1).x << " " << image_points.at(0).x << " " << image_points.at(1).x - image_points.at(0).x << std::endl;
        //std::cout << "w: " << hyp.w<< std::endl;

        hyp.h = hyp_aspect * hyp.w;
        hyp.x = 0;
        hyp.y = 0;
        hyp.classId = std::to_string(hyp_class_id);
        hyp.id = LUT.size();
        hyp.confidence = 1.0;
        LUT.push_back(hyp);

    }


    /*for(int i = 0; i <LUT.size(); i++){
        std::cout << LUT[i].x << " "<< LUT[i].y << " "<< LUT[i].w << " "<< LUT[i].h << "\n";
    }*/

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
    if (LUT.size() == 0 || lut_adress_factor == 0) {
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
                hyp = LUT[((int)disparity_image.at<float>(row,col)) * lut_adress_factor];

                // Check if proposal larger than minimum width
                if( hyp.w > min_hyp_width)
                {

                    // Step size in percentage
                    step_x_adapt = std::floor(float(hyp.w * step_perc));
                    step_y_adapt = std::floor(float(hyp.h * step_perc));

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
                    hyp.x = col - hyp.w/2;
                    hyp.y = row - hyp.h/2;

                    // Check if hyp is insinde image and hyp_width is in range(min,max)
                    if (hyp.x > 0 && hyp.y > 0  && ((hyp.x + hyp.w) < disparity_image.cols) && ((hyp.y + hyp.h) < disparity_image.rows) && (hyp.w > min_hyp_width) && (hyp.w < max_hyp_width)) {

                        // Check disparity in Hyp (6 Points)
                        inspectHypothesisDepth(disparity_image(cv::Rect(hyp.x,hyp.y,hyp.w,hyp.h)), stddev, nan_cnt);
                        //stddev=0.;

                        if (stddev < max_stddev && nan_cnt <= max_nans) {
                            // TODO: can we compute a confidence with nan_cnt and stddev?
                            hyp.id++;
                            hyp.dist = -1.* tx / (disparity_image.at<float>(row,col));
                            dst.at<float>(row,col) = disparity_image.at<float>(row,col);
							hyps.push_back(hyp);
                        }

                    } // end check inside image
                } //end if hyp width > min_hyp_w
            } // end if valid
        } // end for cols
    } // end for rows

}


/**
    generate_py:                Calculates object proposals from the disparity input and return python list inside a python object

    @param disparity_image      Disparity image (CV_32FC1)
    @param tx                   Tx = -fx' * B, where fx is the focal length in x and B is the baseline of the stereo camera. See also here: http://docs.ros.org/jade/api/sensor_msgs/html/msg/CameraInfo.html
    @return                     Python list as list = [x1,y1,w1,h1,x2,y2,w2,h2,...]

*/
boost::python::object DisparitySlidingWindow::generate_py(const cv::Mat &disparity_image, const float &tx) {

    //std::cout <<"disparity image: "<< disparity_image << std::endl;
    // check if we're ready to generate
    if (LUT.size() == 0 || lut_adress_factor == 0) {
        std::cout << "ERROR:\tYou must call 'calculateLookUpTable' before calling 'generate'!" << std::endl;
        return boost::python::list();
    }

    // check if disparity image is of type float
    if (disparity_image.type() != CV_32FC1) {
        std::cout << "ERROR:\tWe expect a float image as an input!" << std::endl;
        return boost::python::list();
    }

    std::vector<Rect> hyps;

    // adaptive step sizes
    int step_x_adapt = 1;
    int step_y_adapt = 0;

    float stddev;
    size_t nan_cnt;
    float nan_point = std::numeric_limits<float>::quiet_NaN ();

    // we need this to save the step size (pixels we want to jump over)
    cv::Mat_<float> disparity_copy = disparity_image;

    // for rows
    for (int row = 0; row < disparity_image.rows; row = row + 1) {

        // for cols
        for (int col = 0; col < disparity_image.cols; col = col + step_x_adapt) {
            //std::cout << "loop" << std::endl;
            // check if value is valid
            if ((!(std::isnan(disparity_image.at<float>(row,col)))) && (!(std::isnan(disparity_copy.at<float>(row,col))))) {

                // Get Hyp width from Table (shift sub-decimals away: *lut_adress_factor)
                hyp = LUT[((int)disparity_image.at<float>(row,col)) * lut_adress_factor];
                //std::cout << hyp.w << std::endl;
                // Check if proposal larger than minimum width
               // std::cout << "hyp w: " << hyp.w << ", min w : " << min_hyp_width << std::endl;
                if( hyp.w > min_hyp_width) {

                    // Step size in percentage
                    step_x_adapt = std::floor(float(hyp.w * step_perc));
                    step_y_adapt = std::floor(float(hyp.h * step_perc));

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
                    hyp.x = col - hyp.w/2;
                    hyp.y = row - hyp.h/2;

                    //std::cout << "hyp x: " << hyp.x << ", hyp y: "<< hyp.y << ", hyp w: " << hyp.w << ", hyp h: "<< hyp.h << std::endl;

                    // Check if hyp is insinde image and hyp_width is in range(min,max)
                    if (hyp.x > 0 && hyp.y > 0  && ((hyp.x + hyp.w) < disparity_image.cols) && ((hyp.y + hyp.h) < disparity_image.rows) && (hyp.w > min_hyp_width) && (hyp.w < max_hyp_width)) {

                        // Check disparity in Hyp (6 Points)
                        inspectHypothesisDepth(disparity_image(cv::Rect(hyp.x,hyp.y,hyp.w,hyp.h)), stddev, nan_cnt);
                        //stddev=0.;
                        //std::cout <<"stddev: " << stddev <<", max stddev: " << max_stddev <<"\n";
                        //std::cout <<"nan_cnt: " << nan_cnt <<", max nan_cnt: " << max_nans <<"\n";
                        if (stddev < max_stddev && nan_cnt <= max_nans) {
                            // TODO: can we compute a confidence with nan_cnt and stddev?
                            hyp.id++;
                            hyp.dist = -1.* tx / (disparity_image.at<float>(row,col));
                            hyps.push_back(hyp);
                        }

                    } // end check inside image
                } //end if hyp width > min_hyp_w
            } // end if valid
        } // end for cols
    } // end for rows

    return toPythonTuple(hyps);

}



/**
    randomHypothesisGenerator:      Creates random rectangles which all the variables of rectangle object are in the range [0,100] and it returns the rectangles inside a python object
                                    depending on the chosen method, the options are:
                                        i. It can return as the list of integers, which the (x,y) coordinates and width, height of rectangles are consecutively added to an integer vector
                                        ii. It can return as the list of python tuples and each entry in the list is tuple of (x,y,w,h)
                                        iii. It can return as the list of rectangles, Rect class is also included in Python side, so the conversion from C++ to Python will be automated  while appending each entry to list
                                        iv. It can return as Numpy array where each row represents a rectangle and in each row (x,y,w,h) is kept.

                                    @param size: It is the number of rectangles to be generated
                                    @param transferOwnership: this one is put for debugging reasons, when the object ownership is translated, the objects should be represented
                                    by pointers, in other cases, the objects can be restored, created without knowing their pointers, their addresses in the memory, therefore for debugging
                                    of transfer of ownership, this option is put.

*/
boost::python::object DisparitySlidingWindow::randomHypothesisGenerator(int size, bool transferOwnership){
    if(transferOwnership){
        std::vector<Rect*> hyps_test;
        hyps_test.resize(size);

        for(int i = 0; i<hyps_test.size(); i++){
            int x = rand()%100;
            int y = rand()%100;
            int w = rand()%100;
            int h = rand()%100;
            hyps_test[i] = new Rect;
            hyps_test[i]->x = x;
            hyps_test[i]->y = y;
            hyps_test[i]->w = w;
            hyps_test[i]->h = h;
        }
        return toPythonRectOwnership(hyps_test);
    }
    else{
        std::vector<Rect> hyps_test;
        hyps_test.resize(size);

        for(int i = 0; i<hyps_test.size(); i++){
            int x = rand()%100;
            int y = rand()%100;
            int w = rand()%100;
            int h = rand()%100;
            hyps_test[i].x = x;
            hyps_test[i].y = y;
            hyps_test[i].w = w;
            hyps_test[i].h = h;
        }
        return rectToMat(hyps_test);
    }
}


/**
    inspectHypothesisDepth      Verifies the disparity values inside a potential object proposal: We expect homogeneous disparity values if it is an object

    @param hyp                  Object proposal as cv rect
    @param stddev               standard deviation as return
    @param nan_count            Number of NaNs of six pixels

*/
void DisparitySlidingWindow::inspectHypothesisDepth(const cv::Mat &hyp, float &stddev, size_t &nan_count) {

    switch (this->homogeneity_verification_method) {

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
    hyp.classId = id;
}

void DisparitySlidingWindow::setHypCounter(const size_t &cnt) {
    hyp.id = cnt;
}

void DisparitySlidingWindow::setMaxNans(const size_t &cnt) {
    max_nans = cnt;
}

void DisparitySlidingWindow::setMaxStddev(const float &val) {
    max_stddev = val;
}

void DisparitySlidingWindow::setHypAspect(const float &aspect) {
    hyp_aspect = aspect;
}

void DisparitySlidingWindow::setMinHypWidth(const int &min_w) {
    min_hyp_width = min_w;
}

void DisparitySlidingWindow::setMaxHypWidth(const int &max_w) {
    max_hyp_width = max_w;
}

void DisparitySlidingWindow::setObjWidth(const float &val) {
    obj_width = val;
}

void DisparitySlidingWindow::setObjHeight(const float &val) {
    obj_height= val;
}

void DisparitySlidingWindow::setStepPercentage(const float &val) {
    step_perc= val;
}



/************************************************************/
/*  All the functions related to conversions between       */
/*  C++ / Python are located in below:   */
/************************************************************/



/**
    toPythonList:               Converts vector of rectangles to python list

    @param std::vector<T>       Vector of Rectangles, ....
    @return                     Python list

*/
boost::python::object toPythonList(std::vector<Rect> vector) {
    typename std::vector<Rect>::iterator iter;
    boost::python::list list;
    for (iter = vector.begin(); iter != vector.end(); ++iter) {
        list.append(iter->x);
        list.append(iter->y);
        list.append(iter->w);
        list.append(iter->h);
    }

    return boost::python::object(list);
}


/**
    toPythonTuple: Converts the vector of rectangles to a list of tuples in python which each tuple keeps (x,y,w,h)

    @param std::vector<Rect>: Vector of rectangles
    @return: Python tuple list

*/
boost::python::object toPythonTuple(std::vector<Rect> vector) {
    typename std::vector<Rect>::iterator iter;
    boost::python::list tuple_list;
    for (iter = vector.begin(); iter != vector.end(); ++iter) {
        boost::python::tuple t = boost::python::make_tuple(iter->x, iter->y, iter->w, iter->h);
        tuple_list.append(t);
    }
    return boost::python::object(tuple_list);
}

/**
    toPythonRectList:   Takes a std library vector of rectangles as input and returns Python list of Rect class(Rect class is already exported to Python,
                            for more information see disparity_sliding_window_python.cpp )
    @param std::vector<Rect>:   Vector of rectangles
*/

boost::python::object toPythonRectList(std::vector<Rect> vector) {
    typename std::vector<Rect>::iterator iter;
    boost::python::list list;
    for (iter = vector.begin(); iter != vector.end(); ++iter) {
        list.append(boost::python::object(*iter));
    }
    return boost::python::object(list);
}



/**
    toPythonRectOwnership: Takes a std vector of rectangles pointers as input and returns Python list of rectangles but classes are transferred from C++
                            and ownership is passed from C++ to Python so when lifetime of rectangles are finished on Python side, objects will be deleted
    @param std::vector<Rect*>: Vector of rectangles pointers
*/

boost::python::object toPythonRectOwnership(std::vector<Rect*> &vector) {
    namespace python = boost::python;
    typename python::manage_new_object::apply<Rect*>::type converter;
    boost::python::list list;
    for (int i = 0; i < vector.size(); ++i) {
        std::unique_ptr<Rect> ptr(vector[i]);
        python::handle<> handle(converter(*ptr));
        list.append(boost::python::object(handle));
        ptr.release();
    }
    return boost::python::object(list);
}

/**
    rectToPythonNPArray:    Takes a vector of rectangles as input, creates a C++ Mat object in such a way that each row in the matrix
                                will keep (x,y,w,h) information of 1 rectangle. The row number will be equal to size of vector and column number will be equal to 4.
                                Later on, this Mat object is converted to Python ND array and is exported to PYTHON
    @param std::vector<Rect>: Vector of rectangles
*/
boost::python::object rectToPythonNPArray(std::vector<Rect> vector){
    boost::python::list l;
    for(int i = 0; i < vector.size(); i++){
        boost::python::tuple t = boost::python::make_tuple(vector[i].x, vector[i].y, vector[i].w, vector[i].h);
        l.append(t);
    }
    boost::python::numeric::array arr = boost::python::numeric::array(l);
    return boost::python::object(arr);
}


/**
    matToPythonNPArray:    Takes a cv::Mat object and converts it to Python ND array and is exported to PYTHON
    @param const cv::Mat &mat: Matrix to be converted to NP array
*/
boost::python::object matToPythonNPArray(const cv::Mat &mat){
    cv::Size s = mat.size();
    npy_intp dims[2] = {s.height, s.width};
    cv::Mat copy_mat;
    mat.convertTo(copy_mat, CV_32F);
    float * data = reinterpret_cast<float*>(copy_mat.ptr<float>(0));

    PyObject * pyObj = PyArray_SimpleNewFromData( 2, dims, NPY_FLOAT32, data);
    boost::python::handle<> handle( pyObj );
    boost::python::numeric::array arr( handle );
    return arr.copy();
}



/**
    rectToMat:      Takes a vector of Rectangles and converts the vector into cv::Mat in a way that
                    each row represents 1 rectangle ( 1 row = (x,y,w,h) of 1 rectangle)
                    @param std::vector<Rect>: Vector of rectangles

*/
boost::python::object rectToMat(const std::vector<Rect> &vector){
    cv::Mat mat = cv::Mat(vector.size(), 4, CV_32S, cv::Scalar(0));
    for(int i = 0; i < vector.size(); i++){
        mat.at<int>(i, 0) = vector[i].x;
        mat.at<int>(i, 1) = vector[i].y;
        mat.at<int>(i, 2) = vector[i].w;
        mat.at<int>(i, 3) = vector[i].h;
    }

    return matToPythonNPArray(mat);
}


/**

    extractRects: Take a python list and creates rectangles from entries in the list.
                  The entries in the list should be either type of boost::python::tuple or Rect.
                  The other type of entries will be skipped.
                  @param boost::python::list &ns: Python list object

*/

std::vector<Rect> extractRects(boost::python::list& ns){
    std::vector<Rect> vector;
    for (int i = 0; i < len(ns); ++i)
    {
        boost::python::extract<boost::python::object> objectExtractor(ns[i]);
        boost::python::object o=objectExtractor();
        std::string object_classname = boost::python::extract<std::string>(o.attr("__class__").attr("__name__"));

        Rect r;
        if(object_classname=="Rect")
            r = Rect(boost::python::extract<Rect>(ns[i]));
        else if(object_classname=="tuple"){
            boost::python::tuple t = boost::python::extract<boost::python::tuple>(o);
            r.x = boost::python::extract<int>(t[0]);
            r.y = boost::python::extract<int>(t[1]);
            r.w = boost::python::extract<int>(t[2]);
            r.h = boost::python::extract<int>(t[3]);
        }
        else
            std::cout<<"this is not either a tuple or a Rect. This is an object: "<<object_classname<<std::endl;
        vector.push_back(r);
    }
    return vector;
}



/**

        py_rectToMat: Takes a python list of objects, extracts the rectangles and returns
                      Python Numpy array which each row of array is (x,y,w,h) of a rectangle.
                      @param boost::python::list &ns: Python list object
*/
boost::python::object py_rectToMat(boost::python::list& ns)
{
    std::vector<Rect> vector = extractRects(ns);
    return rectToMat(vector);
}
