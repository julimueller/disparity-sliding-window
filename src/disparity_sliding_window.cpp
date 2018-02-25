#include <disparity_sliding_window.h>
#include <cmath>
#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>

#include <fstream>

DisparitySlidingWindow::DisparitySlidingWindow(const float &obj_width, const float &obj_height, const float &hyp_aspect, const int &min_hyp_width, const int &max_hyp_width, const int &hyp_class_id, const size_t &max_nans, const float &max_stddev, const float &stepperc) :
    obj_width(obj_width),
    obj_height(obj_height),
    hyp_aspect(hyp_aspect),
    min_hyp_width(min_hyp_width),
    max_hyp_width(max_hyp_width),
    hyp_class_id(hyp_class_id),
    max_nans(max_nans),
    max_stddev(max_stddev),
    step_perc(stepperc)
{
    hyp.id = 0;
    lut_adress_factor = 0;
}

DisparitySlidingWindow::~DisparitySlidingWindow() {

}

bool DisparitySlidingWindow::initLookUpTable(const double &tx, const cv::Matx33d &camera_matrix, const cv::Mat &distortion_matrix, const float min_disp, const float &max_disp, const float &disp_step) {

    // check if inverse disp-step is a multiple of 2
    if (((int)(disp_step/1.))%2 != 0) {
        std::cout << "ERROR:\t Inverse disparity step is not a multiple of two! This lib expects fixed-point floating point disparity images with a sub-decimal precision of n bits." << std::endl;
        return false;
    }

    this->LUT.clear();
    lut_adress_factor = 1./((float)disp_step);

    std::cout << "INFO:\tLUT adress factor: " << lut_adress_factor << std::endl;
    std::cout << "INFO:\tCalculating a hypotheses LUT with " << ((max_disp-min_disp)/disp_step)+1 << " entries." << std::endl;

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

//    std::ofstream of;
//    of.open("/home/afregin/schrott/LUT.txt");

    Rect hyp;
    for (float disparity = min_disp; disparity <= max_disp; disparity += disp_step) {

        // calculate depth in meters
        tvec.at<float>(2,0) = -1.* tx / (disparity);

        // project
        cv::projectPoints(object_points, rvec, tvec, camera_matrix, distortion_matrix, image_points);
        // TODO: this assumes that the object is at the image center, it would be a better to use the matching ray!
        //       but this would mean w'd have a sizeof(float) * (cols/step_x) * (rows/step_y) * (disp_range/disp_step) LUT!

        hyp.w = (image_points.at(1).x - image_points.at(0).x);    // TODO: should we std::round here?
        hyp.h = hyp_aspect * hyp.w;
        hyp.x = 0;
        hyp.y = 0;
        hyp.classId = std::to_string(hyp_class_id);
        hyp.id = LUT.size();
        hyp.confidence = 1.0;
        LUT.push_back(hyp);

//        of << hyp.w << "\n";

    }

    return true;
}

void DisparitySlidingWindow::generate(const cv::Mat_<float> &disparity_image, cv::Mat &dst, std::vector<Rect> &hyps, const double &tx) {

    // check if we're ready to generate
    if (LUT.size() == 0 || lut_adress_factor == 0) {
        std::cout << "ERROR:\tYou must call 'calculateLookUpTable' before calling 'generate'!" << std::endl;
        return;
    }

    int step_x_adapt = 1;
    int step_y_adapt = 0;
    float stddev;
    size_t nan_cnt;
    float nan_point = std::numeric_limits<float>::quiet_NaN ();

    cv::Mat_<float> disparity_copy = disparity_image;

    double dNaN = std::numeric_limits<double>::quiet_NaN();

    dst = cv::Mat(disparity_image.rows, disparity_image.cols, disparity_image.type(), cv::Scalar(dNaN));



    // for rows
    for (int row = 0; row < disparity_image.rows; row = row + 1) {

        // for cols
        for (int col = 0; col < disparity_image.cols; col = col + step_x_adapt) {

            // check if value is valid
            if ((!(std::isnan(disparity_image.at<float>(row,col)))) && (!(std::isnan(disparity_copy.at<float>(row,col))))) {

                // Get Hyp width from Table (shift sub-decimals away: *lut_adress_factor)

                hyp = LUT[((int)disparity_image.at<float>(row,col)) * lut_adress_factor];


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

void DisparitySlidingWindow::inspectHypothesisDepth(const cv::Mat &hyp, float &stddev, size_t &nan_count) {

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
//    std::cout << "sum: " << sum << ", mean: " << mean << ", stddev: " << stddev << ", nans: " << nan_count << std::endl;
}


/////////////////////////////////////////////////////////
// MEMBER SETTER
/////////////////////////////////////////////////////////
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
