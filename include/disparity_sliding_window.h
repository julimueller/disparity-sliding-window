/**
    disparity_sliding_window.h
    Purpose: Declaration of classes and methods needed for calculating object proposals from disparity images

    @author Julian Mueller, University of Ulm
    @version 1.1 02/03/18
*/

#ifndef _DISPARITY_SLIDING_WINDOW_H_
#define _DISPARITY_SLIDING_WINDOW_H_

#include <vector>
#include <opencv2/core/core.hpp>
#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>
#include "pyboost_cvconverter.hpp"

class Rect {

   public:

    Rect():x_(0), y_(0), w_(0), h_(0)
    {}

    Rect(const Rect &rhs):x_(rhs.x_), y_(rhs.y_), w_(rhs.w_), h_(rhs.h_)
    {}

    int x_;
    int y_;
    double w_;
    double h_;
    int dist_;
    std::string classId_;
    float confidence_;
    int id_;

};

class DisparitySlidingWindow {

public:

    DisparitySlidingWindow();
    DisparitySlidingWindow(const float &obj_width, const float &obj_height, const float &hyp_aspect, const int &min_hyp_width, const int &max_hyp_width, const int &hyp_class_id, const size_t &max_nans, const float &max_stddev, const float &stepperc, const int &class_type);
    virtual ~DisparitySlidingWindow();

    bool initLookUpTable(const float &tx, const cv::Mat &camera_matrix, const cv::Mat &distortion_matrix, const float &min_disp, const float &max_disp, const float &disp_step);
    void generate(const cv::Mat &disparity_image, cv::Mat &dst, std::vector<Rect> &hyps, const float &tx);
    boost::python::object randomHypothesisGenerator(int size, bool transferOwnership);

    void setHypCounter(const size_t &cnt);
    void setMaxNans(const size_t &cnt);
    void setMaxStddev(const float &val);
    void setHypAspect(const float &aspect);
    void setMinHypWidth(const int &min_w);
    void setMaxHypWidth(const int &max_w);
    void setHypClassId(const int &id);
    void setObjWidth(const float &val);
    void setObjHeight(const float &val);
    void setStepPercentage(const float &val);

    enum HOMOGENEITY_VERIFICATION {
        TRAFFIC_LIGHT=0,
        PEDESTRIAN=1
    };

private:

    void inspectHypothesisDepth(const cv::Mat &hyp, float &stddev, size_t &nan_count);

    float obj_width_;
    float obj_height_;
    float hyp_aspect_;
    int min_hyp_width_;
    int max_hyp_width_;
    int hyp_class_id_;

    size_t max_nans_;
    float max_stddev_;
    float step_perc_;
    int class_type_;

    Rect hyp_;

    int lut_adress_factor_;
    std::vector<Rect> LUT_;

    int homogeneity_verification_method_;


};

#endif // _DISPARITY_SLIDING_WINDOW_H_
