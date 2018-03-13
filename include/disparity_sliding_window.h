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
#include "pyboost_cvconverter.hpp"

class Rect {

   public:

    Rect():x(0), y(0), w(0), h(0)
    {}

    int x;
    int y;
    int w;
    int h;
    int dist;
    std::string classId;
    float confidence;
    int id;

};

class DisparitySlidingWindow {

public:
    DisparitySlidingWindow(const float &obj_width, const float &obj_height, const float &hyp_aspect, const int &min_hyp_width, const int &max_hyp_width, const int &hyp_class_id, const size_t &max_nans, const float &max_stddev, const float &stepperc, const int &class_type);
    virtual ~DisparitySlidingWindow();

    bool initLookUpTable(const float &tx, const cv::Mat &camera_matrix, const cv::Mat &distortion_matrix, const float &min_disp, const float &max_disp, const float &disp_step);
    void generate(const cv::Mat &disparity_image, cv::Mat &dst, std::vector<Rect> &hyps, const float &tx);
    boost::python::list generate_py(const cv::Mat &disparity_image, const float &tx);
    boost::python::object generate_py_via_mat(const cv::Mat &disparity_image, const float &tx);
    void sayHello();

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

    float obj_width;
    float obj_height;
    float hyp_aspect;
    int min_hyp_width;
    int max_hyp_width;
    int hyp_class_id;
    int class_type;

    size_t max_nans;
    float max_stddev;
    float step_perc;

    Rect hyp;

    int lut_adress_factor;
    std::vector<Rect> LUT;

    int homogeneity_verification_method;


};

#endif // _DISPARITY_SLIDING_WINDOW_H_
