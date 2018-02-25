#ifndef _SMART_SLIDING_WINDOW_H_
#define _SMART_SLIDING_WINDOW_H_

#include <vector>
#include <opencv2/core/core.hpp>

class Rect {

   public:

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
    DisparitySlidingWindow(const float &obj_width, const float &obj_height, const float &hyp_aspect, const int &min_hyp_width, const int &max_hyp_width, const int &hyp_class_id, const size_t &max_nans, const float &max_stddev, const float &stepperc);
    virtual ~DisparitySlidingWindow();

    bool initLookUpTable(const double &tx, const cv::Matx33d &camera_matrix, const cv::Mat &distortion_matrix, const float min_disp, const float &max_disp, const float &disp_step);
    void generate(const cv::Mat_<float> &disparity_image, cv::Mat &dst, std::vector<Rect> &hyps, const double &tx);

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

private:

    void inspectHypothesisDepth(const cv::Mat &hyp, float &stddev, size_t &nan_count);

    float obj_width;
    float obj_height;
    float hyp_aspect;
    int min_hyp_width;
    int max_hyp_width;
    int hyp_class_id;

    size_t max_nans;
    float max_stddev;
    float step_perc;

    Rect hyp;

    int lut_adress_factor;
    std::vector<Rect> LUT;

};

#endif // _SMART_SLIDING_WINDOW_H_
