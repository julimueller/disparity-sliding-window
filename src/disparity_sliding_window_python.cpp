/**
    disparity_sliding_window_python.cpp
    Purpose: Python module to use Disparity Sliding Window in Python

    @author Julian Mueller, University of Ulm
    @version 1.1 02/03/18
*/


#include <boost/python.hpp>
#include "disparity_sliding_window.h"
#include "python_conversions.h"
#include "pyboost_cvconverter.hpp"
#include <boost/python/numeric.hpp>

class DisparitySlidingWindowWrapper {



public:
    DisparitySlidingWindowWrapper(const float &obj_width, const float &obj_height, const float &hyp_aspect, const int &min_hyp_width, const int &max_hyp_width, const int &hyp_class_id, const size_t &max_nans, const float &max_stddev, const float &step_perc, const int &class_type) {
        dsw_ =  DisparitySlidingWindow(obj_width, obj_height, hyp_aspect, min_hyp_width, max_hyp_width, hyp_class_id, max_nans, max_stddev, step_perc, class_type);
    }

    ~DisparitySlidingWindowWrapper() {
    }

    bool initLookUpTable(const float &tx, const cv::Mat &camera_matrix, const cv::Mat &distortion_matrix, const float &min_disp, const float &max_disp, const float &disp_step) {
        return dsw_.initLookUpTable(tx, camera_matrix, distortion_matrix, min_disp, max_disp, disp_step);
    }

    boost::python::object generate_py(const cv::Mat &disparity_image, const float &tx) {
        std::vector<Rect> hyps;
        cv::Mat dst;
        dsw_.generate(disparity_image, dst, hyps, tx);
        std::cout << hyps.size() << std::endl;
        return pyConv_.toPythonTuple(hyps);
    }


private:
    DisparitySlidingWindow dsw_;
    PythonConversions pyConv_;


};

BOOST_PYTHON_MODULE(dsw_python)
{
    //initialize converters
    Py_Initialize();
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
    import_array();
    boost::python::to_python_converter<cv::Mat, pbcvt::matToNDArrayBoostConverter>();
    pbcvt::matFromNDArrayBoostConverter();

    boost::python::class_<DisparitySlidingWindowWrapper>("DisparitySlidingWindowWrapper", boost::python::init<float, float, float, int, int, int, size_t, float, float, int>())
            .def("initLookUpTable", &DisparitySlidingWindowWrapper::initLookUpTable)
            .def("generate_py", &DisparitySlidingWindowWrapper::generate_py);

    boost::python::class_<Rect>("Rect", boost::python::init<>())
            .add_property("x", boost::python::make_getter(&Rect::x_, boost::python::return_value_policy<boost::python::return_by_value>()),
                                 boost::python::make_setter(&Rect::x_, boost::python::return_value_policy<boost::python::return_by_value>()))
            .add_property("y", boost::python::make_getter(&Rect::y_, boost::python::return_value_policy<boost::python::return_by_value>()),
                                 boost::python::make_setter(&Rect::y_, boost::python::return_value_policy<boost::python::return_by_value>()))
            .add_property("w", boost::python::make_getter(&Rect::w_, boost::python::return_value_policy<boost::python::return_by_value>()),
                                 boost::python::make_setter(&Rect::w_, boost::python::return_value_policy<boost::python::return_by_value>()))
            .add_property("h", boost::python::make_getter(&Rect::h_, boost::python::return_value_policy<boost::python::return_by_value>()),
                                 boost::python::make_setter(&Rect::h_, boost::python::return_value_policy<boost::python::return_by_value>()))
            .add_property("dist", boost::python::make_getter(&Rect::dist_, boost::python::return_value_policy<boost::python::return_by_value>()),
                                 boost::python::make_setter(&Rect::dist_, boost::python::return_value_policy<boost::python::return_by_value>()))
            .add_property("classId", boost::python::make_getter(&Rect::classId_, boost::python::return_value_policy<boost::python::return_by_value>()),
                                 boost::python::make_setter(&Rect::classId_, boost::python::return_value_policy<boost::python::return_by_value>()))
            .add_property("confidence", boost::python::make_getter(&Rect::confidence_, boost::python::return_value_policy<boost::python::return_by_value>()),
                                 boost::python::make_setter(&Rect::confidence_, boost::python::return_value_policy<boost::python::return_by_value>()))
            .add_property("id", boost::python::make_getter(&Rect::id_, boost::python::return_value_policy<boost::python::return_by_value>()),
                                 boost::python::make_setter(&Rect::id_, boost::python::return_value_policy<boost::python::return_by_value>()));

    boost::python::def("py_rectToMat", &PythonConversions::py_rectToMat);


}


