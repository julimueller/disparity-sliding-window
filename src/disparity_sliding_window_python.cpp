/**
    disparity_sliding_window_python.cpp
    Purpose: Python module to use Disparity Sliding Window in Python

    @author Julian Mueller, University of Ulm
    @version 1.1 02/03/18
*/


#include <boost/python.hpp>
#include "disparity_sliding_window.h"
#include "pyboost_cvconverter.hpp"


BOOST_PYTHON_MODULE(dsw_python)
{
    //initialize converters
    Py_Initialize();
    import_array();
    boost::python::to_python_converter<cv::Mat, pbcvt::matToNDArrayBoostConverter>();
    pbcvt::matFromNDArrayBoostConverter();

    boost::python::class_<DisparitySlidingWindow>("DisparitySlidingWindow", boost::python::init<float, float, float, int, int, int, size_t, float, float, int>())
            .def("initLookUpTable", &DisparitySlidingWindow::initLookUpTable)
            .def("generate_py", &DisparitySlidingWindow::generate_py);
}
