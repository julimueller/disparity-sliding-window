/**
    disparity_sliding_window_python.cpp
    Purpose: Python module to use Disparity Sliding Window in Python

    @author Julian Mueller, University of Ulm
    @version 1.1 02/03/18
*/


#include <boost/python.hpp>
#include "disparity_sliding_window.h"
#include "pyboost_cvconverter.hpp"
#include <boost/python/numeric.hpp>


BOOST_PYTHON_MODULE(dsw_python)
{
    //initialize converters
    Py_Initialize();
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
    import_array();
    boost::python::to_python_converter<cv::Mat, pbcvt::matToNDArrayBoostConverter>();
    pbcvt::matFromNDArrayBoostConverter();

    boost::python::class_<DisparitySlidingWindow>("DisparitySlidingWindow", boost::python::init<float, float, float, int, int, int, size_t, float, float, int>())
            .def("initLookUpTable", &DisparitySlidingWindow::initLookUpTable)
            .def("generate_py", &DisparitySlidingWindow::generate_py)
            .def("randomHypothesisGenerator", &DisparitySlidingWindow::randomHypothesisGenerator);

    boost::python::class_<Rect>("Rect", boost::python::init<>())
            .def_readwrite("x", &Rect::x)
            .def_readwrite("y", &Rect::y)
            .def_readwrite("w", &Rect::w)
            .def_readwrite("h", &Rect::h)
            .def_readwrite("dist", &Rect::dist)
            .def_readwrite("classId", &Rect::classId)
            .def_readwrite("confidence", &Rect::confidence)
            .def_readwrite("id", &Rect::id);


}
