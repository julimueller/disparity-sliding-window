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
    //import_array();
    boost::python::to_python_converter<cv::Mat, pbcvt::matToNDArrayBoostConverter>();
    pbcvt::matFromNDArrayBoostConverter();

    boost::python::class_<DisparitySlidingWindow>("DisparitySlidingWindow", boost::python::init<float, float, float, int, int, int, size_t, float, float, int>())
            .def("initLookUpTable", &DisparitySlidingWindow::initLookUpTable)
            .def("generate_py", &DisparitySlidingWindow::generate_py)
            .def("randomHypothesisGenerator", &DisparitySlidingWindow::randomHypothesisGenerator);

    boost::python::class_<Rect>("Rect", boost::python::init<>())
            .add_property("x", boost::python::make_getter(&Rect::x, boost::python::return_value_policy<boost::python::return_by_value>()),
                                 boost::python::make_setter(&Rect::x, boost::python::return_value_policy<boost::python::return_by_value>()))
            .add_property("y", boost::python::make_getter(&Rect::y, boost::python::return_value_policy<boost::python::return_by_value>()),
                                 boost::python::make_setter(&Rect::y, boost::python::return_value_policy<boost::python::return_by_value>()))
            .add_property("w", boost::python::make_getter(&Rect::w, boost::python::return_value_policy<boost::python::return_by_value>()),
                                 boost::python::make_setter(&Rect::w, boost::python::return_value_policy<boost::python::return_by_value>()))
            .add_property("h", boost::python::make_getter(&Rect::h, boost::python::return_value_policy<boost::python::return_by_value>()),
                                 boost::python::make_setter(&Rect::h, boost::python::return_value_policy<boost::python::return_by_value>()))
            .add_property("dist", boost::python::make_getter(&Rect::dist, boost::python::return_value_policy<boost::python::return_by_value>()),
                                 boost::python::make_setter(&Rect::dist, boost::python::return_value_policy<boost::python::return_by_value>()))
            .add_property("classId", boost::python::make_getter(&Rect::classId, boost::python::return_value_policy<boost::python::return_by_value>()),
                                 boost::python::make_setter(&Rect::classId, boost::python::return_value_policy<boost::python::return_by_value>()))
            .add_property("confidence", boost::python::make_getter(&Rect::confidence, boost::python::return_value_policy<boost::python::return_by_value>()),
                                 boost::python::make_setter(&Rect::confidence, boost::python::return_value_policy<boost::python::return_by_value>()))
            .add_property("id", boost::python::make_getter(&Rect::id, boost::python::return_value_policy<boost::python::return_by_value>()),
                                 boost::python::make_setter(&Rect::id, boost::python::return_value_policy<boost::python::return_by_value>()));

    boost::python::def("py_rectToMat", &py_rectToMat);


}
