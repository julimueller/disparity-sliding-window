#include <boost/python.hpp>
#include "disparity_sliding_window.h"


BOOST_PYTHON_MODULE(dsw_python)
{
    boost::python::class_<SmartSlidingWindow>("SmartSlidingWindow", boost::python::init<float, float, float, int, int, int, size_t, float, float>())
            .def("initLookUpTable", &SmartSlidingWindow::initLookUpTable)
            .def("generate", &SmartSlidingWindow::generate);
}