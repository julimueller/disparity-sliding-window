#include <boost/python.hpp>
#include "disparity_sliding_window.h"


class PythonConversions {
public:
    PythonConversions();

    boost::python::object toPythonList(std::vector<Rect> vector);
    boost::python::object toPythonTuple(std::vector<Rect> vector);
    boost::python::object toPythonRectList(std::vector<Rect> vector);
    boost::python::object toPythonRectOwnership(std::vector<Rect *> &vector);
    boost::python::object rectToPythonNPArray(std::vector<Rect> vector);
    boost::python::object matToPythonNPArray(const cv::Mat &mat);
    boost::python::object rectToMat(const std::vector<Rect> &vector);
    boost::python::object py_rectToMat(boost::python::list& ns);
    std::vector<Rect> extractRects(boost::python::list &ns);

};
