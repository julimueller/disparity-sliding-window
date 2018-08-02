#include "python_conversions.h"

/**
    Constructor of PythonConversions
*/
PythonConversions::PythonConversions() {

}

/**
    toPythonList:               Converts vector of rectangles to python list

    @param std::vector<T>       Vector of Rectangles, ....
    @return                     Python list

*/
boost::python::object PythonConversions::toPythonList(std::vector<Rect> vector) {
    typename std::vector<Rect>::iterator iter;
    boost::python::list list;
    for (iter = vector.begin(); iter != vector.end(); ++iter) {
        list.append(iter->x_);
        list.append(iter->y_);
        list.append(iter->w_);
        list.append(iter->h_);
    }
    return boost::python::object(list);
}


/**
    toPythonTuple: Converts the vector of rectangles to a list of tuples in python which each tuple keeps (x,y,w,h)

    @param std::vector<Rect>: Vector of rectangles
    @return: Python tuple list

*/
boost::python::object PythonConversions::toPythonTuple(std::vector<Rect> vector) {
    typename std::vector<Rect>::iterator iter;
    boost::python::list tuple_list;
    for (iter = vector.begin(); iter != vector.end(); ++iter) {
        boost::python::tuple t = boost::python::make_tuple(iter->x_, iter->y_, iter->w_, iter->h_);
        tuple_list.append(t);
    }
    return boost::python::object(tuple_list);
}

/**
    toPythonRectList:   Takes a std library vector of rectangles as input and returns Python list of Rect class(Rect class is already exported to Python,
                            for more information see disparity_sliding_window_python.cpp )
    @param std::vector<Rect>:   Vector of rectangles
*/

boost::python::object PythonConversions::toPythonRectList(std::vector<Rect> vector) {
    typename std::vector<Rect>::iterator iter;
    boost::python::list list;
    for (iter = vector.begin(); iter != vector.end(); ++iter) {
        list.append(boost::python::object(*iter));
    }
    return boost::python::object(list);
}



/**
    toPythonRectOwnership: Takes a std vector of rectangles pointers as input and returns Python list of rectangles but classes are transferred from C
                            and ownership is passed from C to Python so when lifetime of rectangles are finished on Python side, objects will be deleted
    @param std::vector<Rect*>: Vector of rectangles pointers
*/

boost::python::object PythonConversions::toPythonRectOwnership(std::vector<Rect*> &vector) {
    namespace python = boost::python;
    typename python::manage_new_object::apply<Rect*>::type converter;
    boost::python::list list;
    for (size_t i = 0; i < vector.size(); ++i) {
        std::unique_ptr<Rect> ptr(vector[i]);
        python::handle<> handle(converter(*ptr));
        list.append(boost::python::object(handle));
        ptr.release();
    }
    return boost::python::object(list);
}

/**
    rectToPythonNPArray:    Takes a vector of rectangles as input, creates a C Mat object in such a way that each row in the matrix
                                will keep (x,y,w,h) information of 1 rectangle. The row number will be equal to size of vector and column number will be equal to 4.
                                Later on, this Mat object is converted to Python ND array and is exported to PYTHON
    @param std::vector<Rect>: Vector of rectangles
*/
boost::python::object PythonConversions::rectToPythonNPArray(std::vector<Rect> vector){
    boost::python::list l;
    for(size_t i = 0; i < vector.size(); i++){
        boost::python::tuple t = boost::python::make_tuple(vector[i].x_, vector[i].y_, vector[i].w_, vector[i].h_);
        l.append(t);
    }
    boost::python::numeric::array arr = boost::python::numeric::array(l);
    return boost::python::object(arr);
}


/**
    matToPythonNPArray:    Takes a cv::Mat object and converts it to Python ND array and is exported to PYTHON
    @param const cv::Mat &mat: Matrix to be converted to NP array
*/
boost::python::object PythonConversions::matToPythonNPArray(const cv::Mat &mat){
    cv::Size s = mat.size();
    npy_intp dims[2] = {s.height, s.width};
    cv::Mat copy_mat;
    mat.convertTo(copy_mat, CV_32F);
    float * data = reinterpret_cast<float*>(copy_mat.ptr<float>(0));

    PyObject * pyObj = PyArray_SimpleNewFromData( 2, dims, NPY_FLOAT32, data);
    boost::python::handle<> handle( pyObj );
    boost::python::numeric::array arr( handle );
    return arr.copy();
}



/**
    rectToMat:      Takes a vector of Rectangles and converts the vector into cv::Mat in a way that
                    each row represents 1 rectangle ( 1 row = (x,y,w,h) of 1 rectangle)
                    @param std::vector<Rect>: Vector of rectangles

*/
boost::python::object PythonConversions::rectToMat(const std::vector<Rect> &vector){
    cv::Mat mat = cv::Mat(vector.size(), 4, CV_32S, cv::Scalar(0));
    for(size_t i = 0; i < vector.size(); i++){
        mat.at<int>(i, 0) = vector[i].x_;
        mat.at<int>(i, 1) = vector[i].y_;
        mat.at<int>(i, 2) = vector[i].w_;
        mat.at<int>(i, 3) = vector[i].h_;
    }

    return matToPythonNPArray(mat);
}


/**

    extractRects: Take a python list and creates rectangles from entries in the list.
                  The entries in the list should be either type of boost::python::tuple or Rect.
                  The other type of entries will be skipped.
                  @param boost::python::list &ns: Python list object

*/

std::vector<Rect> PythonConversions::PythonConversions::extractRects(boost::python::list& ns){
    std::vector<Rect> vector;
    for (int i = 0; i < len(ns); ++i)
    {
        boost::python::extract<boost::python::object> objectExtractor(ns[i]);
        boost::python::object o=objectExtractor();
        std::string object_classname = boost::python::extract<std::string>(o.attr("__class__").attr("__name__"));

        Rect r;
        if(object_classname=="Rect")
            r = Rect(boost::python::extract<Rect>(ns[i]));
        else if(object_classname=="tuple"){
            boost::python::tuple t = boost::python::extract<boost::python::tuple>(o);
            r.x_ = boost::python::extract<int>(t[0]);
            r.y_ = boost::python::extract<int>(t[1]);
            r.w_ = boost::python::extract<int>(t[2]);
            r.h_ = boost::python::extract<int>(t[3]);
        }
        else
            std::cout<<"this is not either a tuple or a Rect. This is an object: "<<object_classname<<std::endl;
        vector.push_back(r);
    }
    return vector;
}


/**

        py_rectToMat: Takes a python list of objects, extracts the rectangles and returns
                      Python Numpy array which each row of array is (x,y,w,h) of a rectangle.
                      @param boost::python::list &ns: Python list object
*/
boost::python::object PythonConversions::py_rectToMat(boost::python::list& ns)
{
    std::vector<Rect> vector = extractRects(ns);
    return rectToMat(vector);
}
