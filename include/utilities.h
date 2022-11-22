#ifndef UTILITIES_H
#define UTILITIES_H

#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// specify pybind namespace
namespace py = pybind11;

/**
 * @brief print sequnce
 *
 * @tparam sequence
 * @param seq
 */
template <typename Sequence>
void print(Sequence &&seq) {
    for (auto &i : seq) std::cout << i << " ";
    std::cout << "\n";
}

/**
 * @brief Makes a copy of the numpy array as a flat vector
 *
 * @tparam T
 * @tparam N
 * @param array
 * @return std::vector<T>
 */
template <typename T, int N>
inline std::vector<T> copy_array_to_vector(py::array_t<T, N> &array) {
    std::vector<T> vec(array.size());
    std::copy(array.data(0), array.data(0) + array.size(), vec.begin());
    return vec;
}

/**
 * @brief Moves a Sequence (like std::vector) to py::array_t
 * See https://github.com/pybind/pybind11/issues/1042#issuecomment-642215028
 * There are modifications to the original function to reshape the array using the shape parameter.
 *
 * @tparam Sequence
 * @param seq
 * @param shape
 * @return py::array_t<typename Sequence::value_type>
 */
template <typename Sequence>
inline py::array_t<typename Sequence::value_type> as_pyarray(Sequence &&seq, py::array::ShapeContainer &&shape = {}) {
    // get data and size
    auto data = seq.data();
    auto seq_size = seq.size();

    // move seq to heap
    std::unique_ptr<Sequence> seq_ptr = std::make_unique<Sequence>(std::move(seq));

    // get capsule object to manage the Sequence data on the heap
    // the lambda function is a destructor that deletes the heap data
    // by reacquiring the ptr with a unique_ptr, which immediately goes
    // out of scope deleting the data
    auto capsule =
        py::capsule(seq_ptr.get(), [](void *p) { std::unique_ptr<Sequence>(reinterpret_cast<Sequence *>(p)); });

    // release ownership of the Sequence data on the heap
    seq_ptr.release();
    // this is now solely managed by the capsule object

    // get default shape that the py::array should be
    // if user did not define it this will be same size as input Sequence
    if (shape->size() == 0) shape->push_back(seq_size);

    // compute stride values for the array
    // this assumes that the array will be in column major order
    std::vector<py::ssize_t> stride(shape->size(), sizeof(typename Sequence::value_type));
    auto shape_it = shape->begin();
    for (auto stride_it = stride.begin() + 1; stride_it != stride.end(); stride_it++) {
        *stride_it = *(stride_it - 1) * *shape_it;
        shape_it++;
    }

    // return the array
    return py::array_t<typename Sequence::value_type>(shape, stride, data, capsule);
}

/**
 * @brief convert to numpy array
 *
 * @tparam Sequence
 * @param seq
 * @param shape
 * @return py::array_t<typename Sequence::value_type>
 */
template <typename Sequence>
inline py::array_t<typename Sequence::value_type> as_pyarray(Sequence &seq, py::array::ShapeContainer &&shape = {}) {
    return as_pyarray(std::move(seq), std::move(shape));
}

#endif