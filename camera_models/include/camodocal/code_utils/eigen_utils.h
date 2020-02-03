#ifndef EIGEN_UTLIS_H
#define EIGEN_UTLIS_H

#include <eigen3/Eigen/Eigen>

#include <iostream>

using namespace std;

namespace eigen_utils
{

typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  Matrix;

template <int num_rows = Eigen::Dynamic, int num_cols = Eigen::Dynamic>
struct EigenTypes
{

    typedef Eigen::Matrix<double, num_rows, num_cols, Eigen::RowMajor> Matrix;

    typedef Eigen::Matrix<double, num_rows, 1> Vector;
};

/**
 *
 */
template <class vector_t>
inline vector_t
SwapSequence(vector_t vec_in)
{
//    int size = vec_in.size();
//    vector_t vec_out(size);

//    for (int i = 0; i < size; i++)
//    {
//        vec_out(size - 1 - i) = vec_in(i);
//    }

//    return vec_out;

  return vec_in.reverse();
}

inline Vector
pushback(Vector vec_in, const double value)
{
    Vector vec_out(vec_in.size() + 1);

    vec_out.segment(0, vec_in.size()) = vec_in;
    vec_out(vec_in.size()) = value;

    return vec_out;
}

template <class T>
inline void
copyMat3ToArry(const Eigen::Matrix<T, 3, 3> mat, T* data)
{
    data[0] = T(mat(0, 0));
    data[1] = T(mat(1, 0));
    data[2] = T(mat(2, 0));
    data[3] = T(mat(0, 1));
    data[4] = T(mat(1, 1));
    data[5] = T(mat(2, 1));
    data[6] = T(mat(0, 2));
    data[7] = T(mat(1, 2));
    data[8] = T(mat(2, 2));
}
template <class T>
inline void
copyArryToMat3(T* data, Eigen::Matrix<T, 3, 3>& mat)
{
    mat(0, 0) = data[0];
    mat(1, 0) = data[1];
    mat(2, 0) = data[2];
    mat(0, 1) = data[3];
    mat(1, 1) = data[4];
    mat(2, 1) = data[5];
    mat(0, 2) = data[6];
    mat(1, 2) = data[7];
    mat(2, 2) = data[8];
}
template <class T>
inline void
copyVector3ToArry(const Eigen::Matrix<T, 3, 1> vec, T* data)
{
    data[0] = T(vec(0));
    data[1] = T(vec(1));
    data[2] = T(vec(2));
}
template <class T>
inline void
copyArryToVector3(T* data, Eigen::Matrix<T, 3, 1>& vec)
{
    vec(0) = data[0];
    vec(1) = data[1];
    vec(2) = data[2];
}

};

#endif // EIGEN_UTLIS_H
