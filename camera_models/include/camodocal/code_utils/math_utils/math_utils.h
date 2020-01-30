#ifndef _MATH_UTILS_H
#define _MATH_UTILS_H

#include <eigen3/Eigen/Eigen>
#include <math.h>
//#include <code_utils/eigen_utils.h>
#include "../eigen_utils.h"

#define DEG2RAD ( M_PI / 180.0 )
#define RAD2DEG ( 180.0 / M_PI )

#define DEG2RADF ( float )( M_PI / 180.0 )
#define RAD2DEGF ( float )( 180.0 / M_PI )

namespace math_utils
{

/// \brief Skew
/// \param vec :input 3-dof vector
/// \return   :output skew symmetric matrix
inline Eigen::Matrix3d
vectorToSkew( const Eigen::Vector3d vec )
{
    Eigen::Matrix3d mat;

    mat << 0.0, -vec( 2 ), vec( 1 ), vec( 2 ), 0.0, -vec( 0 ), -vec( 1 ), vec( 0 ), 0.0;

    return mat;
}

template < typename T >
inline Eigen::Matrix< T, 3, 3 >
vectorToSkew( const Eigen::Matrix< T, 3, 1 > vec )
{
    Eigen::Matrix< T, 3, 3 > mat;

    mat( 0, 0 ) = T( 0 );
    mat( 0, 1 ) = -vec( 2 );
    mat( 0, 2 ) = vec( 1 );
    mat( 1, 0 ) = vec( 2 );
    mat( 1, 1 ) = T( 0 );
    mat( 1, 2 ) = -vec( 0 );
    mat( 2, 0 ) = -vec( 1 );
    mat( 2, 1 ) = vec( 0 );
    mat( 2, 2 ) = T( 0 );

    return mat;
}

/// \brief Skew
/// \param vec :input 3-dof vector
/// \return   :output skew symmetric matrix
inline Eigen::Vector3d
skewToVector( const Eigen::Matrix3d mat )
{
    return Eigen::Vector3d( mat( 2, 1 ), mat( 0, 2 ), mat( 1, 0 ) );
}

/// \brief eigenQtoCeresQ
/// \param q_eigen
/// \return
template < typename T >
inline T*
eigenQtoCeresQ( const T* const q_eigen )
{
    // Eigen convention (x, y, z, w)
    // Ceres convention (w, x, y, z)
    T q_ceres[4] = { q_eigen[3], q_eigen[0], q_eigen[1], q_eigen[2] };

    return q_ceres;
}

/// \brief ceresQtoEigenQ
/// \param q_ceres
/// \return
template < typename T >
inline T*
ceresQtoEigenQ( const T* const q_ceres )
{
    // Ceres convention (w, x, y, z)
    // Eigen convention (x, y, z, w)
    T q_eigen[4] = { q_ceres[1], q_ceres[2], q_ceres[3], q_ceres[0] };

    return q_eigen;
}

/// \brief orientaionOfVectors
/// \param v
/// \param theta
/// \return
inline Eigen::Quaterniond
orientaionOfVectors( const Eigen::Vector3d v, const double theta )
{
    return Eigen::Quaterniond( cos( 0.5 * theta ), sin( 0.5 * theta ) * v( 0 ), sin( 0.5 * theta ) * v( 1 ),
                               sin( 0.5 * theta ) * v( 2 ) );
}

/// \brief orientaionOfVectors :orientation betreen vector v2 to vetor v1
/// \param v1 :begin vector
/// \param v2 :end vector
/// \return   :q_12
inline Eigen::Quaterniond
orientaionOfVectors( const Eigen::Vector3d v1, const Eigen::Vector3d v2 )
{
    double theta = acos( v1.dot( v2 ) );

    Eigen::Vector3d Vk = v1.cross( v2 );

    return Eigen::Quaterniond( cos( 0.5 * theta ), sin( 0.5 * theta ) * Vk( 0 ), sin( 0.5 * theta ) * Vk( 1 ),
                               sin( 0.5 * theta ) * Vk( 2 ) );
}

inline Eigen::Matrix3d
eularToDCM( double yaw, double pitch, double roll )
{
    //    double y = ypr(0) / 180.0 * M_PI;
    //    double p = ypr(1) / 180.0 * M_PI;
    //    double r = ypr(2) / 180.0 * M_PI;
    Eigen::Matrix3d Rz;
    Rz << cos( yaw ), -sin( yaw ), 0, sin( yaw ), cos( yaw ), 0, 0, 0, 1;

    Eigen::Matrix3d Ry;
    Ry << cos( pitch ), 0., sin( pitch ), 0., 1., 0., -sin( pitch ), 0., cos( pitch );

    Eigen::Matrix3d Rx;
    Rx << 1., 0., 0., 0., cos( roll ), -sin( roll ), 0., sin( roll ), cos( roll );

    return Rz * Ry * Rx;
}
}
#endif // _MATH_UTILS_H
