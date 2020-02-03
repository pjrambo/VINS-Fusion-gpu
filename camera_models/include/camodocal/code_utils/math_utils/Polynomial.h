#ifndef POLYNOMIAL_H
#define POLYNOMIAL_H

#include <math.h>
#include <vector>

#include <iostream>

#include <camodocal/code_utils/eigen_utils.h>
#include <eigen3/Eigen/Eigen>

namespace math_utils
{

class Polynomial
{

    public:
    Polynomial( );
    Polynomial( const int _order );
    Polynomial( const eigen_utils::Vector _coeff );

    eigen_utils::Vector getRealRoot( double _y );
    eigen_utils::Vector getRealRoot( double _y, double x_min, double x_max );
    double getOneRealRoot( double _y, double x_min, double x_max );

    eigen_utils::Vector getValue( const eigen_utils::Vector& in );
    double getValue( const double in );

    void setPolyOrder( int _order );
    int getPolyOrder( ) const;

    void setPolyCoeff( const eigen_utils::Vector& value );
    void setPolyCoeff( int order_index, double value );

    eigen_utils::Vector getPolyCoeff( ) const;
    double getPolyCoeff( int order_index ) const;

    Polynomial& operator=( const Polynomial& other );
    friend std::ostream& operator<<( std::ostream& out, const Polynomial& poly );
    std::string toString( void ) const;

    private:
    double Evaluate( double _x );

    // _y = f(x)
    bool FindRoots( const double y, const eigen_utils::Vector& polynomial_in, eigen_utils::Vector& real, eigen_utils::Vector& imaginary );

    void FindLinearPolynomialRoots( const eigen_utils::Vector& Polynomial, eigen_utils::Vector& real, eigen_utils::Vector& imag );

    void FindQuadraticPolynomialRoots( const eigen_utils::Vector& Polynomial,
                                       eigen_utils::Vector& real,
                                       eigen_utils::Vector& imag );

    void BuildCompanionMatrix( const eigen_utils::Vector& Polynomial, eigen_utils::Matrix* companion_matrix_ptr );

    // Balancing function as described by B. N. Parlett and C. Reinsch,
    // "Balancing a Matrix for Calculation of Eigenvalues and Eigenvectors".
    // In: Numerische Mathematik, Volume 13, Number 4 (1969), 293-304,
    // Springer Berlin / Heidelberg. DOI: 10.1007/BF02165404
    void BalanceCompanionMatrix( eigen_utils::Matrix* companion_matrix_ptr );

    private:
    int m_order;
    // c0, c1, c2, ..., cn
    // c0 + c1X + c2X^2 + ... +cnX^n
    eigen_utils::Vector m_coeff;
};

typedef struct
{
    double x;
    double y;
} Sample;

class PolynomialFit : public Polynomial
{
    public:
    PolynomialFit( int _order );
    PolynomialFit( int _order, eigen_utils::Vector _x, eigen_utils::Vector _y );

    void loadSamples( const eigen_utils::Vector& x, const eigen_utils::Vector& y );
    void loadSample( const Sample sample );
    void clearSamples( );

    eigen_utils::Vector getCoeff( );
    Polynomial& getPolynomial( );

    private:
    eigen_utils::Vector Fit( );

    private:
    int data_size;
    vector< Sample > samples;
    Polynomial* poly;
    eigen_utils::Vector coeff;
};
}
#endif // POLYNOMIAL_H
