#include <camodocal/code_utils/math_utils/Polynomial.h>
#include <camodocal/code_utils/sys_utils.h>
#include <iostream>

using namespace math_utils;

Polynomial::Polynomial( ) {}

Polynomial::Polynomial( const int _order )
{
    m_order = _order;
    m_coeff = eigen_utils::Vector::Zero( m_order + 1 );
}

Polynomial::Polynomial( const eigen_utils::Vector _coeff )
{
    m_order = _coeff.size( ) - 1;
    m_coeff = _coeff;
}

double
Polynomial::getValue( const double in )
{
    return Evaluate( in );
}

eigen_utils::Vector
Polynomial::getRealRoot( double _y )
{
    eigen_utils::Vector realRoots, imagRoots;

    FindRoots( _y, m_coeff, realRoots, imagRoots );

    eigen_utils::Vector realRoot;
    for ( int i = 0; i < realRoots.size( ); i++ )
    {

        if ( imagRoots( i ) == 0 )
            realRoot = eigen_utils::pushback( realRoot, realRoots( i ) );
    }
    // std::cout<<"real roots: "<<realRoot<<std::endl;
    return realRoot;
}

eigen_utils::Vector
Polynomial::getRealRoot( double _y, double x_min, double x_max )
{
    eigen_utils::Vector realRoots, imagRoots;

    FindRoots( _y, m_coeff, realRoots, imagRoots );

    eigen_utils::Vector realRoot;
    for ( int i = 0; i < realRoots.size( ); i++ )
    {
        if ( imagRoots( i ) == 0 && realRoots( i ) >= x_min && realRoots( i ) <= x_max )
            realRoot = eigen_utils::pushback( realRoot, realRoots( i ) );
    }
    std::cout << "real roots: " << realRoot << std::endl;
    return realRoot;
}

double
Polynomial::getOneRealRoot( double _y, double x_min, double x_max )
{
    eigen_utils::Vector realRoots, imagRoots;

    FindRoots( _y, m_coeff, realRoots, imagRoots );

    // eigen_utils::Vector realRoot;
    //  std::cout << " max angle " << x_max ;
    for ( int i = 0; i < realRoots.size( ); ++i )
    {
        if ( imagRoots( i ) == 0 )
        {
            //      std::cout <<  " r " << realRoots(i);
            if ( realRoots( i ) >= x_min && realRoots( i ) <= x_max )
            {
                return realRoots( i );
                // realRoot = eigen_utils::pushback(realRoot, realRoots(i));
            }
        }
        if ( i == realRoots.size( ) - 1 )
            return 0;
    }
    //  std::cout << std::endl;

    // TODO:method to get one value
    // if (realRoot.size() >= 1)
    //    return realRoot(0);
    // else
    //    return 0;
}

eigen_utils::Vector
Polynomial::getValue( const eigen_utils::Vector& in )
{
    eigen_utils::Vector out( in.size( ) );

    for ( int i = in.size( ) - 1; i >= 0; --i )
    {
        out( i ) = Evaluate( in( i ) );
    }

    return out;
}

void
Polynomial::setPolyOrder( int _order )
{
    m_order = _order;
    m_coeff.resize( m_order );
}

int
Polynomial::getPolyOrder( ) const
{
    return m_order;
}

void
Polynomial::setPolyCoeff( const eigen_utils::Vector& value )
{
    m_coeff = value;
}

void
Polynomial::setPolyCoeff( int order_index, double value )
{
    m_coeff( order_index ) = value;
}

eigen_utils::Vector
Polynomial::getPolyCoeff( ) const
{
    return m_coeff;
}

double
Polynomial::getPolyCoeff( int order_index ) const
{
    return m_coeff( order_index );
}

Polynomial&
Polynomial::operator=( const Polynomial& other )
{
    if ( this != &other )
    {
        m_order = other.m_order;
        m_coeff = other.m_coeff;
    }

    return *this;
}

string
Polynomial::toString( ) const
{
    std::ostringstream oss;
    oss << "Polynomial :" << std::endl;
    oss << "|   order|" << getPolyOrder( ) << std::endl;
    oss << "|   coeff|" << getPolyCoeff( ).transpose( ) << std::endl;
    return oss.str( );
}

ostream&
operator<<( ostream& out, const Polynomial& poly )
{
    out << "Polynomial :" << std::endl;
    out << "|   order|" << poly.getPolyOrder( ) << std::endl;
    out << "|   coeff|" << poly.getPolyCoeff( ).transpose( ) << std::endl;
    return out;
}

double
Polynomial::Evaluate( double _x )
{
    double value_out = 0;

    for ( int i = m_order; i >= 0; --i )
    {
        value_out += m_coeff( i ) * pow( _x, i );
    }

    return value_out;
}

bool
Polynomial::FindRoots( const double y, const eigen_utils::Vector& polynomial_in, eigen_utils::Vector& real, eigen_utils::Vector& imaginary )
{
    if ( polynomial_in.size( ) == 0 )
        return false;

    int degree = polynomial_in.size( ) - 1;

    // count high order zero coefferent
    int zero_num = 0;
    for ( int i = degree; i >= 0; --i )
    {
        if ( polynomial_in( i ) == 0.0 )
            zero_num++;
        else
            break;
    }

    degree -= zero_num;
    eigen_utils::Vector polynomial_coeff( degree + 1 );
    polynomial_coeff = polynomial_in.segment( 0, degree + 1 );

    polynomial_coeff( 0 ) -= y;

    if ( degree == 0 )
    {
        std::cout << " Is the polynomial constant?" << std::endl;
        return false;
    }
    // Linear
    if ( degree == 1 )
    {
        FindLinearPolynomialRoots( polynomial_coeff, real, imaginary );
        return true;
    }
    if ( degree == 2 )
    {
        FindQuadraticPolynomialRoots( polynomial_coeff, real, imaginary );
        return true;
    }
    else if ( degree > 2 )
    {
        // The degree is now known to be at least 3. For cubic or higher
        // roots we use the method of companion matrices.

        // Divide by leading term
        const double leading_term = polynomial_coeff( degree );
        //    std::cout<< " polynomial_in: " << polynomial_in.transpose() <<
        //    std::endl;

        polynomial_coeff /= leading_term;

        // Build and balance the companion matrix to the polynomial.
        eigen_utils::Matrix companion_matrix( degree, degree );
        BuildCompanionMatrix( polynomial_coeff, &companion_matrix );
        // BalanceCompanionMatrix(&companion_matrix);
        // Find its (complex) eigenvalues.
        Eigen::EigenSolver< eigen_utils::Matrix > solver( companion_matrix, false );
        if ( solver.info( ) != Eigen::Success )
        {
            return false;
        }
        else
        {
            real      = solver.eigenvalues( ).real( );
            imaginary = solver.eigenvalues( ).imag( );

            return true;
        }
    }
    else
        return false;
}

void
Polynomial::FindLinearPolynomialRoots( const eigen_utils::Vector& polynomial, eigen_utils::Vector& real, eigen_utils::Vector& imag )
{
    real.resize( 1 );
    imag.resize( 1 );

    real( 0 ) = -polynomial( 0 ) / polynomial( 1 );
    imag( 0 ) = 0;
}

void
Polynomial::FindQuadraticPolynomialRoots( const eigen_utils::Vector& polynomial,
                                          eigen_utils::Vector& real,
                                          eigen_utils::Vector& imag )
{
    const double a      = polynomial( 2 );
    const double b      = polynomial( 1 );
    const double c      = polynomial( 0 );
    const double D      = b * b - 4 * a * c;
    const double sqrt_D = sqrt( fabs( D ) );

    real.resize( 2 );
    imag.resize( 2 );

    // Real roots.
    if ( D >= 0 )
    {
        // Stable quadratic roots according to BKP Horn.
        // http://people.csail.mit.edu/bkph/articles/Quadratics.pdf
        if ( b >= 0 )
        {
            real( 0 ) = ( -b - sqrt_D ) / ( 2.0 * a );
            imag( 0 ) = 0;

            real( 1 ) = ( 2.0 * c ) / ( -b - sqrt_D );
            imag( 1 ) = 0;
            return;
        }
        else
        {
            real( 0 ) = ( 2.0 * c ) / ( -b + sqrt_D );
            imag( 0 ) = 0;

            real( 1 ) = ( -b + sqrt_D ) / ( 2.0 * a );
            imag( 1 ) = 0;
            return;
        }
    }
    else
    {
        // Use the normal quadratic formula for the complex case.
        real( 0 ) = -b / ( 2.0 * a );
        imag( 0 ) = sqrt_D / ( 2.0 * a );

        real( 1 ) = -b / ( 2.0 * a );
        imag( 1 ) = -sqrt_D / ( 2.0 * a );
        return;
    }
}

void
Polynomial::BuildCompanionMatrix( const eigen_utils::Vector& polynomial, eigen_utils::Matrix* companion_matrix_ptr )
{
    eigen_utils::Matrix& companion_matrix = *companion_matrix_ptr;

    const int degree = polynomial.size( ) - 1;

    // companion_matrix.resize(degree, degree);
    companion_matrix.setZero( );
    companion_matrix.diagonal( -1 ).setOnes( );
    companion_matrix.col( degree - 1 ) = -polynomial.head( degree );
}

void
Polynomial::BalanceCompanionMatrix( eigen_utils::Matrix* companion_matrix_ptr )
{
    eigen_utils::Matrix& companion_matrix            = *companion_matrix_ptr;
    eigen_utils::Matrix companion_matrix_offdiagonal = companion_matrix;
    companion_matrix_offdiagonal.diagonal( ).setZero( );

    const int degree = companion_matrix.rows( );

    // gamma <= 1 controls how much a change in the scaling has to
    // lower the 1-norm of the companion matrix to be accepted.
    //
    // gamma = 1 seems to lead to cycles (numerical issues?), so
    // we set it slightly lower.
    const double gamma = 0.9;

    // Greedily scale row/column pairs until there is no change.
    bool scaling_has_changed;
    do
    {
        scaling_has_changed = false;

        for ( int i = 0; i < degree; ++i )
        {
            const double row_norm = companion_matrix_offdiagonal.row( i ).lpNorm< 1 >( );
            const double col_norm = companion_matrix_offdiagonal.col( i ).lpNorm< 1 >( );

            // Decompose row_norm/col_norm into mantissa * 2^exponent,
            // where 0.5 <= mantissa < 1. Discard mantissa (return value
            // of frexp), as only the exponent is needed.
            int exponent = 0;
            std::frexp( row_norm / col_norm, &exponent );
            exponent /= 2;

            if ( exponent != 0 )
            {
                const double scaled_col_norm = std::ldexp( col_norm, exponent );
                const double scaled_row_norm = std::ldexp( row_norm, -exponent );
                if ( scaled_col_norm + scaled_row_norm < gamma * ( col_norm + row_norm ) )
                {
                    // Accept the new scaling. (Multiplication by powers of 2
                    // should not
                    // introduce rounding errors (ignoring non-normalized
                    // numbers and
                    // over- or underflow))
                    scaling_has_changed = true;
                    companion_matrix_offdiagonal.row( i ) *= std::ldexp( 1.0, -exponent );
                    companion_matrix_offdiagonal.col( i ) *= std::ldexp( 1.0, exponent );
                }
            }
        }
    } while ( scaling_has_changed );

    companion_matrix_offdiagonal.diagonal( ) = companion_matrix.diagonal( );
    companion_matrix                         = companion_matrix_offdiagonal;
}

PolynomialFit::PolynomialFit( int _order )
: Polynomial( _order )
{
    poly = new Polynomial( _order );
    samples.clear( );
    data_size = 0;
}

PolynomialFit::PolynomialFit( int _order, eigen_utils::Vector _x, eigen_utils::Vector _y )
: Polynomial( _order )
{
    poly = new Polynomial( _order );
    samples.clear( );
    data_size = 0;

    loadSamples( _x, _y );
}

void
PolynomialFit::loadSamples( const eigen_utils::Vector& x, const eigen_utils::Vector& y )
{
    if ( x.size( ) != y.size( ) )
        return;

    for ( int i = x.size( ) - 1; i >= 0; --i )
    {
        Sample sample;
        sample.x = x( i );
        sample.y = y( i );
        samples.push_back( sample );
        data_size++;
    }
}

void
PolynomialFit::loadSample( const Sample sample )
{
    samples.push_back( sample );
    data_size++;
}

void
PolynomialFit::clearSamples( )
{
    samples.clear( );
    data_size = 0;
}

eigen_utils::Vector
PolynomialFit::getCoeff( )
{
    coeff = this->Fit( );
    poly->setPolyCoeff( coeff );

    return coeff;
}

Polynomial&
PolynomialFit::getPolynomial( )
{
    return *poly;
}

eigen_utils::Vector
PolynomialFit::Fit( )
{
    int num_constraints = data_size;

    const int degree = this->getPolyOrder( );

    eigen_utils::Matrix lhs = eigen_utils::Matrix::Zero( num_constraints, num_constraints );
    eigen_utils::Vector rhs = eigen_utils::Vector::Zero( num_constraints );

    int row = 0;

    for ( int i = 0; i < data_size; ++i )
    {
        const Sample& sample = samples[i];

        for ( int j = 0; j <= degree; ++j )
        {
            lhs( row, j ) = pow( sample.x, degree - j );
        }

        rhs( row ) = sample.y;

        ++row;
    }

    return ( lhs.fullPivLu( ).solve( rhs ).segment( 0, degree + 1 ) ).reverse( );
}
