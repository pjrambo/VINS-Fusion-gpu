#ifndef SYS_UTILS_H
#define SYS_UTILS_H

#include <iostream>
#include <time.h>

namespace sys_utils
{

inline bool
float_equal( const float a, const float b )
{
    if ( std::abs( a - b ) < 1e-6 )
        return true;
    else
        return false;
}

inline bool
double_equal( const double a, const double b )
{
    if ( std::abs( a - b ) < 1e-6 )
        return true;
    else
        return false;
}

inline unsigned long long
timeInMicroseconds( void )
{
    struct timespec tp;

    clock_gettime( CLOCK_REALTIME, &tp );

    return ( tp.tv_sec * 1000000 + tp.tv_nsec / 1000 );
}

inline double
timeInSeconds( void )
{
    struct timespec tp;

    clock_gettime( CLOCK_REALTIME, &tp );

    return ( static_cast< double >( tp.tv_sec ) + static_cast< double >( tp.tv_nsec ) / 1000000000.0 );
}

inline void
PrintWarning( std::string str )
{
    std::cout << "\033[33;40;1m" << str << "\033[0m" << std::endl;
}

inline void
PrintError( std::string str )
{
    std::cout << "\033[31;47;1m" << str << "\033[0m" << std::endl;
}

inline void
PrintInfo( std::string str )
{
    std::cout << "\033[32;40;1m" << str << "\033[0m" << std::endl;
}
}
#endif // SYS_UTILS_H
