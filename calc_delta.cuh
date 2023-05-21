// gaussian and plane functions 
#ifndef CALC_DELTA_CUH
#define CALC_DELTA_CUH

#include <cmath>

__device__ double gaussian(double x, double y, double A, double sigx, double sigy)
{
	double x0 = 156.;
	double y0 = 136.;
	double result = A * exp(-pow(x-x0,2.)/(2.*pow(sigx,2.)) - pow(y-y0,2.)/(2.*pow(sigy,2.)));
	return result;
}
__device__ double plane(double x, double y, double b0, double b1, double b2) 
{
	double result = b0 + b1*x + b2*y;
	return result;
}

#endif /* CALC_DELTA_CUH */
