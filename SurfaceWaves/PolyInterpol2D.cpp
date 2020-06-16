#include "PolyInterpol2D.hpp"

// Constructor
PolyInterpol2D::PolyInterpol2D(double px_in[3], double py_in[3])
{
	for (int i = 0; i < 3; i++){
		px[i] = px_in[i];
		py[i] = py_in[i];
	}

	double fx0, fx1, fx2, fx0x1, fx1x2, fx0x1x2;

	fx0 = py[0];
	fx1 = py[1];
	fx2 = py[2];

	fx0x1 = (fx1 - fx0)/(px[1] - px[0]);
	fx1x2 = (fx2 - fx1)/(px[2] - px[1]);

	fx0x1x2 = (fx1x2 - fx0x1)/(px[2] - px[0]);

	a0 = fx0;
	a1 = fx0x1;
	a2 = fx0x1x2;
}

// Get-functions
double PolyInterpol2D::Get_a0()
{
	return a0;
}

double PolyInterpol2D::Get_a1()
{
	return a1;
}

double PolyInterpol2D::Get_a2()
{
	return a2;
}

// Approximated functions
double PolyInterpol2D::Func(double x)
{
	return a0 + a1*(x - px[0]) + a2*(x - px[0])*(x - px[1]);
}

double PolyInterpol2D::DFunc(double x)
{
	return a1 - a2*(px[1] + px[0]) + 2*a2*x;
}

double PolyInterpol2D::DDFunc(double x)
{
	return 2*a2;
}

















































