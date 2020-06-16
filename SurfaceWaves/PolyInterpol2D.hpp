#ifndef POLYINTERPOL2D_H
#define POLYINTERPOL2D_H


class PolyInterpol2D {
private:
	double px[3];
	double py[3];

	double a0, a1, a2;

public:
	PolyInterpol2D(double px_in[3], double py_in[3]);

	double Get_a0();
	double Get_a1();
	double Get_a2();

	double Func(double x);
	double DFunc(double x);
	double DDFunc(double x);
};


#endif




















































