/*********************************************************
 *                                                       *
 * Surface waves                                         *
 *                                                       *
 * Created by Sander Bøe Thygesen, June 15th 2020        *
 *                                                       *
 * Code used for Master thesis in applied mathematics    *
 *														 *
 ********************************************************/

#define _USE_MATH_DEFINES

#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <math.h>
#include <tgmath.h>
#include <vector>
#include <mkl.h>
#include <boost/numeric/odeint.hpp>
#include <boost/math/quadrature/gauss.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <chrono>
#include <omp.h>

#include "PolyInterpol2D.hpp"

// Definitions
#define	G 9.81   
#define L 10.0
#define H 1.0

using namespace std;
using namespace std::chrono;
using namespace boost::numeric::odeint;
using namespace boost::math::quadrature;
using namespace boost::numeric::ublas;

// Global variables 
const int NUM_POINTS_0 = 200;
const int NUM_POINTS_1 = 50;
const int NUM_POINTS_2 = 200;
const int NUM_POINTS_3 = 50;
				
const int NUM_POINTS_T = 8000;	

const double x_start = 0.0;
const double x_end = L;
const double z_start = -H;
const double z_end = 0.4;		// Must be changed if simulation yields higher amplitude waves.

const int NUM_POINTS_X = (int)round(L)*15;			      
const int NUM_POINTS_Z = (int)round(z_end - z_start)*15; 

const double T_START = 0.0;
const double T_FINAL = 5.0;
const double DT = T_FINAL/NUM_POINTS_T;											// Step size in time

const double Shift = 10.6;

const double RHO = 2.0;															// Threshold for when to use numerical integrals

const int Tot_N = NUM_POINTS_0 + NUM_POINTS_1 + NUM_POINTS_2 + NUM_POINTS_3;

const double StableThreshold = pow(10.0,10.0);

const int N[4] = {NUM_POINTS_0, NUM_POINTS_1, NUM_POINTS_2, NUM_POINTS_3};

const double epsilon1 = 1.0;

bool Testing = false;

typedef boost::array<double, NUM_POINTS_0> state_type1;
typedef boost::array<double, NUM_POINTS_0*2> state_type2;						// Times two since both phi and eta is calculated at the surface
runge_kutta4<state_type2> Stepper;

double Results[NUM_POINTS_T][NUM_POINTS_0];										// Surface wave solution
double Results_Phi_z[NUM_POINTS_T][NUM_POINTS_0];
double Results_Phi_x[NUM_POINTS_T][NUM_POINTS_0];
double Results_Eta_x[NUM_POINTS_T][NUM_POINTS_0];
double EdgeValues[NUM_POINTS_T][3];

double BoundaryValues[NUM_POINTS_T][Tot_N*2];									// Should contain all the boundary data. The data should be ordered in the direction of paramtrization and have PHI_STAR first, then PHI.

double SystemLeftMatrix[Tot_N][Tot_N];
double SystemRightMatrix[Tot_N][Tot_N];

state_type1 eta, phi;
double SendEta[NUM_POINTS_0];

int counter = 0;

auto LastCheck = high_resolution_clock::now();
auto CurrentCheck = high_resolution_clock::now();
auto TimeDuration = duration_cast<milliseconds>(CurrentCheck - LastCheck);

void Dump() 
{
	std::ofstream Solution;
	Solution.open("Data/Solution.csv");
	std::ofstream Solution_Phi_z;
	Solution_Phi_z.open("Data/Solution_Phi_z.csv");
	std::ofstream Solution_Phi_x;
	Solution_Phi_x.open("Data/Solution_Phi_x.csv");
	std::ofstream Solution_Eta_x;
	Solution_Eta_x.open("Data/Solution_Eta_x.csv");
	std::ofstream EdgeValuesFile;
	EdgeValuesFile.open("Data/EdgeValues.csv");
	for (int j = 0; j < NUM_POINTS_T; j++){
		for (int i = 0; i < NUM_POINTS_0; i++) {
			Solution << Results[j][i] << " ";
			Solution_Phi_z << Results_Phi_z[j][i] << " ";
			Solution_Phi_x << Results_Phi_x[j][i] << " ";
			Solution_Eta_x << Results_Eta_x[j][i] << " ";
		}
		EdgeValuesFile << "t = " << (double)((j+1)*DT) << ": Absolute difference between edge values = " << fabs(EdgeValues[j][0] - EdgeValues[j][1]) << endl
		<< "left edge = " << Results_Eta_x[j][0] << " and right edge = " << Results_Eta_x[j][1] << endl;
		
		Solution << endl;
		Solution_Phi_z << endl;
		Solution_Phi_x << endl;
		Solution_Eta_x << endl;
	}

	return;
}

inline int Translate_t(double t)
{
	return (int)(round(t*(NUM_POINTS_T)/T_FINAL));
}

inline double h(double x)
{
	//return H;
	return H - 0.25*exp(-4.0*pow(x-3.0,2.0));
	//return H - 0.25*exp(-0.2*pow(x-10.0,2.0));
	//return H - x/13;
	/*if (x < 8) {
		return H;
	} else if (x >= 8 and x < 16) {
		return -0.1*(x - 8) + H;
	} else if (x >= 16 and x < 24) {
		return 0.2;
	} else if (x >= 24 and x < 32) {
		return 0.1*(x - 32) + H;
	} else if (x >= 32) {
		return H;
	}*/
	/*if (x < Shift) {
		return H;
	} else if (x >= Shift and x < Shift + 1.6) {
		return -0.2625*(x - Shift) + H;
	} else if (x >= Shift + 1.6 and x < Shift + 2.0*1.6) {
		return 0.11;
	} else if (x >= Shift + 2.0*1.6 and x < Shift + 3.0*1.6) {
		return 0.2625*(x - (Shift + 3.0*1.6)) + H;
	} else if (x >= Shift + 3.0*1.6) {
		return H;
	}*/
}

inline double dxh(double x)
{
	//return 0.0;
	return 2.0*exp(-4.0*pow(x - 3.0,2.0))*(x-3.0);
	//return 0.1*exp(-0.2*pow(x-10.0,2.0))*(x-10.0);
	//return -1.0/13.0;
	/*if (x < 8) {
		return 0;
	} else if (x >= 8 and x < 16) {
		return -0.1;
	} else if (x >= 16 and x < 24) {
		return 0;
	} else if (x >= 24 and x < 32) {
		return 0.1;
	} else if (x >= 32) {
		return 0;
	}*/
	/*if (x < Shift) {
		return 0.0;
	} else if (x >= Shift and x < Shift + 1.6) {
		return -0.2625;
	} else if (x >= Shift + 1.6 and x < Shift + 2.0*1.6) {
		return 0.0;
	} else if (x >= Shift + 2.0*1.6 and x < Shift + 3.0*1.6) {
		return 0.2625;
	} else if (x >= Shift + 3.0*1.6) {
		return 0.0;
	}*/
}

inline double InflowX(double z, double t)
{
	//return 0.0;
	//return 1.4*exp(-10.0*pow(t - 1.0, 2.0));
	return 0.4*exp(-6.0*pow(t - 2.0, 2.0)); //*exp(-75.0*pow(z + 0.5, 2.0));
	//return 2.8*exp(-6.0*pow(t - 2.0, 2.0));//*exp(-75.0*pow(z + 0.5, 2.0));
	//return 1.5*exp(-6.0*pow(t - 1.5, 2.0));
}

inline double InflowZ(double z, double t)
{
	return 0;
}

inline double GreensFunction(double x, double z, double m_1, double m_2)
{
	return log(sqrt(pow(x - m_1, 2.0) + pow(z - m_2, 2.0)))/(2.0*M_PI);
}

inline double dnGreensFunction(double x, double dsx, double z, double dsz, double m_1, double m_2)
{
	return ((x - m_1)*dsz - (z - m_2)*dsx)/((pow(x - m_1, 2.0) + pow(z - m_2, 2.0))*sqrt(pow(dsx, 2.0) + pow(dsz, 2.0))*2.0*M_PI);
}

double x(double s, int i)
{
	switch(i) {
		case 0:
			return L*(1.0-s);
		case 1:
			return 0.0;
		case 2:
			return L*s;
		case 3:
			return L;
		default:
			cout << "Warning! Illigal i: " << i << endl;
			return 0;
	}
}

double dsx(double s, int i)
{
	switch(i) {
		case 0:
			return -L;
		case 1:
			return 0.0;
		case 2:
			return L;
		case 3:
			return 0.0;
		default:
			cout << "Warning! Illigal i: " << i << endl;
			return 0;
	}
}

double z(double s, int i)
{
	// REMEMBER THAT WHEN i = 0 s GOES FROM RIGHT TO LEFT WHILE eta GOES FROM LEFT TO RIGHT!!!
	double Ds = 1.0/N[0];
	if (i == 0) {
		int index;
		index = (int)(round(s*(double)(N[i] - 1)));
		return eta[NUM_POINTS_0 - 1 - index];
	} else if (i == 1) {
		double px[3] = {x(1.0 - Ds/2.0, 0), x(1.0 - Ds/2.0 - Ds, 0), x(1.0 - Ds/2.0 - 2.0*Ds, 0)};
		double py[3] = {eta[0], eta[1], eta[2]};
		PolyInterpol2D Poly_y(px, py);
		return Poly_y.Func(0)*(1-s) - s*h(0);
		//return eta[0]*(1-s) - s*h(0);
	} else if (i == 2) {
		return -h(x(s,i));
	} else if (i == 3) {
		double px[3] = {x(Ds/2.0 + 2.0*Ds, 0), x(Ds/2.0 + Ds, 0), x(Ds/2.0, 0)};
		double py[3] = {eta[NUM_POINTS_0-3], eta[NUM_POINTS_0-2], eta[NUM_POINTS_0-1]};
		PolyInterpol2D Poly_y(px, py);
		return -h(L)*(1.0 - s) + Poly_y.Func(L)*s;
		//return -h(L)*(1 - s) + eta[NUM_POINTS_0-1]*s;
	} else {
		cout << "Warning! Illigal i: " << i << endl;
	}
}

double dsz(double s, int i)
{
	// REMEMBER THAT WHEN i = 0 s GOES FROM RIGHT TO LEFT WHILE eta GOES FROM LEFT TO RIGHT!!!
	double Ds = 1.0/N[0];
	if (i == 0) {
		int index;
		index = (int)(round(s*(N[i] - 1)));
		if (index == 0) {
			double px[3] = {s, s + Ds, s + 2*Ds};
			double py[3] = {z(px[0],i), z(px[1],i), z(px[2],i)};
			PolyInterpol2D Poly_y(px, py);
			return Poly_y.DFunc(s);
		} else if (index == N[i] - 1) {
			double px[3] = {s - 2*Ds, s - Ds, s};
			double py[3] = {z(px[0],i), z(px[1],i), z(px[2],i)};
			PolyInterpol2D Poly_y(px, py);
			return Poly_y.DFunc(s);
		} else {
			double px[3] = {s - Ds, s, s + Ds};
			double py[3] = {z(px[0],i), z(px[1],i), z(px[2],i)};
			PolyInterpol2D Poly_y(px, py);
			return Poly_y.DFunc(s);
		}
	} else if (i == 1) {
		double px[3] = {x(1.0 - Ds/2.0, 0), x(1.0 - Ds/2.0 - Ds, 0), x(1.0 - Ds/2.0 - 2.0*Ds, 0)};
		double py[3] = {eta[0], eta[1], eta[2]};
		PolyInterpol2D Poly_y(px, py);
		return -Poly_y.Func(0) - h(0);
		//return -eta[0] - h(0);
	} else if (i == 2) {
		return -dxh(x(s,i))*dsx(s,i);
	} else if (i == 3) {
		double px[3] = {x(Ds/2.0 + 2.0*Ds, 0), x(Ds/2.0 + Ds, 0), x(Ds/2.0, 0)};
		double py[3] = {eta[NUM_POINTS_0-3], eta[NUM_POINTS_0-2], eta[NUM_POINTS_0-1]};
		PolyInterpol2D Poly_y(px, py);
		return h(L) + Poly_y.Func(L);
		//return h(L) + eta[NUM_POINTS_0-1];
	} else {
		cout << "Warning! Illigal i: " << i << endl;
	}	
}

state_type2 Set_initial_conditions()
{	
	state_type2 eta_phi;

	double Ds = 1.0/NUM_POINTS_0;
	for (int i = 0; i < NUM_POINTS_0; i++) {
		double s_i = Ds*(NUM_POINTS_0 - 1 - i + 1.0/2.0);
		double x_val = x(s_i, 0);
		eta_phi[i] = 0.0;//0.1*exp(-5.0*pow(x_val - L/2.0, 2.0));
		eta_phi[NUM_POINTS_0 + i] = 0.0; // 0.2*exp(-5*pow(i*L/(NUM_POINTS_0-1) - L/2,2));
	}

	return eta_phi;
}

double AntiDerivative(double u, double Constant, double epsilon)
{
	return -2.0*u + u*log(Constant*pow(epsilon*u, 2.0));
}

void CalculateElement(matrix<double>* A, matrix<double>* B, int i, int j, int l, int k, bool DoB = true)
{
	double Ds_l = 1.0/N[i];
	double Ds_k = 1.0/N[j];
	double epsilon = Ds_l/2;
	double s_l = Ds_l*(l + 1.0/2.0);
	double s_k = Ds_k*(k + 1.0/2.0);
	double ml1 = x(s_l, i);
	double ml2 = z(s_l, i);
	double mk1 = x(s_k, j);
	double mk2 = z(s_k, j);

	double d = (double)sqrt(pow(ml1 - mk1, 2) + pow(ml2 - mk2, 2));

	double px[3];
	if (l == 0) {
		px[0] = s_l;
		px[1] = s_l + Ds_l;
		px[2] = s_l + 2.0*Ds_l;
	} else if (l == N[i] - 1) {
		px[0] = s_l - 2.0*Ds_l;
		px[1] = s_l - Ds_l;
		px[2] = s_l;
	} else {
		px[0] = s_l - Ds_l;
		px[1] = s_l;
		px[2] = s_l + Ds_l;
	}
	double py[3] = {z(px[0],i), z(px[1],i), z(px[2],i)};
	PolyInterpol2D* Poly_z = new PolyInterpol2D(px, py);

	double dsC1 = dsx(s_l,i);
	double dsC2 = dsz(s_l,i);
	double VelLength = sqrt(pow(dsC1, 2.0) + pow(dsC2, 2.0));

	if (d == 0) {
		double a1 = x(1.0,i) - x(0.0,i);
		double alpha1 = a1;

		double b1 = Poly_z->Get_a1();
		double b2 = Poly_z->Get_a2();
		double beta1 = b1 + b2*Ds_l;
		double beta2 = b2;

		// Matrix A:
		(*A)(k,l) = epsilon*(VelLength/(M_PI))*((alpha1*beta2)/pow(pow(alpha1,2.0) + pow(beta1,2.0),3.0/2.0));

		// Matrix B:
		double Constant = pow(alpha1, 2.0) + pow(beta1, 2.0);

		if (DoB) {
			(*B)(k,l) = (VelLength*epsilon/(4.0*M_PI))*(AntiDerivative(1.0, Constant, epsilon) - AntiDerivative(-1.0, Constant, epsilon));
		} else {
			(*B)(k,l) = 0;
		}

	} else if (d <= RHO) {
		double I_dnG, I_G;
		if (i == 0) {
			auto F_dnG = [&Poly_z, &mk1, &mk2, &i](double s) {return (dnGreensFunction(x(s,i), dsx(s,i), Poly_z->Func(s), Poly_z->DFunc(s), mk1, mk2));};
			auto F_G = [&Poly_z, &mk1, &mk2, &i](double s) {return (GreensFunction(x(s,i), Poly_z->Func(s), mk1, mk2));};
			I_dnG = VelLength*gauss<double, 10>::integrate(F_dnG, s_l - epsilon, s_l + epsilon);
			if (DoB) {
				I_G = VelLength*gauss<double, 10>::integrate(F_G, s_l - epsilon, s_l + epsilon);
			} else {
				I_G = 0;
			}
		} else {
			auto F_dnG = [&mk1, &mk2, &i](double s) {return (dnGreensFunction(x(s,i), dsx(s,i), z(s,i), dsz(s,i), mk1, mk2));};
			auto F_G = [&mk1, &mk2, &i](double s) {return (GreensFunction(x(s,i), z(s,i), mk1, mk2));};
			I_dnG = VelLength*gauss<double, 10>::integrate(F_dnG, s_l - epsilon, s_l + epsilon);
			if (DoB) {
				I_G = VelLength*gauss<double, 10>::integrate(F_G, s_l - epsilon, s_l + epsilon);
			} else {
				I_G = 0;
			}
		}
		// Matrix A:
		(*A)(k,l) = I_dnG;
		// Matrix B:
		(*B)(k,l) = I_G;

	} else {
		double Length = Ds_l;

		// Matrix A:
		(*A)(k,l) = dnGreensFunction(ml1, dsx(s_l,i), ml2, dsz(s_l,i), mk1, mk2)*Length*VelLength;
		// Matrix B:
		if (DoB) {
			(*B)(k,l) = GreensFunction(ml1, ml2, mk1, mk2)*Length*VelLength; 
		}
		else {
			(*B)(k,l) = 0;
		}
	}

	delete Poly_z;

	return;
}

void CalculateBlock(matrix<double>* A, matrix<double>* B, int i, int j, bool DoB = true)
{
	#pragma omp parallel for collapse(2)
	for (int k = 0; k < N[j]; k++) {
		for (int l = 0; l < N[i]; l++) {
			CalculateElement(A, B, i, j, l, k, DoB);
		}
	}

	return;
}

void FillSystemMatrices(matrix<double>* LeftMat, matrix<double>* RightMat, int FromRow, int ToRow, int FromCol, int ToCol)
{
	int i = 0;
	int j;
	for (int l = FromCol; l < ToCol; l++) {
		j = 0;
		for (int k = FromRow; k < ToRow; k++) {
			SystemLeftMatrix[l][k] = (*LeftMat)(j,i);
			SystemRightMatrix[l][k] = (*RightMat)(j,i);
			j++;
		}
		i++;
	}

	return;
}

void Construct_system_matrices()
{
	int i, j, n; 
	int Start_i, Start_j;
	int End_i = 0;
	int End_j = 0; 
	matrix<double>* TmpA;
	matrix<double>* TmpB;
	identity_matrix<double>* TmpI;

// ----------------------------------------------------------

	i = 0;

	Start_i = End_i;
	End_i += N[i];

	// i=0,j=0:
	j = 0;

	Start_j = End_j;
	End_j += N[j];

	TmpA = new matrix<double>(N[j],N[i]);
	TmpB = new matrix<double>(N[j],N[i]);
	TmpI = new identity_matrix<double>(N[0]);

	CalculateBlock(TmpA, TmpB, i, j);

	(*TmpA) = (*TmpA) - 0.5*(*TmpI);
	FillSystemMatrices(TmpB, TmpA, Start_j, End_j, Start_i, End_i);
	delete TmpA;
	delete TmpB;
	delete TmpI;


	// i=0,j=1:
	j = 1;

	Start_j = End_j;
	End_j += N[j];

	TmpA = new matrix<double>(N[j],N[i]);
	TmpB = new matrix<double>(N[j],N[i]);

	CalculateBlock(TmpA, TmpB, i, j);

	FillSystemMatrices(TmpB, TmpA, Start_j, End_j, Start_i, End_i);	
	delete TmpA;
	delete TmpB;


	// i=0,j=2:
	j = 2;

	Start_j = End_j;
	End_j += N[j];

	TmpA = new matrix<double>(N[j],N[i]);
	TmpB = new matrix<double>(N[j],N[i]);

	CalculateBlock(TmpA, TmpB, i, j);

	FillSystemMatrices(TmpB, TmpA, Start_j, End_j, Start_i, End_i);
	delete TmpA;
	delete TmpB;


	// i=0,j=3:
	j = 3;

	Start_j = End_j;
	End_j += N[j];

	TmpA = new matrix<double>(N[j],N[i]);
	TmpB = new matrix<double>(N[j],N[i]);

	CalculateBlock(TmpA, TmpB, i, j);

	FillSystemMatrices(TmpB, TmpA, Start_j, End_j, Start_i, End_i);
	delete TmpA;
	delete TmpB;

// ---------------------------------------------------------

	End_j = 0;

	i = 1;

	Start_i = End_i;
	End_i += N[i];

	// i=1,j=0:
	j = 0;

	Start_j = End_j;
	End_j += N[j];

	TmpA = new matrix<double>(N[j],N[i]);
	TmpB = new matrix<double>(N[j],N[i]);

	CalculateBlock(TmpA, TmpB, i, j);

	(*TmpA) = - (*TmpA);
	(*TmpB) = - (*TmpB);
	FillSystemMatrices(TmpA, TmpB, Start_j, End_j, Start_i, End_i);
	delete TmpA;
	delete TmpB;


	// i=1,j=1:
	j = 1;

	Start_j = End_j;
	End_j += N[j];

	TmpA = new matrix<double>(N[j],N[i]);
	TmpB = new matrix<double>(N[j],N[i]);
	TmpI = new identity_matrix<double>(N[1]);

	CalculateBlock(TmpA, TmpB, i, j);

	(*TmpA) = - (*TmpA) + 0.5*(*TmpI);
	(*TmpB) = - (*TmpB);
	FillSystemMatrices(TmpA, TmpB, Start_j, End_j, Start_i, End_i);
	delete TmpA;
	delete TmpB;
	delete TmpI;


	// i=1,j=2:
	j = 2;

	Start_j = End_j;
	End_j += N[j];

	TmpA = new matrix<double>(N[j],N[i]);
	TmpB = new matrix<double>(N[j],N[i]);

	CalculateBlock(TmpA, TmpB, i, j);

	(*TmpA) = - (*TmpA);
	(*TmpB) = - (*TmpB);
	FillSystemMatrices(TmpA, TmpB, Start_j, End_j, Start_i, End_i);
	delete TmpA;
	delete TmpB;


	// i=1,j=3:
	j = 3;

	Start_j = End_j;
	End_j += N[j];

	TmpA = new matrix<double>(N[j],N[i]);
	TmpB = new matrix<double>(N[j],N[i]);

	CalculateBlock(TmpA, TmpB, i, j);

	(*TmpA) = - (*TmpA);
	(*TmpB) = - (*TmpB);
	FillSystemMatrices(TmpA, TmpB, Start_j, End_j, Start_i, End_i);
	delete TmpA;
	delete TmpB;

// -----------------------------------------------------------

	End_j = 0;

	i = 2;

	Start_i = End_i;
	End_i += N[i];

	// i=2,j=0:
	j = 0;

	Start_j = End_j;
	End_j += N[j];

	TmpA = new matrix<double>(N[j],N[i]);
	TmpB = new matrix<double>(N[j],N[i]);

	if (!Testing)
		CalculateBlock(TmpA, TmpB, i, j, false);
	else 
		CalculateBlock(TmpA, TmpB, i, j);

	(*TmpA) = - (*TmpA);
	(*TmpB) = - (*TmpB);
	FillSystemMatrices(TmpA, TmpB, Start_j, End_j, Start_i, End_i);
	delete TmpA;
	delete TmpB;


	// i=2,j=1:
	j = 1;

	Start_j = End_j;
	End_j += N[j];

	TmpA = new matrix<double>(N[j],N[i]);
	TmpB = new matrix<double>(N[j],N[i]);

	if (!Testing)
		CalculateBlock(TmpA, TmpB, i, j, false);
	else 
		CalculateBlock(TmpA, TmpB, i, j);

	(*TmpA) = - (*TmpA);
	(*TmpB) = - (*TmpB);
	FillSystemMatrices(TmpA, TmpB, Start_j, End_j, Start_i, End_i);
	delete TmpA;
	delete TmpB;


	// i=2,j=2:
	j = 2;

	Start_j = End_j;
	End_j += N[j];

	TmpA = new matrix<double>(N[j],N[i]);
	TmpB = new matrix<double>(N[j],N[i]);
	TmpI = new identity_matrix<double>(N[2]);

	if (!Testing)
		CalculateBlock(TmpA, TmpB, i, j, false);
	else 
		CalculateBlock(TmpA, TmpB, i, j);

	(*TmpA) = - (*TmpA) + 0.5*(*TmpI);
	(*TmpB) = - (*TmpB);
	FillSystemMatrices(TmpA, TmpB, Start_j, End_j, Start_i, End_i);
	delete TmpA;
	delete TmpB;
	delete TmpI;


	// i=2,j=3:
	j = 3;

	Start_j = End_j;
	End_j += N[j];

	TmpA = new matrix<double>(N[j],N[i]);
	TmpB = new matrix<double>(N[j],N[i]);

	if (!Testing)
		CalculateBlock(TmpA, TmpB, i, j, false);
	else 
		CalculateBlock(TmpA, TmpB, i, j);

	(*TmpA) = - (*TmpA);
	(*TmpB) = - (*TmpB);
	FillSystemMatrices(TmpA, TmpB, Start_j, End_j, Start_i, End_i);
	delete TmpA;
	delete TmpB;

// --------------------------------------------------------------

	End_j = 0;

	i = 3;

	Start_i = End_i;
	End_i += N[i];

	// i=3,j=0:
	j = 0;

	Start_j = End_j;
	End_j += N[j];

	TmpA = new matrix<double>(N[j],N[i]);
	TmpB = new matrix<double>(N[j],N[i]);

	if (!Testing)
		CalculateBlock(TmpA, TmpB, i, j, false);
	else 
		CalculateBlock(TmpA, TmpB, i, j);

	(*TmpA) = - (*TmpA);
	(*TmpB) = - (*TmpB);
	FillSystemMatrices(TmpA, TmpB, Start_j, End_j, Start_i, End_i);
	delete TmpA;
	delete TmpB;


	// i=3,j=1:
	j = 1;

	Start_j = End_j;
	End_j += N[j];

	TmpA = new matrix<double>(N[j],N[i]);
	TmpB = new matrix<double>(N[j],N[i]);

	if (!Testing)
		CalculateBlock(TmpA, TmpB, i, j, false);
	else 
		CalculateBlock(TmpA, TmpB, i, j);

	(*TmpA) = - (*TmpA);
	(*TmpB) = - (*TmpB);
	FillSystemMatrices(TmpA, TmpB, Start_j, End_j, Start_i, End_i);
	delete TmpA;
	delete TmpB;


	// i=3,j=2:
	j = 2;

	Start_j = End_j;
	End_j += N[j];

	TmpA = new matrix<double>(N[j],N[i]);
	TmpB = new matrix<double>(N[j],N[i]);

	if (!Testing)
		CalculateBlock(TmpA, TmpB, i, j, false);
	else 
		CalculateBlock(TmpA, TmpB, i, j);

	(*TmpA) = - (*TmpA);
	(*TmpB) = - (*TmpB);
	FillSystemMatrices(TmpA, TmpB, Start_j, End_j, Start_i, End_i);
	delete TmpA;
	delete TmpB;


	// i=3,j=3:
	j = 3;

	Start_j = End_j;
	End_j += N[j];

	TmpA = new matrix<double>(N[j],N[i]);
	TmpB = new matrix<double>(N[j],N[i]);
	TmpI = new identity_matrix<double>(N[3]);

	if (!Testing)
		CalculateBlock(TmpA, TmpB, i, j, false);
	else 
		CalculateBlock(TmpA, TmpB, i, j);

	(*TmpA) = - (*TmpA) + 0.5*(*TmpI);
	(*TmpB) = - (*TmpB);
	FillSystemMatrices(TmpA, TmpB, Start_j, End_j, Start_i, End_i);
	delete TmpA;
	delete TmpB;
	delete TmpI;


	return;
}

void Store_result(const state_type2& eta_phi, const double t)
{
	int IndexT = Translate_t(t);

	for (int i = 0; i < NUM_POINTS_0; i++) {
		Results[IndexT][i] = eta_phi[i];
	}

	return;
}

void Construct_dsf(state_type1* dsphi_0)
{
	double Ds = 1.0/NUM_POINTS_0;
	double px[3];
	double py[3];

	for (int l = 0; l < NUM_POINTS_0; l++) {
		double s_l = Ds*(l + 1.0/2.0);
		if (l == 0) {
			px[0] = s_l;
			py[0] = phi[l];
			px[1] = s_l + Ds;
			py[1] = phi[l+1];
			px[2] = s_l + 2.0*Ds;
			py[2] = phi[l+2];
		} else if (l == NUM_POINTS_0 - 1) {
			px[0] = s_l - 2.0*Ds;
			py[0] = phi[l-2];
			px[1] = s_l - Ds;
			py[1] = phi[l-1];
			px[2] = s_l;
			py[2] = phi[l];
		} else {
			px[0] = s_l - Ds;
			py[0] = phi[l-1];
			px[1] = s_l;
			py[1] = phi[l];
			px[2] = s_l + Ds;
			py[2] = phi[l+1];
		}
		PolyInterpol2D Poly_dsphi(px,py);
		(*dsphi_0)[l] = Poly_dsphi.DFunc(s_l);
	}

	return;
}

void Construct_dxf(state_type1* eta_x, state_type1* eta_xx)
{
	double Ds = L/NUM_POINTS_0;
	double px[3];
	double py[3];

	for (int l = 0; l < NUM_POINTS_0; l++) {
		double s_l = Ds*(l + 1.0/2.0);
		if (l == 0) {
			px[0] = s_l;
			py[0] = eta[l];
			px[1] = s_l + Ds;
			py[1] = eta[l+1];
			px[2] = s_l + 2.0*Ds;
			py[2] = eta[l+2];
		} else if (l == NUM_POINTS_0 - 1) {
			px[0] = s_l - 2.0*Ds;
			py[0] = eta[l-2];
			px[1] = s_l - Ds;
			py[1] = eta[l-1];
			px[2] = s_l;
			py[2] = eta[l];
		} else {
			px[0] = s_l - Ds;
			py[0] = eta[l-1];
			px[1] = s_l;
			py[1] = eta[l];
			px[2] = s_l + Ds;
			py[2] = eta[l+1];
		}
		PolyInterpol2D Poly_eta_x(px,py);
		(*eta_x)[l] = Poly_eta_x.DFunc(s_l);
		(*eta_xx)[l] = Poly_eta_x.DDFunc(s_l);
	}

	return;
}

void Construct_phi_x_phi_z(state_type1* phi_x, state_type1* phi_z, state_type1* dnphi_0, state_type1* dsphi_0)
{
	double Ds = 1.0/NUM_POINTS_0;
	double px[3];
	double pz[3];
	
	for (int l = 0; l < NUM_POINTS_0; l++) {
		double s_l = Ds*(l + 1.0/2.0);
		if (l == 0) {
			px[0] = s_l;
			px[1] = s_l + Ds;
			px[2] = s_l + 2.0*Ds;
		} else if (l == NUM_POINTS_0 - 1) {
			px[0] = s_l - 2.0*Ds;
			px[1] = s_l - Ds;
			px[2] = s_l;
		} else {
			px[0] = s_l - Ds;
			px[1] = s_l;
			px[2] = s_l + Ds;
		}
		pz[0] = z(px[0],0);
		pz[1] = z(px[1],0);
		pz[2] = z(px[2],0);

		PolyInterpol2D Poly_z(px,pz);
		double x_s = dsx(s_l,0);
		double z_s = Poly_z.DFunc(s_l);

		double gamma = sqrt(pow(x_s, 2.0) + pow(z_s, 2.0));

		(*phi_x)[l] = (z_s*gamma*(*dnphi_0)[l] + x_s*(*dsphi_0)[l])/(pow(x_s, 2.0) + pow(z_s, 2.0));
		(*phi_z)[l] = (-x_s*gamma*(*dnphi_0)[l] + z_s*(*dsphi_0)[l])/(pow(x_s, 2.0) + pow(z_s, 2.0));
	}

	return;
}

void Construct_dxxf(state_type1* eta_xx)
{
	double Dx = L/NUM_POINTS_0;
	double px[3];
	double py[3];

	for (int l = 0; l < NUM_POINTS_0; l++) {
		if (l == 0) {
			py[0] = eta[l];
			py[1] = eta[l+1];
			py[2] = eta[l+2];
		} else if (l == NUM_POINTS_0 - 1) {
			py[0] = eta[l-2];
			py[1] = eta[l-1];
			py[2] = eta[l];
		} else {
			py[0] = eta[l-1];
			py[1] = eta[l];
			py[2] = eta[l+1];
		}
		(*eta_xx)[l] = (py[2] - 2.0*py[1] + py[0])/pow(Dx,2.0);
	}

	return;
}

void System(const state_type2& eta_phi, state_type2& dFdt, const double t)
{
	// Split eta and phi.
	#pragma omp parallel for
	for (int i = 0; i < NUM_POINTS_0; i++) {
		// Split:
		eta[i] = eta_phi[i];
		phi[i] = eta_phi[NUM_POINTS_0 + i];
	}

	// Construct system matrices
	Construct_system_matrices();

	// Create phi_star:
	double phi_star[Tot_N];
	double z_val;
	double s_l;
	double Ds = 1.0/NUM_POINTS_1;
	for (int i = 0; i < Tot_N; i++) {
		if (i < N[0]) {
			phi_star[i] = phi[NUM_POINTS_0-1-i];
		} else if (i >= N[0] && i < N[0]+N[1]) {
			s_l = Ds*(i - N[0] + 1.0/2.0);
			z_val = z(s_l,1);
			phi_star[i] = -InflowX(z_val,t);
		} else if (i >= N[0]+N[1] && i < N[0]+N[1]+N[2]) {
			phi_star[i] = 0.0;
		} else if (i >= N[0]+N[1]+N[2]) {
			phi_star[i] = 0.0;
		}
	}	

	// Matrix multiplication between system-right-matrix and phi_star:
	double* SystemRightMatrix_Array = (double*) SystemRightMatrix;
	double phi_star_result[Tot_N];
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Tot_N, 1, Tot_N, 1, SystemRightMatrix_Array, Tot_N, phi_star, Tot_N, 0, phi_star_result, Tot_N);


	//GMRES:

	/*double* SystemLeftMatrix_Array = (double*) SystemLeftMatrix;

	const int buffer = 128;
	MKL_INT ipar[buffer];
	int ipar14 = min(150, Tot_N);
	double dpar[buffer], tmp[(2*ipar14 + 1)*Tot_N + ipar14*(ipar14 + 9)/2 + 1];
	double b[Tot_N];
	double computed_solution[Tot_N];
	double residual[Tot_N];

	MKL_INT itercount;
	MKL_INT RCI_request, i, ivar;
	double dvar;

	ivar = Tot_N;

	for (i = 0; i < Tot_N; i++) {
		computed_solution[i] = 1.0;
		b[i] = phi_star_result[i];
	}

	dfgmres_init(&ivar, computed_solution, phi_star_result, &RCI_request, ipar, dpar, tmp);
	ipar[8]=0;
	ipar[9]=1;
	ipar[11]=1;
	dpar[0]=1.0E-3;
	dfgmres_check(&ivar, computed_solution, phi_star_result, &RCI_request, ipar, dpar, tmp);
	if (RCI_request != 0) {
		cout << "WARNING! GMRES FAILED!" << endl;
		return;
	}

ONE:
	dfgmres(&ivar, computed_solution, phi_star_result, &RCI_request, ipar, dpar, tmp);

	if (RCI_request == 0) goto COMPLETE;

	if (RCI_request == 1)
	{
		//cblas_dgemv(CblasColMajor, CblasNoTrans, Tot_N, Tot_N, 1, SystemLeftMatrix_Array, Tot_N, &tmp[ipar[21]-1], 1, 0, &tmp[ipar[22]-1], 1);
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Tot_N, 1, Tot_N, 1, SystemLeftMatrix_Array, Tot_N, &tmp[ipar[21]-1], Tot_N, 0, &tmp[ipar[22]-1], Tot_N);
		goto ONE;
	}

	if (RCI_request == 2)
	{
		ipar[12]=1;

		dfgmres_get(&ivar, computed_solution, phi_star_result, &RCI_request, ipar, dpar, tmp, &itercount);

		//cblas_dgemv(CblasColMajor, CblasNoTrans, Tot_N, Tot_N, 1, SystemLeftMatrix_Array, Tot_N, phi_star_result, 1, 0, residual, 1);
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Tot_N, 1, Tot_N, 1, SystemLeftMatrix_Array, Tot_N, phi_star_result, Tot_N, 0, residual, Tot_N);
		dvar=-1.0E0;
		i=1;
		daxpy(&ivar, &dvar, b, &i, residual, &i);
		dvar=dnrm2(&ivar,residual,&i);
		if (dvar<1.0E-3) goto COMPLETE;
		else goto ONE;
	}

	else {
		cout << "Failed with RCI_request = " << RCI_request << endl;
		return;
	}

COMPLETE:
	ipar[12] = 0;
	dfgmres_get(&ivar, computed_solution, phi_star_result, &RCI_request, ipar, dpar, tmp, &itercount);

	for (i = 0; i < Tot_N; i++) {
		phi_star_result[i] = computed_solution[i];
	}*/




	// LU-factorization:

	// Solve linear system (The solution will be contained in phi_star_result):
	double* SystemLeftMatrix_Array = (double*) SystemLeftMatrix;
	int ipiv[Tot_N];
	LAPACKE_dgetrf(LAPACK_COL_MAJOR, Tot_N, Tot_N, SystemLeftMatrix_Array, Tot_N, ipiv);		// SystemLeftMatrix_Array is now the PLU-factorisation of SystemLeftMatrix
	LAPACKE_dgetrs(LAPACK_COL_MAJOR, 'N', Tot_N, 1, SystemLeftMatrix_Array, Tot_N, ipiv, phi_star_result, Tot_N);

	int IndexT = Translate_t(t);
	// Store boundary data:
	#pragma omp parallel for
	for (int i = 0; i < Tot_N; i++) {
		BoundaryValues[IndexT][i] = phi_star[i];
		BoundaryValues[IndexT][Tot_N + i] = phi_star_result[i];
	}


	// Make arrays for needed vectors:
	state_type1 *phi_x, *phi_xx, *phi_z, *eta_x, *eta_xx, *dnphi_0, *dsphi_0;
	phi_x = new state_type1;
	phi_xx = new state_type1;
	phi_z = new state_type1;
	eta_x = new state_type1;
	eta_xx = new state_type1;
	dnphi_0 = new state_type1;
	dsphi_0 = new state_type1;

	#pragma omp parallel for
	for (int i = 0; i < NUM_POINTS_0; i++) {
		(*dnphi_0)[i] = phi_star_result[NUM_POINTS_0-1-i]; 		// Reversed for same reason as before
	}

	// Construct the arrays to get the missing information:
	Construct_dsf(dsphi_0);
	Construct_dxf(eta_x, eta_xx);
	Construct_dxxf(eta_xx);
	Construct_phi_x_phi_z(phi_x, phi_z, dnphi_0, dsphi_0);

	bool Stable = true;

	// Set differentiated values:
	double epsilon_eta = 0.25;
	#pragma omp parallel for
	for (int i = 0; i < NUM_POINTS_0; i++) {
		dFdt[i] = (*phi_z)[i] - (*eta_x)[i]*(*phi_x)[i];
		dFdt[NUM_POINTS_0 + i] = -(1.0/2.0)*(pow((*phi_x)[i],2.0) + pow((*phi_z)[i],2.0)) - G*eta[i];
		if (fabs(dFdt[i]) > StableThreshold or fabs(dFdt[NUM_POINTS_0 + i]) > StableThreshold) {
			Stable = false;
		}
	}

	for (int i = 0; i < NUM_POINTS_0; i++) {
		Results_Phi_z[IndexT][i] = (*phi_z)[i];
		Results_Phi_x[IndexT][i] = (*phi_x)[i];
		Results_Eta_x[IndexT][i] = (*eta_x)[i];
	}
	EdgeValues[IndexT][0] = eta[0];
	EdgeValues[IndexT][1] = eta[NUM_POINTS_0-1];


	if (!Stable)
		cout << "Warning! Values above stable threshold!" << endl;

	if (counter % 30 == 0) {
		CurrentCheck = high_resolution_clock::now();
		TimeDuration = duration_cast<milliseconds>(CurrentCheck - LastCheck);
		cout << "t = " << t << ": doing 30 calls took " << TimeDuration.count() << " milliseconds." << endl;
		LastCheck = CurrentCheck;
	}

	delete phi_x;
	delete phi_xx;
	delete phi_z;
	delete eta_x;
	delete eta_xx;
	delete dnphi_0;
	delete dsphi_0;

	counter++;
	
	return;
}

void Solve()
{
	Testing = false;

	cout << "The solver is now starting with" << endl
	<< "N_0: " << NUM_POINTS_0 << endl
	<< "N_1: " << NUM_POINTS_1 << endl
	<< "N_2: " << NUM_POINTS_2 << endl
	<< "N_3: " << NUM_POINTS_3 << endl
	<< "L: " << L << endl
	<< "N_t: " << NUM_POINTS_T << endl
	<< "dt: " << DT << endl
	<< "t_final: " << T_FINAL << endl;

	state_type2 eta_phi = Set_initial_conditions();

	integrate_const(Stepper, System, eta_phi, T_START, T_FINAL, DT, Store_result);

	return;
}

double LaplaceSolution(double s, int i)
{
	double x_val = x(s,i);
	double z_val = z(s,i);
	double dsx_val = dsx(s,i);
	double dsz_val = dsz(s,i);

	return cos(x_val)*(exp(-z_val) + exp(z_val));
}

double DxLaplaceSolution(double s, int i)
{
	double x_val = x(s,i);
	double z_val = z(s,i);
	double dsx_val = dsx(s,i);
	double dsz_val = dsz(s,i);

	return -sin(x_val)*(exp(-z_val) + exp(z_val));
}

double DzLaplaceSolution(double s, int i)
{
	double x_val = x(s,i);
	double z_val = z(s,i);
	double dsx_val = dsx(s,i);
	double dsz_val = dsz(s,i);

	return cos(x_val)*(-exp(-z_val) + exp(z_val));
}

double DnLaplaceSolution(double s, int i)
{
	double x_val = x(s,i);
	double z_val = z(s,i);
	double dsx_val = dsx(s,i);
	double dsz_val = dsz(s,i);
	double phi_x = DxLaplaceSolution(s, i);
	double phi_z = DzLaplaceSolution(s, i);
	double gamma = sqrt(pow(dsx_val, 2.0) + pow(dsz_val, 2.0));

	return (phi_x*dsz_val - phi_z*dsx_val)/gamma;
}

void FillArray(double* a, int i, bool NormalDerivative)
{
	double s_n;
	double Ds = 1.0/N[i];
	for (int n = 0; n < N[i]; n++) {
		s_n = Ds*(n + 1.0/2.0);
		if (NormalDerivative)
			a[n] = DnLaplaceSolution(s_n, i);
		else
			a[n] = LaplaceSolution(s_n, i);
	}
}

void Test_System()
{
	Testing = true;

	// Set eta = 0 for simplicity:
	#pragma omp parallel for
	for (int n = 0; n < NUM_POINTS_0; n++) {
		double x_val = L*(double)n/((double)NUM_POINTS_0-1.0);
		eta[n] = 0.1*sin(x_val*(1 + exp(-4*pow(x_val, 2))) + 1);//eta[n] = 0.3*sin((double)n*L/(1.0*(double)NUM_POINTS_0-1.0));//sin((double)n*L/((double)NUM_POINTS_0-1.0));//0.2*exp(-5*pow(n*L/(NUM_POINTS_0-1) - L/2,2));
	} 

	// Set known values (phi_0, dnphi_1, dnphi_2, dnphi_3):
	double phi_0[NUM_POINTS_0], dnphi_1[NUM_POINTS_1], dnphi_2[NUM_POINTS_2], dnphi_3[NUM_POINTS_3];
	FillArray(phi_0, 0, false);
	FillArray(dnphi_1, 1, true);
	FillArray(dnphi_2, 2, true);
	FillArray(dnphi_3, 3, true);

	// Set unknown values (dnphi_0, phi_1, phi_2, phi_3):
	double dnphi_0[NUM_POINTS_0], phi_1[NUM_POINTS_1], phi_2[NUM_POINTS_2], phi_3[NUM_POINTS_3];
	FillArray(dnphi_0, 0, true);
	FillArray(phi_1, 1, false);
	FillArray(phi_2, 2, false);
	FillArray(phi_3, 3, false);

	// Solve system:
	Construct_system_matrices();

	// Create phi_star and real solution:
	double phi_star[Tot_N];
	double Real_solution[Tot_N];
	int m = 0;
	for (int n = 0; n < Tot_N; n++) {
		if (n < N[0]) {
			m = n;
			phi_star[n] = phi_0[m];
			Real_solution[n] = dnphi_0[m];
		} else if (n >= N[0] && n < N[0]+N[1]) {
			m = n - N[0];
			phi_star[n] = dnphi_1[m];
			Real_solution[n] = phi_1[m];
		} else if (n >= N[0]+N[1] && n < N[0]+N[1]+N[2]) {
			m = n - N[0] - N[1];
			phi_star[n] = dnphi_2[m];
			Real_solution[n] = phi_2[m];
		} else if (n >= N[0]+N[1]+N[2]) {
			m = n - N[0] - N[1] - N[2];
			phi_star[n] = dnphi_3[m];
			Real_solution[n] = phi_3[m];
		}
	}

	// Matrix multiplication between system-right-matrix and phi_star:
	double* SystemRightMatrix_Array = (double*) SystemRightMatrix;
	double phi_star_result[Tot_N];
	cblas_dgemv(CblasColMajor, CblasNoTrans, Tot_N, Tot_N, 1, SystemRightMatrix_Array, Tot_N, phi_star, 1, 0, phi_star_result, 1);

	// Solve linear system (The solution will be contained in phi_star_result):
	double* SystemLeftMatrix_Array = (double*) SystemLeftMatrix;
	int ipiv[Tot_N];
	int SuccessLU = LAPACKE_dgetrf(LAPACK_COL_MAJOR, Tot_N, Tot_N, SystemLeftMatrix_Array, Tot_N, ipiv);		// SystemLeftMatrix_Array is now the PLU-factorisation of SystemLeftMatrix
	int SuccessSolve = LAPACKE_dgetrs(LAPACK_COL_MAJOR, 'N', Tot_N, 1, SystemLeftMatrix_Array, Tot_N, ipiv, phi_star_result, Tot_N);

	cout << "The solver was completed with exit code \n" 
	<< "LU: " << SuccessLU << endl
	<< "Solve: " << SuccessSolve << endl;

	std::ofstream TestAnalyticalSolution;
	TestAnalyticalSolution.open("Data/TestAnalyticalSolution.csv");
	std::ofstream TestNumericalSolution;
	TestNumericalSolution.open("Data/TestNumericalSolution.csv");

	double ErrorArray[Tot_N];
	double MD = 0;
	int i = 0;
	double Ds;
	double s_n;
	double x_val, z_val;
	m = 0;
	for (int n = 0; n < Tot_N; n++) {
		TestAnalyticalSolution << Real_solution[n] << " ";
		TestNumericalSolution << phi_star_result[n] << " ";
		ErrorArray[n] = fabs(Real_solution[n] - phi_star_result[n]);
		MD += ErrorArray[n];

		if (n == 0) {
			i = 0;
			Ds = 1.0/N[i];
		} else if (n == N[0]) {
			i = 1;
			Ds = 1.0/N[i];
			m = 0;
		} else if (n == N[0] + N[1]) {
			i = 2;
			Ds = 1.0/N[i];
			m = 0;
		} else if (n == N[0] + N[1] + N[2]) {
			i = 3;
			Ds = 1.0/N[i];
			m = 0;
		}
		s_n = Ds*(m + 1.0/2.0);
		x_val = x(s_n, i);
		z_val = z(s_n, i);
		//cout << "At (x,z) = " << "(" << x_val << "," << z_val << ") we have error: " << ErrorArray[n] << endl;

		m++;
	}
	cout << "MD for solution:" << endl;
	MD = MD/Tot_N;

	cout << MD << endl;


	return;
}

void DumpVectorfield()
{
	// Use midpoint rule for all components except the Greens function when it's close to its singularity.

	// First compute phi inside the domain or possibly subdomain:
	// NB! Some points might be outside the domain since the bottom is not flat and the surface is dynamic. 
	// These points will be set to zero
	// The number of discrete points inside the domain need not necessarily reflect the number of discrete points on the boundary. 

	const double DX = fabs(x_end - x_start)/(NUM_POINTS_X + 1);
	const double DZ = fabs(z_end - z_start)/(NUM_POINTS_Z + 1);	

	for (double t = 0.0; t < T_FINAL; t += DT) {
		int IndexT = Translate_t(t);

		// Set eta
		#pragma omp parallel for
		for (int i = 0; i < NUM_POINTS_0; i++) {
			eta[i] = Results[IndexT][i];
		}

		double phi[NUM_POINTS_Z][NUM_POINTS_X];
		double phi_x[NUM_POINTS_Z][NUM_POINTS_X];
		double phi_z[NUM_POINTS_Z][NUM_POINTS_X];

		#pragma omp parallel for collapse(2)
		for (int x_i = 0; x_i < NUM_POINTS_X; x_i++) {
			for (int z_i = 0; z_i < NUM_POINTS_Z; z_i++) {
				double x_val = x_start + (x_i + 1)*DX;
				double z_val = z_start + (z_i + 1)*DZ;
				double phi_val = 0;
				int x_i_surface = (int)round(NUM_POINTS_0*x_val/L - 0.5);

				if (z_val + DZ < Results[IndexT][x_i_surface] and z_val - DZ > -h(x_val)) {
					// Compute the boundary integral:
					for (int i = 0; i < 4; i++) {
						double Ds = 1.0/N[i];
						for (int l = 0; l < N[i]; l++) {
							double s_l = Ds*(l + 1.0/2.0);
							double ml1 = x(s_l, i);
							double ml2 = z(s_l, i);
							double mk1 = x_val;
							double mk2 = z_val;
							double epsilon = Ds/2.0;

							double d = (double)sqrt(pow(ml1 - mk1, 2.0) + pow(ml2 - mk2, 2.0));
							
							double px[3];
							if (l == 0) {
								px[0] = s_l;
								px[1] = s_l + Ds;
								px[2] = s_l + 2.0*Ds;
							} else if (l == N[i] - 1) {
								px[0] = s_l - 2.0*Ds;
								px[1] = s_l - Ds;
								px[2] = s_l;
							} else {
								px[0] = s_l - Ds;
								px[1] = s_l;
								px[2] = s_l + Ds;
							}
							double py[3] = {z(px[0],i), z(px[1],i), z(px[2],i)};
							PolyInterpol2D* Poly_z = new PolyInterpol2D(px, py);

							double dsC1 = dsx(s_l,i);
							double dsC2 = Poly_z->DFunc(s_l);
							double VelLength = sqrt(pow(dsC1, 2.0) + pow(dsC2, 2.0));

							double I_dnG, I_G;
							if (d <= RHO) {
								if (i == 0 || l == 0 || l == N[i] - 1) {
									auto F_dnG = [&Poly_z, &mk1, &mk2, &i](double s) {return (dnGreensFunction(x(s,i), dsx(s,i), Poly_z->Func(s), Poly_z->DFunc(s), mk1, mk2));};
									auto F_G = [&Poly_z, &mk1, &mk2, &i](double s) {return (GreensFunction(x(s,i), Poly_z->Func(s), mk1, mk2));};
									I_dnG = VelLength*gauss<double, 10>::integrate(F_dnG, s_l - epsilon, s_l + epsilon);
									I_G = VelLength*gauss<double, 10>::integrate(F_G, s_l - epsilon, s_l + epsilon);
								} else {
									auto F_dnG = [&mk1, &mk2, &i](double s) {return (dnGreensFunction(x(s,i), dsx(s,i), z(s,i), dsz(s,i), mk1, mk2));};
									auto F_G = [&mk1, &mk2, &i](double s) {return (GreensFunction(x(s,i), z(s,i), mk1, mk2));};
									I_dnG = VelLength*gauss<double, 10>::integrate(F_dnG, s_l - epsilon, s_l + epsilon);
									I_G = VelLength*gauss<double, 10>::integrate(F_G, s_l - epsilon, s_l + epsilon);
								}

							} else {
								double Length = Ds;

								// Matrix A:
								I_dnG = dnGreensFunction(ml1, dsx(s_l,i), ml2, dsz(s_l,i), mk1, mk2)*Length*VelLength;
								// Matrix B:
								I_G = GreensFunction(ml1, ml2, mk1, mk2)*Length*VelLength; 
							}

							delete Poly_z;

							double phi_l;
							double dnphi_l;
							if (i == 0) {
								phi_l = BoundaryValues[IndexT][l];
								dnphi_l = BoundaryValues[IndexT][Tot_N + l];
							} else if (i == 1) {
								dnphi_l = BoundaryValues[IndexT][NUM_POINTS_0 + l];
								phi_l = BoundaryValues[IndexT][Tot_N + NUM_POINTS_0 + l];
							} else if (i == 2) {
								dnphi_l = BoundaryValues[IndexT][NUM_POINTS_0 + NUM_POINTS_1 + l];
								phi_l = BoundaryValues[IndexT][Tot_N + NUM_POINTS_0 + NUM_POINTS_1 + l];
							} else if (i == 3) {
								dnphi_l = BoundaryValues[IndexT][NUM_POINTS_0 + NUM_POINTS_1 + NUM_POINTS_2 + l];
								phi_l = BoundaryValues[IndexT][Tot_N + NUM_POINTS_0 + NUM_POINTS_1 + NUM_POINTS_2 + l];
							}

							phi_val += phi_l*I_dnG - I_G*dnphi_l;
						}
					}

					phi[z_i][x_i] = phi_val;
				} else {
					phi[z_i][x_i] = 0.0;
				}
			}
		}
		// Computes derivatives using first order centeral difference:
		// The derivatives will have its outermost points set to zero since the neighbooring points would be on the boundary.
		#pragma omp parallel for collapse(2)
		for (int x_i = 0; x_i < NUM_POINTS_X; x_i++) {
			for (int z_i = 0; z_i < NUM_POINTS_Z; z_i++) {
				if (x_i == 0 or x_i == NUM_POINTS_X-1 or z_i == 0 or z_i == NUM_POINTS_Z-1) {
					phi_x[z_i][x_i] = 0.0;
					phi_z[z_i][x_i] = 0.0;
				} else {
					double x_val = x_start + (x_i + 1)*DX;
					double z_val = z_start + (z_i + 1)*DZ;
					int x_i_surface = (int)round(NUM_POINTS_0*x_val/L - 0.5);

					if (phi[z_i + 1][x_i] != 0.0 and phi[z_i - 1][x_i] != 0.0 and phi[z_i][x_i + 1] != 0.0 and phi[z_i][x_i - 1] != 0.0) {
						phi_x[z_i][x_i] = (phi[z_i][x_i + 1] - phi[z_i][x_i - 1])/(2*DX);
						phi_z[z_i][x_i] = (phi[z_i + 1][x_i] - phi[z_i - 1][x_i])/(2*DZ);
					} else {
						phi_x[z_i][x_i] = 0.0;
						phi_z[z_i][x_i] = 0.0;
					}
				}
			}
		}
		// Dump the three matrices:
		// Each time step three new matrices will be created
		char FileNamePhi_x[40];
		char FileNamePhi_z[40];
		char FileNamePhi[40]; 
		sprintf(FileNamePhi_x, "Data/VectorField/Phi_x%d.csv", IndexT);
		sprintf(FileNamePhi_z, "Data/VectorField/Phi_z%d.csv", IndexT);
		sprintf(FileNamePhi, "Data/VectorField/Phi%d.csv", IndexT);

		std::ofstream Solution_Phi_x;
		std::ofstream Solution_Phi_z;
		std::ofstream Solution_Phi;

		Solution_Phi_x.open(FileNamePhi_x);
		Solution_Phi_z.open(FileNamePhi_z);
		Solution_Phi.open(FileNamePhi);

		cout << "Dumping for t = " << t << endl;
		for (int j = 0; j < NUM_POINTS_Z; j++){
			for (int i = 0; i < NUM_POINTS_X; i++) {
				Solution_Phi_x << phi_x[j][i] << " ";
				Solution_Phi_z << phi_z[j][i] << " ";
				Solution_Phi << phi[j][i] << " ";
			}
			Solution_Phi_x << endl;
			Solution_Phi_z << endl;
			Solution_Phi << endl;
		}

		Solution_Phi_x.close();
		Solution_Phi_z.close();
		Solution_Phi.close();
	}


	return;
}

int main(int argc, char** argv)
{	
	// Save the parameters:
	std::ofstream Parameters;
	Parameters.open("Data/Parameters.csv");
	Parameters << "Parameters used:" << endl
	<< "N_0: " << NUM_POINTS_0 << endl
	<< "N_1: " << NUM_POINTS_1 << endl
	<< "N_2: " << NUM_POINTS_2 << endl
	<< "N_3: " << NUM_POINTS_3 << endl
	<< "N_x: " << NUM_POINTS_X << endl
	<< "N_z: " << NUM_POINTS_Z << endl
	<< "H: " << H << endl
	<< "L: " << L << endl
	<< "x_start: " << x_start << endl
	<< "x_end: " << x_end << endl
	<< "z_start: " << z_start << endl
	<< "z_end: " << z_end << endl
	<< "N_t: " << NUM_POINTS_T << endl
	<< "t_final: " << T_FINAL << endl
	<< "Initial condition?: " << "No!" << endl;

	auto start = high_resolution_clock::now();
	//Test_System();
	Solve();
	auto stop = high_resolution_clock::now();

	auto duration = duration_cast<milliseconds>(stop - start);

	cout << "The solution was found in " << duration.count() << " milleseconds." << endl;

	cout << "The dumping process is now starting" << endl;

	start = high_resolution_clock::now();
	Dump();
	DumpVectorfield();
	stop = high_resolution_clock::now();

	duration = duration_cast<seconds>(stop - start);

	cout << "The dumping took " << duration.count() << " seconds." << endl;

	return 0;
}















































