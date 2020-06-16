README
	
	This code simulates surface waves in a wave tank using the non-linear surface waves equation and boundary integral equations. 

	The code uses external library functions from BOOST and MKL and must be linked when compiling. The makefile contains an incomplete example for compiling on Linux operative systems. The code must also be compiled together with the "PolyInterpol2D" source code. 

	The simulation parameters are set inside the SurfaceWaves.cpp file. When the program is finished the simulation data is stored in the "Data" folder and can be plotted in any way the user prefers. Inside the "Data" folder is the "VectorField" folder that contains data for the vector field. 

