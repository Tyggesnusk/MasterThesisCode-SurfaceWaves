CPRO_PATH=
MKLROOT=${CPRO_PATH}/mkl
MKLFLAGS= -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl    -m64 -I${MKLROOT}/include
BOOSTFLAG= 

all: SurfaceWaves

SurfaceWaves: PolyInterpol2D.cpp SurfaceWaves.cpp 
	g++ -std=c++11 -fopenmp -O3 -o SurfaceWaves PolyInterpol2D.cpp SurfaceWaves.cpp $(MKLFLAGS) $(BOOSTFLAG)





