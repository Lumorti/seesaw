CXX=g++
CXXFLAGS=-Wfatal-errors -O3
HDIR=${MSKHOME}/mosek/9.2/tools/platform/linux64x86/h
LIBDIR=${MSKHOME}/mosek/9.2/tools/platform/linux64x86/bin
LDLIBSMOSEK= -I$(HDIR) -L$(LIBDIR) -Wl,-rpath-link,$(LIBDIR) -Wl,-rpath=$(LIBDIR) -lmosek64 -lfusion64

all:
	$(CXX) $(CXXFLAGS) -o seesaw seesaw.cpp $(LDLIBSMOSEK)
