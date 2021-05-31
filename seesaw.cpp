#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <complex>
#include <iomanip>
#include <random>
#include <fstream>
#include <math.h>

// Eigen
#include <Eigen/Dense>

// MOSEK
#include "fusion.h"

// Because otherwise things start looking messy
using complex1 = std::vector<std::complex<double>>;
using complex2 = std::vector<complex1>;
using complex3 = std::vector<complex2>;
using complex4 = std::vector<complex3>;
using real1 = std::vector<double>;
using real2 = std::vector<real1>;
using real3 = std::vector<real2>;
using real4 = std::vector<real3>;

// Useful values
const double root2 = sqrt(2.0);
const std::complex<double> im = sqrt(std::complex<double>(-1.0));

// Seesaw iterations
const int numIters = 10;
const double tol = 1E-5;

// How much to output (0 == none, 1 == normal, 2 == extra)
int verbosity = 1;

// Whether to force the rank of each matrix
bool restrictRankA = false;
bool restrictRankB = false;

// Pretty print a generic 1D vector with length n 
template <typename type> void prettyPrint(std::string pre, std::vector<type> arr, int n){

	// Used fixed precision
	std::cout << std::fixed << std::setprecision(2);

	// For the first line, add the pre text
	std::cout << pre << " | ";

	// For the x values, combine them all on one line
	for (int x=0; x<n; x++){
		std::cout << std::setw(5) << arr[x] << " ";
	}

	// Output the row
	std::cout << "|" << std::endl;

}

// Pretty print a generic 2D array with width w and height h
template <typename type> void prettyPrint(std::string pre, std::vector<std::vector<type>> arr, int w, int h){

	// Used fixed precision
	std::cout << std::fixed << std::setprecision(2);

	// Loop over the array
	std::string rowText;
	for (int y=0; y<h; y++){

		// For the first line, add the pre text
		if (y == 0){
			rowText = pre;

		// Otherwise pad accordingly
		} else {
			rowText = "";
			while (rowText.length() < pre.length()){
				rowText += " ";
			}
		}

		// Spacing
		std::cout << rowText << " | ";

		// For the x values, combine them all on one line
		for (int x=0; x<w; x++){
			std::cout << std::setw(5) << arr[y][x] << " ";
		}

		// Output the row
		std::cout << "|" << std::endl;

	}

}

// Pretty print a complex 2D array with width w and height h
void prettyPrint(std::string pre, complex2 arr, int w, int h){

	// Used fixed precision
	std::cout << std::fixed << std::setprecision(2);

	// Loop over the array
	std::string rowText;
	for (int y=0; y<h; y++){

		// For the first line, add the pre text
		if (y == 0){
			rowText = pre;

		// Otherwise pad accordingly
		} else {
			rowText = "";
			while (rowText.length() < pre.length()){
				rowText += " ";
			}
		}

		// Spacing
		std::cout << rowText << " | ";

		// For the x values, combine them all on one line
		for (int x=0; x<w; x++){
			if (std::imag(arr[y][x]) >= 0){
				std::cout << std::setw(5) << std::real(arr[y][x]) << "+" << std::abs(std::imag(arr[y][x])) << "i ";
			} else {
				std::cout << std::setw(5) << std::real(arr[y][x]) << "-" << std::abs(std::imag(arr[y][x])) << "i ";
			}
		}

		// Output the row
		std::cout << "|" << std::endl;

	}

}

// Perform an approximated seesaw method to optimise just B TODO
void seesawApprox(int d, int n){

	// The inequality value to eventually return
	double finalResult = 0;

	int numMeasureB = n;
	int numOutcomeB = d;

	real4 Br(numMeasureB, real3(numOutcomeB, real2(d, real1(d))));
	real4 Bi(numMeasureB, real3(numOutcomeB, real2(d, real1(d))));

	auto dimRefB = monty::new_array_ptr(std::vector<int>({d, d}));

	// Randomise B
	std::random_device rd;
	std::mt19937 generator(rd());
	std::uniform_real_distribution<double> distribution(-1.0, 1.0);
	for (int y=0; y<numMeasureB; y++){
		for (int b=0; b<numOutcomeB-1; b++){
			for (int j=0; j<d; j++){
				for (int k=0; k<d; k++){

					// Create some random values
					Br[y][b][j][k] = distribution(generator);
					Br[y][b][k][j] = Br[y][b][j][k];

					// Imaginary only on the off-diagonals
					if (j != k){
						Bi[y][b][j][k] = distribution(generator);
						Bi[y][b][k][j] = -Bi[y][b][j][k];
					}

				}
			}
		}
	}

	// Ensure each measurement sums to the identity 
	for (int y=0; y<numMeasureB; y++){
		for (int b=0; b<numOutcomeB; b++){
			for (int j=0; j<d; j++){
				for (int k=0; k<d; k++){
					if (b < numOutcomeB-1){
						Br[y][numOutcomeB-1][j][k] -= Br[y][b][j][k];
						Bi[y][numOutcomeB-1][j][k] -= Bi[y][b][j][k];
					} else if (b == numOutcomeB - 1 && j == k){
						Br[y][numOutcomeB-1][j][k] += 1.0;
					}
				}
			}
		}

	}

	// Initial output
	for (int y=0; y<numMeasureB; y++){
		for (int b=0; b<numOutcomeB; b++){
			prettyPrint("before Br = ", Br[y][b], d, d);
			std::cout << std::endl;
			prettyPrint("before Bi = ", Bi[y][b], d, d);
			std::cout << std::endl;
		}
	}

	// Keep iterating
	for (int iter=0; iter<numIters; iter++){

		// For each pair of measurements
		for (int k=0; k<numMeasureB; k++){
			for (int l=k+1; l<numMeasureB; l++){

				// For each outcome
				for (int i=0; i<numOutcomeB; i++){

					// Precalculate the sum of the other measurements
					real2 otherBr(d, real1(d, 0));
					real2 otherBi(d, real1(d, 0));
					real2 otherBr2(d, real1(d, 0));
					real2 otherBi2(d, real1(d, 0));
					for (int j=0; j<numOutcomeB; j++){
						for (int a=0; a<d; a++){
							for (int b=0; b<d; b++){
								otherBr[a][b] += Br[l][j][a][b];
								otherBi[a][b] += Bi[l][j][a][b];
								otherBr2[a][b] += Br[k][j][a][b];
								otherBi2[a][b] += Bi[k][j][a][b];
							}
						}
					}

					// Create the references to these arrays
					auto otherBrRef = monty::new_array_ptr(otherBr);
					auto otherBiRef = monty::new_array_ptr(otherBi);
					auto otherBrRef2 = monty::new_array_ptr(otherBr2);
					auto otherBiRef2 = monty::new_array_ptr(otherBi2);

					// -----------------------
					//    optimising B_k^i
					// -----------------------

					// Create the MOSEK model 
					mosek::fusion::Model::t modelB = new mosek::fusion::Model(); 

					// The moment matrices to optimise
					mosek::fusion::Variable::t BrOpt = modelB->variable(dimRefB, mosek::fusion::Domain::inRange(-1.0, 1.0));
					mosek::fusion::Variable::t BiOpt = modelB->variable(dimRefB, mosek::fusion::Domain::inRange(-1.0, 1.0));

					// Constrain the combined matrix to be PSD
					modelB->constraint(mosek::fusion::Expr::vstack(
											mosek::fusion::Expr::hstack(
												BrOpt, 
												mosek::fusion::Expr::neg(BiOpt)
											), 
											mosek::fusion::Expr::hstack(
												BiOpt,
												BrOpt 
											)
									   ), mosek::fusion::Domain::inPSDCone(2*d));

					// Symmetry
					modelB->constraint(mosek::fusion::Expr::sub(BrOpt->index(0, 1), BrOpt->index(1, 0)), mosek::fusion::Domain::equalsTo(0.0));
					modelB->constraint(mosek::fusion::Expr::add(BiOpt->index(0, 1), BiOpt->index(1, 0)), mosek::fusion::Domain::equalsTo(0.0));

					// Must sum to identity
					modelB->constraint(mosek::fusion::Expr::add(BrOpt, otherBrRef2), mosek::fusion::Domain::equalsTo(mosek::fusion::Matrix::eye(d)));
					modelB->constraint(mosek::fusion::Expr::add(BiOpt, otherBiRef2), mosek::fusion::Domain::equalsTo(mosek::fusion::Matrix::sparse(d, d)));

					// Set up the objective function 
					modelB->objective(mosek::fusion::ObjectiveSense::Minimize, mosek::fusion::Expr::sub(mosek::fusion::Expr::dot(BrOpt, otherBrRef), mosek::fusion::Expr::dot(BiOpt, otherBiRef)));

					// Ensure the probability isn't imaginary
					modelB->constraint(mosek::fusion::Expr::add(mosek::fusion::Expr::dot(otherBrRef, BiOpt), mosek::fusion::Expr::dot(otherBiRef, BrOpt)), mosek::fusion::Domain::equalsTo(0.0));

					// Solve the SDP
					modelB->solve();
					
					// Extract the results
					finalResult = n - modelB->primalObjValue();
					int matHeightB = d;
					int matWidthB = d;
					auto tempBr = *(BrOpt->level());
					auto tempBi = *(BiOpt->level());
					for (int a=0; a<matWidthB*matHeightB; a++){
						Br[k][i][a/matWidthB][a%matWidthB] = tempBr[a];
						Bi[k][i][a/matWidthB][a%matWidthB] = tempBi[a];
					}

					// Destroy the model
					modelB->dispose();

					// Output after this section
					std::cout << std::fixed << std::setprecision(5) << "iter " << std::setw(3) << iter << " after B opt " << finalResult << std::endl;

					// Precalculate the sum of the other measurements
					otherBr2 = real2(d, real1(d, 0));
					otherBi2 = real2(d, real1(d, 0));
					for (int j=0; j<numOutcomeB; j++){
						for (int a=0; a<d; a++){
							for (int b=0; b<d; b++){
								otherBr2[a][b] += Br[k][j][a][b];
								otherBi2[a][b] += Bi[k][j][a][b];
							}
						}
					}

					// Create the references to these arrays
					otherBrRef2 = monty::new_array_ptr(otherBr2);
					otherBiRef2 = monty::new_array_ptr(otherBi2);

					// -----------------------
					//    optimising B_l^i
					// -----------------------
					
					// Create the MOSEK model 
					mosek::fusion::Model::t modelB2 = new mosek::fusion::Model(); 

					// The moment matrices to optimise
					mosek::fusion::Variable::t BrOpt2 = modelB2->variable(dimRefB, mosek::fusion::Domain::inRange(-1.0, 1.0));
					mosek::fusion::Variable::t BiOpt2 = modelB2->variable(dimRefB, mosek::fusion::Domain::inRange(-1.0, 1.0));

					// Constrain the combined matrix to be PSD
					modelB2->constraint(mosek::fusion::Expr::vstack(
											mosek::fusion::Expr::hstack(
												BrOpt2, 
												mosek::fusion::Expr::neg(BiOpt2)
											), 
											mosek::fusion::Expr::hstack(
												BiOpt2,
												BrOpt2 
											)
									   ), mosek::fusion::Domain::inPSDCone(2*d));

					// Must sum to identity
					modelB2->constraint(mosek::fusion::Expr::add(BrOpt2, otherBrRef), mosek::fusion::Domain::equalsTo(mosek::fusion::Matrix::eye(d)));
					modelB2->constraint(mosek::fusion::Expr::add(BiOpt2, otherBiRef), mosek::fusion::Domain::equalsTo(mosek::fusion::Matrix::sparse(d, d)));

					// Symmetry
					modelB2->constraint(mosek::fusion::Expr::sub(BrOpt2->index(0, 1), BrOpt2->index(1, 0)), mosek::fusion::Domain::equalsTo(0.0));
					modelB2->constraint(mosek::fusion::Expr::add(BiOpt2->index(0, 1), BiOpt2->index(1, 0)), mosek::fusion::Domain::equalsTo(0.0));

					// Set up the objective function 
					modelB2->objective(mosek::fusion::ObjectiveSense::Minimize, mosek::fusion::Expr::sub(mosek::fusion::Expr::dot(BrOpt2, otherBrRef2), mosek::fusion::Expr::dot(BiOpt2, otherBiRef2)));

					// Ensure the probability isn't imaginary
					modelB2->constraint(mosek::fusion::Expr::add(mosek::fusion::Expr::dot(otherBrRef2, BiOpt2), mosek::fusion::Expr::dot(otherBiRef2, BrOpt2)), mosek::fusion::Domain::equalsTo(0.0));

					// Solve the SDP
					modelB2->solve();

					// Extract the results
					finalResult = n - modelB2->primalObjValue();
					auto tempBr2 = *(BrOpt2->level());
					auto tempBi2 = *(BiOpt2->level());
					for (int a=0; a<d*d; a++){
						Br[l][i][a/d][a%d] = tempBr2[a];
						Bi[l][i][a/d][a%d] = tempBi2[a];
					}

					// Destroy the model
					modelB2->dispose();

					// Output after this section
					std::cout << std::fixed << std::setprecision(5) << "iter " << std::setw(3) << iter << " after B opt " << finalResult << std::endl;

				}

			}
		}

	}

	// Final output
	for (int y=0; y<numMeasureB; y++){
		for (int b=0; b<numOutcomeB; b++){
			prettyPrint("after Br = ", Br[y][b], d, d);
			std::cout << std::endl;
			prettyPrint("after Bi = ", Bi[y][b], d, d);
			std::cout << std::endl;
		}
	}

}

// Perform the seesaw method to optimise both A and B TODO
void seesawExtended(int d, int n){

	// The inequality value to eventually return
	double finalResult = 0;

	// How many permutations
	int numPerm = n*(n-1)/2;

	int numMeasureA = d*d*numPerm*numPerm;
	int numOutcomeA = 3;
	int numMeasureB = n;
	int numOutcomeB = d;

	// The arrays to store the operators (real and imaginary separately)
	real3 Ar(numMeasureA*numOutcomeA, real2(d, real1(d)));
	real3 Ai(numMeasureA*numOutcomeA, real2(d, real1(d)));
	real3 Br(numMeasureB*numOutcomeB, real2(d, real1(d)));
	real3 Bi(numMeasureB*numOutcomeB, real2(d, real1(d)));

	// Keep seesawing 
	double prevResult = -1;
	for (int iter=0; iter<numIters; iter++){

		// ----------------------------
		//    Fixing A, optimising B
		// ----------------------------

		// Create references to the fixed matrices
		std::vector<mosek::fusion::Expression::t> ArRef;
		std::vector<mosek::fusion::Expression::t> AiRef;
		for (int i=0; i<numMeasureA*numOutcomeA; i++){
			ArRef.push_back(mosek::fusion::Expr::constTerm(monty::new_array_ptr(Ar[i])));
			AiRef.push_back(mosek::fusion::Expr::constTerm(monty::new_array_ptr(Ai[i])));
		}

		// Create the MOSEK model 
		mosek::fusion::Model::t modelB = new mosek::fusion::Model(); 

		// The moment matrices to optimise
		mosek::fusion::Variable::t BrOpt = modelB->variable(d, numMeasureB*numOutcomeB, mosek::fusion::Domain::inPSDCone(d));
		mosek::fusion::Variable::t BiOpt = modelB->variable(d, numMeasureB*numOutcomeB, mosek::fusion::Domain::inPSDCone(d));

		// Set up the objective function 
		mosek::fusion::Expression::t obj;
		for (int i=0; i<n; i++){
			for (int j=i+1; j<n; j++){
				int m = j - i + ((i-1)*(2*n-i))/2;
				for (int x1=m; x1<(m+1)*d; x1++){
					for (int x2=m; x2<(m+1)*d; x2++){
						//obj = mosek::fusion::Expr::add(obj, 
							//mosek::fusion::Expr::dot(
								//mosek::fusion::Expr::sub(ArRef[x1*x2*numOutcomeA+0], ArRef[x1*x2*numOutcomeA+1]),
								//mosek::fusion::Expr::sub(BrOpt->index(i*numOutcomeB+x1), BrOpt->index(j*numOutcomeB+x2))
							//));
					}
				}
			}
		}
		modelB->objective(mosek::fusion::ObjectiveSense::Maximize, obj);

		// For each set of measurements, the matrices should sum to the identity
		for (int y=0; y<numMeasureB; y++){
			mosek::fusion::Expression::t total;
			for (int b=0; b<numOutcomeB; b++){
				total = mosek::fusion::Expr::add(total, BrOpt->index(y*numOutcomeB+b));
			}
			modelB->constraint(total, mosek::fusion::Domain::equalsTo(mosek::fusion::Matrix::eye(d)));
		}

		// Solve the SDP
		modelB->solve();
		
		// Extract the results
		finalResult = modelB->primalObjValue() / d;
		int matHeightB = d*d;
		int matWidthB = numMeasureB*numOutcomeB;
		auto tempBr = *(BrOpt->level());
		auto tempBi = *(BiOpt->level());
		//for (int i=0; i<matWidthB*matHeightB; i++){
			//Br[i/matWidthB][i%matWidthB] = tempBr[i];
			//Bi[i/matWidthB][i%matWidthB] = tempBi[i];
		//}

		// Destroy the model
		modelB->dispose();

		// Output after this section
		std::cout << std::fixed << std::setprecision(5) << "iter " << std::setw(3) << iter << " after B opt " << finalResult << std::endl;

	}

}

// Perform the seesaw method to optimise both A and B
void seesaw(int d, int n){

	// Number of measurements
	int numMeasureA = n;
	int numMeasureB = n*(n-1);

	// Number of possible outputs
	int numOutcomeA = 2;
	int numOutcomeB = 2;

	// The inequality value to maximise
	double finalResult = 0;

	// The coefficients defining the inequality
	real2 C(numOutcomeA*numMeasureA, real1(numOutcomeB*numMeasureB));

	// For each CHSH pair
	int y1 = 0;
	int y2 = 1;
	for (int x1=0; x1<numMeasureA; x1++){
		for (int x2=x1+1; x2<numMeasureA; x2++){

			// A_x1 B_y1
			for (int a=0; a<numOutcomeA; a++){
				for (int b=0; b<numOutcomeB; b++){
					if (a == b){
						C[x1*numOutcomeA+a][y1*numOutcomeB+b] = 1.0;
					} else {
						C[x1*numOutcomeA+a][y1*numOutcomeB+b] = -1.0;
					}

				}
			}
			
			// A_x2 B_y1
			for (int a=0; a<numOutcomeA; a++){
				for (int b=0; b<numOutcomeB; b++){
					if (a == b){
						C[x2*numOutcomeA+a][y1*numOutcomeB+b] = 1.0;
					} else {
						C[x2*numOutcomeA+a][y1*numOutcomeB+b] = -1.0;
					}
				}
			}
			
			// A_x1 B_y2
			for (int a=0; a<numOutcomeA; a++){
				for (int b=0; b<numOutcomeB; b++){
					if (a == b){
						C[x1*numOutcomeA+a][y2*numOutcomeB+b] = 1.0;
					} else {
						C[x1*numOutcomeA+a][y2*numOutcomeB+b] = -1.0;
					}
				}
			}
			
			// - A_x2 B_y2
			for (int a=0; a<numOutcomeA; a++){
				for (int b=0; b<numOutcomeB; b++){
					if (a == b){
						C[x2*numOutcomeA+a][y2*numOutcomeB+b] = -1.0;
					} else {
						C[x2*numOutcomeA+a][y2*numOutcomeB+b] = 1.0;
					}
				}
			}
			
			// Use a different pair of Bob's measurements
			y1 += 2;
			y2 += 2;

		}
	}

	// The arrays to store the operators (real and imaginary separately)
	real2 Ar(numOutcomeA*numMeasureA, real1(d*d));
	real2 Ai(numOutcomeA*numMeasureA, real1(d*d));
	real2 Br(d*d, real1(numOutcomeB*numMeasureB));
	real2 Bi(d*d, real1(numOutcomeB*numMeasureB));

	// Create an identity vector
	std::vector<double> identity(d*d, 0.0);
	for (int i=0; i<d; i++){
		identity[i*(d+1)] = 1.0;
	}

	// Randomise A
	std::random_device rd;
	std::mt19937 generator(rd());
	std::uniform_real_distribution<double> distribution(-1.0, 1.0);
	for (int x=0; x<numMeasureA; x++){
		for (int a=0; a<numOutcomeA-1; a++){
			for (int j=0; j<d; j++){
				for (int k=0; k<d; k++){

					// Create some random values
					Ar[x*numOutcomeA+a][j*d+k] = distribution(generator);

					// Imaginary only on the off-diagonals
					if (j != k){
						Ai[x*numOutcomeA+a][j*d+k] = distribution(generator);
					}

				}
			}
		}
	}

	// Which columns should be identical
	std::vector<std::vector<int>> matchingRows;
	for (int j=0; j<d; j++){
		for (int i=j+1; i<d; i++){
			int downLeft = i-j;
			matchingRows.push_back({j*d+i, (j+downLeft)*d+(i-downLeft)});
		}
	}

	// Force these to be identical
	for (int i=0; i<matchingRows.size(); i++){
		for (int j=0; j<numMeasureA*numOutcomeA; j++){
			Ar[j][matchingRows[i][0]] = Ar[j][matchingRows[i][1]];
			Ai[j][matchingRows[i][0]] = -Ai[j][matchingRows[i][1]];
		}
	}

	// Ensure each measurement sums to the identity 
	for (int x=0; x<numMeasureA; x++){
		for (int j=0; j<d*d; j++){
			for (int a=0; a<numOutcomeA-1; a++){
				Ar[x*numOutcomeA+numOutcomeA-1][j] -= Ar[x*numOutcomeA+a][j];
				Ai[x*numOutcomeA+numOutcomeA-1][j] -= Ai[x*numOutcomeA+a][j];
			}
			Ar[x*numOutcomeA+numOutcomeA-1][j] += identity[j];
		}
	}

	// Output the raw arrays
	if (verbosity >= 2){
		prettyPrint("C = ", C, numOutcomeB*numMeasureB, numOutcomeA*numMeasureA);
		std::cout << std::endl;
		prettyPrint("Ar =", Ar, d*d, numMeasureA*numOutcomeA);
		std::cout << std::endl;
		prettyPrint("Ai =", Ai, d*d, numMeasureA*numOutcomeA);
		std::cout << std::endl;
	}

	// The rank for A to try
	restrictRankA = true;
	restrictRankB = false;
	real2 rankA = real2(numOutcomeA*numMeasureA, real1(1, d/2.0));
	real2 rankB = real2(1, real1(numOutcomeB*numMeasureB, d/2.0));
	if (d == 5){
		rankA = {{2}, {3}, {2}, {3}, {3}, {2}};
	}
	prettyPrint("rank of A = ", rankA, 1, numOutcomeA*numMeasureA);

	// Create the arrays to be used to select the columns of the B array 
	std::vector<std::shared_ptr<monty::ndarray<int,1>>> columnsStartRefB;
	std::vector<std::shared_ptr<monty::ndarray<int,1>>> columnsEndRefB;
	for (int i=0; i<numMeasureB*numOutcomeB; i++){
		columnsStartRefB.push_back(monty::new_array_ptr(std::vector<int>({0, i})));
		columnsEndRefB.push_back(monty::new_array_ptr(std::vector<int>({d*d, i+1})));
	}

	// Create the arrays to be used to select the rows of the B array 
	std::vector<std::shared_ptr<monty::ndarray<int,1>>> rowsStartRefB;
	std::vector<std::shared_ptr<monty::ndarray<int,1>>> rowsEndRefB;
	for (int i=0; i<d*d; i++){
		rowsStartRefB.push_back(monty::new_array_ptr(std::vector<int>({i, 0})));
		rowsEndRefB.push_back(monty::new_array_ptr(std::vector<int>({i+1, numOutcomeB*numMeasureB})));
	}

	// Create the arrays to be used to select the columns of the A array 
	std::vector<std::shared_ptr<monty::ndarray<int,1>>> columnsStartRefA;
	std::vector<std::shared_ptr<monty::ndarray<int,1>>> columnsEndRefA;
	for (int i=0; i<d*d; i++){
		columnsStartRefA.push_back(monty::new_array_ptr(std::vector<int>({0, i})));
		columnsEndRefA.push_back(monty::new_array_ptr(std::vector<int>({numMeasureA*numOutcomeA, i+1})));
	}

	// Create the arrays to be used to select the rows of the A array 
	std::vector<std::shared_ptr<monty::ndarray<int,1>>> rowsStartRefA;
	std::vector<std::shared_ptr<monty::ndarray<int,1>>> rowsEndRefA;
	for (int i=0; i<numMeasureA*numOutcomeA; i++){
		rowsStartRefA.push_back(monty::new_array_ptr(std::vector<int>({i, 0})));
		rowsEndRefA.push_back(monty::new_array_ptr(std::vector<int>({i+1, d*d})));
	}

	// Create the collapsed identity
	auto identityRef = monty::new_array_ptr(identity);

	// Create a reference to an array of zeros
	auto zero1DRef = monty::new_array_ptr(real1(d*d, 0));
	auto zeroRefB = monty::new_array_ptr(real2(1, real1(numOutcomeB*numMeasureB, 0.0)));
	auto zeroRefA = monty::new_array_ptr(real2(numOutcomeA*numMeasureA, real1(1, 0.0)));
	auto zero2DRef = monty::new_array_ptr(real2(numOutcomeA*numMeasureA, real1(numOutcomeB*numMeasureB, 0)));

	// Create a reference to the desired rank array
	auto rankARef = monty::new_array_ptr(rankA);
	auto rankBRef = monty::new_array_ptr(rankB);

	// The sizes of the variable matrix
	auto dimRefA = monty::new_array_ptr(std::vector<int>({numOutcomeA*numMeasureA, d*d}));
	auto dimRefB = monty::new_array_ptr(std::vector<int>({d*d, numOutcomeB*numMeasureB}));

	// C is always fixed
	auto CRef = monty::new_array_ptr(C);

	// Keep seesawing 
	double prevResult = -1;
	for (int iter=0; iter<numIters; iter++){

		// ----------------------------
		//    Fixing A, optimising B
		// ----------------------------

		// Create references to the fixed matrices
		auto ArRef = monty::new_array_ptr(Ar);
		auto AiRef = monty::new_array_ptr(Ai);

		// Create the MOSEK model 
		mosek::fusion::Model::t modelB = new mosek::fusion::Model(); 

		// The moment matrices to optimise
		mosek::fusion::Variable::t BrOpt = modelB->variable(dimRefB, mosek::fusion::Domain::inRange(-1.0, 1.0));
		mosek::fusion::Variable::t BiOpt = modelB->variable(dimRefB, mosek::fusion::Domain::inRange(-1.0, 1.0));

		// Set up the objective function 
		modelB->objective(mosek::fusion::ObjectiveSense::Maximize, mosek::fusion::Expr::dot(CRef, mosek::fusion::Expr::sub(mosek::fusion::Expr::mul(ArRef, BrOpt), mosek::fusion::Expr::mul(AiRef, BiOpt))));

		// Force the trace of each matrix to be a certain value 
		if (restrictRankB){
			mosek::fusion::Expression::t sum = mosek::fusion::Expr::add(BrOpt->slice(rowsStartRefB[0], rowsEndRefB[0]), BrOpt->slice(rowsStartRefB[d+1], rowsEndRefB[d+1]));
			for (int i=2; i<d; i++){
				sum = mosek::fusion::Expr::add(sum, BrOpt->slice(rowsStartRefB[i*(d+1)], rowsEndRefB[i*(d+1)]));
			}
			modelB->constraint(sum, mosek::fusion::Domain::equalsTo(rankBRef));
		}

		// Ensure the probability isn't imaginary
		modelB->constraint(mosek::fusion::Expr::add(mosek::fusion::Expr::mul(ArRef, BiOpt), mosek::fusion::Expr::mul(AiRef, BrOpt)), mosek::fusion::Domain::equalsTo(zero2DRef));

		// For each set of measurements, the matrices should sum to the identity
		for (int i=0; i<columnsStartRefB.size(); i+=numOutcomeB){
			modelB->constraint(mosek::fusion::Expr::sum(BrOpt->slice(columnsStartRefB[i], columnsEndRefB[i+numOutcomeB-1]), 1), mosek::fusion::Domain::equalsTo(identityRef));
			modelB->constraint(mosek::fusion::Expr::sum(BiOpt->slice(columnsStartRefB[i], columnsEndRefB[i+numOutcomeB-1]), 1), mosek::fusion::Domain::equalsTo(zero1DRef));
		}

		// Each section of B should also be >= 0
		for (int i=0; i<columnsStartRefB.size(); i++){
			modelB->constraint(mosek::fusion::Expr::vstack(
									mosek::fusion::Expr::hstack(
										BrOpt->slice(columnsStartRefB[i],columnsEndRefB[i])->reshape(d, d), 
										mosek::fusion::Expr::neg(BiOpt->slice(columnsStartRefB[i],columnsEndRefB[i])->reshape(d, d))
									), 
									mosek::fusion::Expr::hstack(
										BiOpt->slice(columnsStartRefB[i],columnsEndRefB[i])->reshape(d, d),
										BrOpt->slice(columnsStartRefB[i],columnsEndRefB[i])->reshape(d, d) 
									)
							   ), mosek::fusion::Domain::inPSDCone(2*d));
		}

		// Symmetry constraints 
		for (int i=0; i<matchingRows.size(); i++){
			modelB->constraint(mosek::fusion::Expr::sub(BrOpt->slice(rowsStartRefB[matchingRows[i][0]], rowsEndRefB[matchingRows[i][0]]), BrOpt->slice(rowsStartRefB[matchingRows[i][1]], rowsEndRefB[matchingRows[i][1]])), mosek::fusion::Domain::equalsTo(zeroRefB));
			modelB->constraint(mosek::fusion::Expr::add(BiOpt->slice(rowsStartRefB[matchingRows[i][0]], rowsEndRefB[matchingRows[i][0]]), BiOpt->slice(rowsStartRefB[matchingRows[i][1]], rowsEndRefB[matchingRows[i][1]])), mosek::fusion::Domain::equalsTo(zeroRefB));
		}

		// Solve the SDP
		modelB->solve();
		
		// Extract the results
		finalResult = modelB->primalObjValue() / d;
		int matHeightB = d*d;
		int matWidthB = numMeasureB*numOutcomeB;
		auto tempBr = *(BrOpt->level());
		auto tempBi = *(BiOpt->level());
		for (int i=0; i<matWidthB*matHeightB; i++){
			Br[i/matWidthB][i%matWidthB] = tempBr[i];
			Bi[i/matWidthB][i%matWidthB] = tempBi[i];
		}

		// Destroy the model
		modelB->dispose();

		// Output after this section
		std::cout << std::fixed << std::setprecision(5) << "iter " << std::setw(3) << iter << " after B opt " << finalResult << std::endl;

		// ----------------------------
		//    Fixing B, optimising A
		// ----------------------------
		
		// Create references to the fixed matrix
		auto BrRef = monty::new_array_ptr(Br);
		auto BiRef = monty::new_array_ptr(Bi);

		// Create the MOSEK model 
		mosek::fusion::Model::t modelA = new mosek::fusion::Model(); 

		// The moment matrices to optimise
		mosek::fusion::Variable::t ArOpt = modelA->variable(dimRefA, mosek::fusion::Domain::inRange(-1.0, 1.0));
		mosek::fusion::Variable::t AiOpt = modelA->variable(dimRefA, mosek::fusion::Domain::inRange(-1.0, 1.0));

		// Set up the objective function 
		modelA->objective(mosek::fusion::ObjectiveSense::Maximize, mosek::fusion::Expr::dot(CRef, mosek::fusion::Expr::sub(mosek::fusion::Expr::mul(ArOpt, BrRef), mosek::fusion::Expr::mul(AiOpt, BiRef))));

		// Force the trace of each matrix to be a certain value 
		if (restrictRankA){
			mosek::fusion::Expression::t sum = mosek::fusion::Expr::add(ArOpt->slice(columnsStartRefA[0], columnsEndRefA[0]), ArOpt->slice(columnsStartRefA[d+1], columnsEndRefA[d+1]));
			for (int i=2; i<d; i++){
				sum = mosek::fusion::Expr::add(sum, ArOpt->slice(columnsStartRefA[i*(d+1)], columnsEndRefA[i*(d+1)]));
			}
			modelA->constraint(sum, mosek::fusion::Domain::equalsTo(rankARef));
		}

		// Ensure the probability isn't imaginary
		modelA->constraint(mosek::fusion::Expr::add(mosek::fusion::Expr::mul(ArOpt, BiRef), mosek::fusion::Expr::mul(AiOpt, BrRef)), mosek::fusion::Domain::equalsTo(zero2DRef));

		// For each set of measurements, the matrices should sum to the identity
		for (int i=0; i<rowsStartRefA.size(); i+=numOutcomeA){
			modelA->constraint(mosek::fusion::Expr::sum(ArOpt->slice(rowsStartRefA[i], rowsEndRefA[i+numOutcomeA-1]), 0), mosek::fusion::Domain::equalsTo(identityRef));
			modelA->constraint(mosek::fusion::Expr::sum(AiOpt->slice(rowsStartRefA[i], rowsEndRefA[i+numOutcomeA-1]), 0), mosek::fusion::Domain::equalsTo(zero1DRef));
		}

		// Each section of A should also be >= 0
		for (int i=0; i<rowsStartRefA.size(); i++){
			modelA->constraint(mosek::fusion::Expr::vstack(
									mosek::fusion::Expr::hstack(
										ArOpt->slice(rowsStartRefA[i],rowsEndRefA[i])->reshape(d, d), 
										mosek::fusion::Expr::neg(AiOpt->slice(rowsStartRefA[i],rowsEndRefA[i])->reshape(d, d))
									), 
									mosek::fusion::Expr::hstack(
										AiOpt->slice(rowsStartRefA[i],rowsEndRefA[i])->reshape(d, d),
										ArOpt->slice(rowsStartRefA[i],rowsEndRefA[i])->reshape(d, d) 
									)
							   ), mosek::fusion::Domain::inPSDCone(2*d));
		}

		// Symmetry constraints 
		for (int i=0; i<matchingRows.size(); i++){
			modelA->constraint(mosek::fusion::Expr::sub(ArOpt->slice(columnsStartRefA[matchingRows[i][0]], columnsEndRefA[matchingRows[i][0]]), ArOpt->slice(columnsStartRefA[matchingRows[i][1]], columnsEndRefA[matchingRows[i][1]])), mosek::fusion::Domain::equalsTo(zeroRefA));
			modelA->constraint(mosek::fusion::Expr::add(AiOpt->slice(columnsStartRefA[matchingRows[i][0]], columnsEndRefA[matchingRows[i][0]]), AiOpt->slice(columnsStartRefA[matchingRows[i][1]], columnsEndRefA[matchingRows[i][1]])), mosek::fusion::Domain::equalsTo(zeroRefA));
		}

		// Solve the SDP
		modelA->solve();
		
		// Extract the results
		finalResult = modelA->primalObjValue() / d;
		auto tempAr = *(ArOpt->level());
		auto tempAi = *(AiOpt->level());
		int matWidthA = d*d;
		int matHeightA = numMeasureA*numOutcomeA;
		for (int i=0; i<matWidthA*matHeightA; i++){
			Ar[i/matWidthA][i%matWidthA] = tempAr[i];
			Ai[i/matWidthA][i%matWidthA] = tempAi[i];
		}

		// Destroy the model
		modelA->dispose();

		// Output after this section
		std::cout << std::fixed << std::setprecision(5) << "iter " << std::setw(3) << iter << " after A opt " << finalResult << std::endl;

		// See if it's converged
		if (std::abs(finalResult-prevResult) < tol){
			break;
		}

		// The new is now the old
		prevResult = finalResult;

	}

	// Output the raw arrays
	if (verbosity >= 2){
		std::cout << std::endl;
		prettyPrint("Ar =", Ar, d*d, numMeasureA*numOutcomeA);
		std::cout << std::endl;
		prettyPrint("Ai =", Ai, d*d, numMeasureA*numOutcomeA);
		std::cout << std::endl;
		prettyPrint("Br =", Br, numMeasureB*numOutcomeB, d*d);
		std::cout << std::endl;
		prettyPrint("Bi =", Bi, numMeasureB*numOutcomeB, d*d);
		std::cout << std::endl;
	}

	// Extract the results from the A matrix
	complex4 A(numMeasureA, complex3(numOutcomeA, complex2(d, complex1(d))));
	for (int x=0; x<numMeasureA; x++){
		for (int a=0; a<numOutcomeA; a++){
			for (int i=0; i<d*d; i++){
				A[x][a][i/d][i%d] = Ar[x*numOutcomeA+a][i] + im*Ai[x*numOutcomeA+a][i];
			}
			prettyPrint("A[" + std::to_string(x) + "][" + std::to_string(a) + "] = ", A[x][a], d, d);
			std::cout << std::endl;
		}
	}
	
	// Write some of the eigenvectors to file
	if (d == 2){
		std::ofstream outFile;
		outFile.open ("vectors.dat");
		for (int x=0; x<numMeasureA; x++){

			// Copy the matrix
			Eigen::Matrix2cd matTest;
			for (int i=0; i<d; i++){
				for (int j=0; j<d; j++){
					matTest(i,j) = A[x][0][j][i];
				}
			}

			// Get the eigenvalues
			Eigen::ComplexEigenSolver<Eigen::Matrix2cd> eigensolver(matTest);
			if (eigensolver.info() != Eigen::Success) abort();
			std::vector<std::complex<double>> vec(d);
			for (int i=0; i<d; i++){
				vec[i] = eigensolver.eigenvectors().col(1)(i);
			}

			// Make sure the |0> component is real 
			double phase = atan(-std::imag(vec[0]) / std::real(vec[0]));
			std::complex<double> phaseComplex(cos(phase), sin(phase));
			vec[0] *= phaseComplex;
			vec[1] *= phaseComplex;

			// Turn this into a point on the Bloch-sphere
			double theta = 2*acos(std::real(vec[0]));
			std::complex<double> test = vec[1] / sin(theta/2);
			double phi = std::arg(test);
			double xCoord = cos(phi)*sin(theta);
			double yCoord = sin(phi)*sin(theta);
			double zCoord = cos(theta);

			// Write this to file
			outFile << "0 0 0 " << xCoord << " " << yCoord << " " << zCoord << std::endl;;

		}
		outFile.close();
	}

	// Extract the results from the B matrix
	complex4 B(numMeasureB, complex3(numOutcomeB, complex2(d, complex1(d))));
	for (int y=0; y<numMeasureB; y++){
		for (int b=0; b<numOutcomeB; b++){
			for (int i=0; i<d*d; i++){
				B[y][b][i/d][i%d] = Br[i][y*numOutcomeB+b] + im*Bi[i][y*numOutcomeB+b];
			}
			if (verbosity >= 2){
				prettyPrint("B[" + std::to_string(y) + "][" + std::to_string(b) + "] = ", B[y][b], d, d);
				std::cout << std::endl;
			}
		}
	}

	// Output the results
	std::cout << std::setprecision(5) << "final result = " << finalResult << " <= " << (numMeasureB/2)*2*root2 << std::endl;

}

// Standard cpp entry point
int main (int argc, char ** argv) {

	// Whether to use the extension for d > 2
	int method = 1;

	// The number of measurements
	int n = 2;

	// The dimension
	int d = 2;

	// Loop over the command-line arguments
	for (int i=1; i<argc; i++){

		// Convert the char array to a standard string for easier processing
		std::string arg = argv[i];

		// If asking for help
		if (arg == "-h" || arg == "--help") {
			std::cout << "" << std::endl;
			std::cout << "------------------------------" << std::endl;
			std::cout << "  program that uses a seesaw" << std::endl;
			std::cout << "    of a Bell-scenario to" << std::endl;
			std::cout << "       check for MUBs" << std::endl;
			std::cout << "------------------------------" << std::endl;
			std::cout << "-h         show the help" << std::endl;
			std::cout << "-d [int]   set the dimension" << std::endl;
			std::cout << "-n [int]   set the number of measurements" << std::endl;
			std::cout << "-v         verbose output" << std::endl;
			std::cout << "-e         use the extended method" << std::endl;
			std::cout << "" << std::endl;
			return 0;

		// Set the number of measurements for Alice
		} else if (arg == "-n") {
			n = std::stoi(argv[i+1]);
			i += 1;

		// Set the dimension
		} else if (arg == "-d") {
			d = std::stoi(argv[i+1]);
			i += 1;

		// Set the verbosity
		} else if (arg == "-v") {
			verbosity = 2;
			i += 1;

		// Use the extended method
		} else if (arg == "-e") {
			method = 2;

		// Use the approximate method
		} else if (arg == "-a") {
			method = 3;

		// Otherwise it's an error
		} else {
			std::cout << "ERROR - unknown argument: \"" << arg << "\"" << std::endl;
			return 1;
		}

	}

	// Perform the seesaw
	if (method == 3){
		seesawApprox(d, n);
	} else if (method == 2){
		seesawExtended(d, n);
	} else {
		seesaw(d, n);
	}

}

