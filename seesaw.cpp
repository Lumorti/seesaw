#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <complex>
#include <iomanip>

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

// Dimension
const int d = 2;

// Number of possible measurements
const int numMeasureA = 2;
const int numMeasureB = 2;

// Number of possible outputs
const int numOutcomeA = 2;
const int numOutcomeB = 2;

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

// Get the trace of a matrix (summing the diagonals)
template <typename type> type trace(std::vector<std::vector<type>> mat){
	type sum = 0;
	for (int i=0; i<mat.size(); i++){
		sum += mat[i][i];
	}
	return sum;
}

// Get the inner product of two matrices (summing the multiplied diagonals)
template <typename type> type inner(std::vector<std::vector<type>> mat1, std::vector<std::vector<type>> mat2){
	type sum = 0;
	for (int i=0; i<mat1.size(); i++){
		sum += mat1[i][i] * mat2[i][i];
	}
	return sum;
}

// Transpose a matrix
template <typename type> std::vector<std::vector<type>> transpose(std::vector<std::vector<type>> mat){
	std::vector<std::vector<type>> matTran(mat[0].size(), std::vector<type>(mat.size()));
	for (int i=0; i<mat.size(); i++){
		for (int j=0; j<mat[0].size(); j++){
			matTran[j][i] = mat[i][j];
		}
	}
	return matTran;
}

// Multiply two matrices
template <typename type> std::vector<std::vector<type>> multiply(std::vector<std::vector<type>> mat1, std::vector<std::vector<type>> mat2){

	// Set the dimensions: n x m (x) m x q = n x q
	std::vector<std::vector<type>> mult(mat1.size(), std::vector<type>(mat2[0].size()));

	// For each element in the new matrix
	for (int i=0; i<mult.size(); i++){
		for (int j=0; j<mult[0].size(); j++){

			// Add the row from mat1 times the column from mat2
			for (int k=0; k<mat2.size(); k++){
				mult[i][j] += mat1[i][k] * mat2[k][j];
			}
			
		}
	}

	return mult;

}

// Get the output product of two matrices
template <typename type> std::vector<std::vector<type>> outer(std::vector<std::vector<type>> mat1, std::vector<std::vector<type>> mat2){

	// Set the dimensions: n x m (x) p x q = np x mq
	std::vector<std::vector<type>> product(mat1.size()*mat2.size(), std::vector<type>(mat1[0].size()*mat2[0].size()));

	// Loop over the first matrix
	for (int i=0; i<mat1.size(); i++){
		for (int j=0; j<mat1[0].size(); j++){

			// And also the second
			for (int k=0; k<mat2.size(); k++){
				for (int l=0; l<mat2[0].size(); l++){

					// The components of the outer product
					product[i*mat2.size()+k][j*mat2[0].size()+l] = mat1[i][j] * mat2[k][l];
		
				}
			}

		}
	}

	// Return this much larger matrix
	return product;

}

// Given C, A, B and rho, see what this evaluates to
double evaluate(real2 &C, complex4 &A, complex4 &B, complex2 &rho){

	// The total 
	double S = 0;
	double prob = 0;
	double expect = 0;

	// For each combination of measurements 
	for (int x = 0; x < A.size(); x++){
		for (int y = 0; y < B.size(); y++){

			// For each combination of outcomes
			for (int a = 0; a < A[0].size(); a++){
				for (int b = 0; b < B[0].size(); b++){

					// Calculate the probability
					prob = trace(multiply(rho, outer(A[x][a], B[y][b]))).real();

					// Since the outcomes are +1 or -1
					if (a == b){
						expect += prob;
					} else {
						expect -= prob;
					}

				}
			}

			// Add based on the Bell inequality definition
			S += C[x][y] * expect;
			std::cout << "x = " << x << "  y = " << y << "    expect = " << expect << "    C = " << C[x][y] << std::endl;

		}
	}

	// Return the evaluated inequality
	return S;

}

// Perform the seesaw method to optimise both A and B
double seesaw(real2 &Ar, real2 &Ai, real2 &Br, real2 &Bi, real2 &C){

	// The inequality value to eventually return
	double finalResult = 0;

	// Create the arrays to be used to select the columns of the B array 
	std::vector<std::shared_ptr<monty::ndarray<int,1>>> columnsStartRefB;
	std::vector<std::shared_ptr<monty::ndarray<int,1>>> columnsEndRefB;
	for (int i=0; i<numMeasureA*numOutcomeA; i++){
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
	std::vector<double> identity(d*d, 0.0);
	for (int i=0; i<d; i++){
		identity[i*(d+1)] = 1.0;
	}
	auto identityRef = monty::new_array_ptr(identity);

	// The list of row indices which should be identical 
	std::vector<std::vector<int>> matchingRows;
	for (int i=1; i<d; i++){
		for (int j=0; j<d-1; j++){
			matchingRows.push_back({j*d+i, (j+1)*d+(i-1)});
		}
	}

	// Create a reference to an array of zeros
	std::vector<double> zero1D(d*d, 0);
	std::vector<std::vector<double>> zero(1, std::vector<double>(d*d, 0));
	auto zero1DRef = monty::new_array_ptr(zero1D);
	auto zeroRefB = monty::new_array_ptr(zero);
	auto zeroRefA = monty::new_array_ptr(transpose(zero));

	// The sizes of the variable matrix
	auto dimRefA = monty::new_array_ptr(std::vector<int>({numOutcomeA*numMeasureA, d*d}));
	auto dimRefB = monty::new_array_ptr(std::vector<int>({d*d, numOutcomeB*numMeasureB}));

	// C is always fixed
	auto CRef = monty::new_array_ptr(C);

	// Keep seesawing 
	for (int iter=0; iter<3; iter++){

		// ----------------------------
		//    Fixing A, optimising B
		// ----------------------------

		// Create references to the fixed matrices
		auto ArRef = monty::new_array_ptr(Ar);
		auto AiRef = monty::new_array_ptr(Ai);

		// Clear the B matrices
		Br = real2(d*d, real1(numOutcomeB*numMeasureB, 0));
		Bi = real2(d*d, real1(numOutcomeB*numMeasureB, 0));

		// Create the MOSEK model 
		mosek::fusion::Model::t modelB = new mosek::fusion::Model(); 

		// The moment matrices to optimise
		mosek::fusion::Variable::t BrOpt = modelB->variable(dimRefB, mosek::fusion::Domain::inRange(-1.0, 1.0));
		mosek::fusion::Variable::t BiOpt = modelB->variable(dimRefB, mosek::fusion::Domain::inRange(-1.0, 1.0));

		// Set up the objective function 
		modelB->objective(mosek::fusion::ObjectiveSense::Maximize, mosek::fusion::Expr::dot(CRef, mosek::fusion::Expr::sub(mosek::fusion::Expr::mul(ArRef, BrOpt), mosek::fusion::Expr::mul(AiRef, BiOpt))));

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
		int matWidthB = d*d;
		int matHeightB = numMeasureB*numOutcomeB;
		auto tempBr = *(BrOpt->level());
		auto tempBi = *(BiOpt->level());
		for (int i=0; i<matWidthB*matHeightB; i++){
			Br[i/matWidthB][i%matHeightB] = tempBr[i];
			Bi[i/matWidthB][i%matHeightB] = tempBi[i];
		}

		// Destroy the model
		modelB->dispose();

		// Output after this section
		std::cout << "iter " << iter << " after B opt " << finalResult << std::endl;

		// TODO 
		return finalResult;

		// ----------------------------
		//    Fixing B, optimising A
		// ----------------------------
		
		// Create references to the fixed matrix
		//auto BRef = monty::new_array_ptr(B);

		//// Clear the A matrix
		//A = real2(numOutcomeB*numMeasureB, real1(d*d, 0));

		//// Create the MOSEK model 
		//mosek::fusion::Model::t modelA = new mosek::fusion::Model(); 

		//// The moment matrix to optimise
		//mosek::fusion::Variable::t AOpt = modelA->variable(dimRefA, mosek::fusion::Domain::inRange(-1.0, 1.0));

		//// Set up the objective function 
		//modelA->objective(mosek::fusion::ObjectiveSense::Maximize, mosek::fusion::Expr::dot(CRef, mosek::fusion::Expr::mul(AOpt, BRef)));

		//// For each set of measurements, the matrices should sum to the identity
		//for (int i=0; i<rowsStartRefA.size(); i+=numOutcomeA){
			//modelA->constraint(mosek::fusion::Expr::sum(AOpt->slice(rowsStartRefA[i], rowsEndRefA[i+numOutcomeA-1]), 0), mosek::fusion::Domain::equalsTo(identityRef));
		//}

		//// Each section of A should also be >= 0
		//for (int i=0; i<rowsStartRefA.size(); i++){
			//modelA->constraint(AOpt->slice(rowsStartRefA[i], rowsEndRefA[i])->reshape(d, d), mosek::fusion::Domain::inPSDCone(d));
		//}

		//// Symmetry constraints 
		//for (int i=0; i<matchingRows.size(); i++){
			//modelA->constraint(mosek::fusion::Expr::sub(AOpt->slice(columnsStartRefA[matchingRows[i][0]], columnsEndRefA[matchingRows[i][0]]), AOpt->slice(columnsStartRefA[matchingRows[i][1]], columnsEndRefA[matchingRows[i][1]])), mosek::fusion::Domain::equalsTo(zeroRefA));
		//}

		//// Solve the SDP
		//modelA->solve();
		
		//// Extract the results
		//finalResult = modelA->primalObjValue() / d;
		//auto tempA = *(AOpt->level());
		//int matWidthA = d*d;
		//int matHeightA = numMeasureA*numOutcomeA;
		//for (int i=0; i<matWidthA*matHeightA; i++){
			//A[i/matWidthA][i%matHeightA] = tempA[i];
		//}

		//// Destroy the model
		//modelA->dispose();

		//// Output after this section
		//std::cout << "iter " << iter << " after A opt " << finalResult << std::endl;

	}

	// Return the evaluated expression, A and B have also been modified
	return finalResult;

}

// Standard cpp entry point
int main (int argc, char ** argv) {

	// Useful values
	double root2 = sqrt(2.0);

	// The coefficients defining the inequality
	real2 C = {{ 1.0,-1.0,-1.0, 1.0}, 
		       {-1.0, 1.0, 1.0,-1.0},
			   { 1.0,-1.0, 1.0,-1.0},
			   {-1.0, 1.0,-1.0, 1.0}};

	// The arrays to store the operators (real and imaginary separately)
	real2 Ar(numOutcomeA*numMeasureA, real1(d*d));
	real2 Ai(numOutcomeA*numMeasureA, real1(d*d));
	real2 Br(d*d, real1(numOutcomeB*numMeasureB));
	real2 Bi(d*d, real1(numOutcomeB*numMeasureB));

	// The initial guess for A
	Ar = {{1.0, 0.0,-1.0, 0.0}, 
		  {0.0, 0.0, 0.0, 1.0},
		  {1.0, 1.0, 0.0, 1.0},
		  {0.0, 1.0, 0.0, 1.0}};
	Ai = {{0.0, 0.0, 0.0, 0.0}, 
	      {0.0, 0.0, 0.0, 0.0},
		  {0.0, 0.0, 0.0, 0.0},
		  {0.0, 0.0, 0.0, 0.0}};

	// Perform the seesaw
	double result = seesaw(Ar, Ai, Br, Bi, C);

	// Output the results
	std::cout << std::endl;
	prettyPrint("Ar = ", Ar, numMeasureB*numOutcomeB, d*d);
	std::cout << std::endl;
	prettyPrint("Ai = ", Ai, numMeasureB*numOutcomeB, d*d);
	std::cout << std::endl;
	prettyPrint("Br = ", Br, d*d, numMeasureB*numOutcomeB);
	std::cout << std::endl;
	prettyPrint("Bi = ", Bi, d*d, numMeasureB*numOutcomeB);
	std::cout << std::endl;
	std::cout << "final result = " << result << " <= " << 2*root2 << std::endl;

}

