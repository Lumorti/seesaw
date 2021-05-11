#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <complex>
#include <iomanip>

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
std::complex<double> trace(complex2 mat){
	std::complex<double> sum = 0;
	for (int i=0; i<mat.size(); i++){
		sum += mat[i][i];
	}
	return sum;
}

// Get the inner product of two matrices (summing the multiplied diagonals)
std::complex<double> inner(complex2 mat1, complex2 mat2){
	std::complex<double> sum = 0;
	for (int i=0; i<mat1.size(); i++){
		sum += mat1[i][i] * mat2[i][i];
	}
	return sum;
}

// Transpose a matrix
complex2 transpose(complex2 mat){
	complex2 matTran(mat[0].size(), complex1(mat.size()));
	for (int i=0; i<mat.size(); i++){
		for (int j=0; j<mat[0].size(); j++){
			matTran[j][i] = mat[i][j];
		}
	}
	return matTran;
}

// Get the output product of two matrices 
complex2 outer(complex2 mat1, complex2 mat2){

	// Set the dimensions: n x m (x) p x q = np x mq
	complex2 product(mat1.size()*mat2.size(), std::vector<std::complex<double>>(mat1[0].size()*mat2[0].size()));

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
double evaluate(real4 &C, complex4 &A, complex4 &B, complex2 &rho){

	// The total 
	double S = 0;

	// For each combination of measurements and results
	for (int a = 0; a < A.size(); a++){
		for (int b = 0; b < B.size(); b++){
			for (int x = 0; x < A[a].size(); x++){
				for (int y = 0; y < B[b].size(); y++){

					// Add the contribution
					S += C[a][b][x][y] * inner(rho, outer(A[a][x], B[b][y])).real();

				}
			}
		}
	}

	// Return the evaluated inequality
	return S;

}

// Standard cpp entry point
int main (int argc, char ** argv) {

	// Useful values
	int stateSize = pow(2, d);
	double root2 = sqrt(2.0);
	double overRoot2 = 1.0/sqrt(2.0);

	// The coefficients C such that S = sum(C_{a,b,x,y}*p(a,b|x,y))
	real4 C(numOutcomeA, real3(numOutcomeB, real2(numMeasureA, real1(numMeasureB))));

	// Sets of operators on Alice and Bob
	complex4 A(numMeasureA, complex3(numOutcomeA, complex2(d, complex1(d))));
	complex4 B(numMeasureB, complex3(numOutcomeB, complex2(d, complex1(d))));

	// The shared quantum state
	complex2 rho(stateSize, complex1(stateSize));

	// The known best values for A for the CHSH inequality
	A[0][0] = {{1, 0},
			   {0, 0}};
	A[0][1] = {{0, 0},
			   {0, 1}};
	A[1][0] = {{0.5, 0.5},
			   {0.5, 0.5}};
	A[1][1] = {{ 0.5, -0.5},
			   {-0.5,  0.5}};

	// The known best values for B for the CHSH inequality TODO
	
	// The known best state for rho for the CHSH inequality
	complex2 psiPlus = {{overRoot2, 0, 0, overRoot2}};
	rho = outer(transpose(psiPlus), psiPlus);
	
	// Define the CHSH inequality 
	real2 expectCoeffs(numMeasureA, real1(numMeasureB));
	expectCoeffs[0][0] = 1;
	expectCoeffs[0][1] = 1;
	expectCoeffs[1][0] = 1;
	expectCoeffs[1][1] = -1;
	for (int i=0; i<expectCoeffs.size(); i++){
		for (int j=0; j<expectCoeffs[i].size(); j++){
			for (int k=0; k<numOutcomeA; k++){
				for (int l=0; l<numOutcomeB; l++){
					if (k == 0){
						C[k][l][i][j] = expectCoeffs[i][j];
					} else {
						C[k][l][i][j] = -expectCoeffs[i][j];
					}
				}
			}
		}
	}

	// Outputs
	prettyPrint("rho = ", rho, stateSize, stateSize);

	// Evaluate once
	double result = evaluate(C, A, B, rho);
	
	// Output the result
	std::cout << result << std::endl;

}

