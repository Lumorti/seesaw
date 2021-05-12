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

// Multiply two matrices
complex2 multiply(complex2 mat1, complex2 mat2){

	// Set the dimensions: n x m (x) m x q = n x q
	complex2 mult(mat1.size(), std::vector<std::complex<double>>(mat2[0].size()));

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

// Standard cpp entry point
int main (int argc, char ** argv) {

	// Useful values
	int stateSize = pow(2, d);
	double root2 = sqrt(2.0);
	double overRoot2 = 1.0/sqrt(2.0);

	// The coefficients C such that S = sum(C_{a,b,x,y}*p(a,b|x,y))
	real2 C(numMeasureA, real1(numMeasureB));

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
	double t1 = 1.0+root2;
	double t2 = 1.0-root2;
	double t3 = -1.0+root2;
	double t4 = -1.0-root2;
	double mag1 = 4.0+2*root2;
	double mag2 = 4.0-2*root2;
	double mag3 = 4.0-2*root2;
	double mag4 = 4.0+2*root2;
	B[0][0] = {{pow(t1, 2)/mag1, t1/mag1},
			   {t1/mag1,         1/mag1}};
	B[0][1] = {{pow(t2, 2)/mag2, t2/mag2},
			   {t2/mag2,         1/mag2}};
	B[1][0] = {{pow(t3, 2)/mag3, t3/mag3},
			   {t3/mag3,         1/mag3}};
	B[1][1] = {{pow(t4, 2)/mag4, t4/mag4},
			   {t4/mag4,         1/mag4}};
	
	// The known best state for rho for the CHSH inequality
	complex2 psiPlus = {{overRoot2, 0, 0, overRoot2}};
	rho = outer(transpose(psiPlus), psiPlus);
	
	// Define the CHSH inequality 
	C[0][0] = 1;
	C[0][1] = -1;
	C[1][0] = 1;
	C[1][1] = 1;

	// Outputs
	prettyPrint("rho = ", rho, stateSize, stateSize);
	for (int x = 0; x < A.size(); x++){
		for (int a = 0; a < A[x].size(); a++){
			std::cout << std::endl;
			prettyPrint("A["+std::to_string(x)+"]["+std::to_string(a)+"]=", A[x][a], d, d);
		}
	}
	for (int y = 0; y < B.size(); y++){
		for (int b = 0; b < B[y].size(); b++){
			std::cout << std::endl;
			prettyPrint("B["+std::to_string(y)+"]["+std::to_string(b)+"]=", B[y][b], d, d);
		}
	}

	// Evaluate once
	std::cout << std::endl;
	double result = evaluate(C, A, B, rho);
	
	// Output the result
	std::cout << std::endl;
	std::cout << "result = " << result << std::endl;

}

