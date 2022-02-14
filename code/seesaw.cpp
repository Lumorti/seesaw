#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <complex>
#include <iomanip>
#include <random>
#include <fstream>
#include <chrono>
#include <math.h>

// Allow use of "2i" for complex
using namespace std::complex_literals;

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
using int1 = std::vector<int>;
using int2 = std::vector<int1>;

// Hyperrectangle structure
class hyperRect {
	public:
		real1 l;
		real1 L;
		real1 m;
		real1 M;
		hyperRect(int a, int b){
			l = real1(a*a);
			L = real1(a*a);
			m = real1(b*b);
			M = real1(b*b);
		};
		void operator=(hyperRect oth){
			l = oth.l;
			L = oth.L;
			m = oth.m;
			M = oth.M;
		}
		double getVolume(){
			double vol = -100000;
			for (int i=0; i<m.size(); i++){
				if (M[i]-m[i] > vol){
					vol = M[i]-m[i];
				}
			}
			return vol;
		}
};

// Useful values
const double root2 = sqrt(2.0);
const std::complex<double> im = sqrt(std::complex<double>(-1.0));

// How many hyperrects to split into each iteration
int numRects = 2;

// How many decimals to output for the matrices
int precision = 9;

// Seesaw iterations
int numIters = 100000;

// Convergence criteria
double tol = 1e-8;
double epsilon = 1e-5;
int numInRowRequired = 10;

// What counts as zero
double zeroThresh = 1e-7;

// How much to output (0 == none, 1 == normal, 2 == extra)
int verbosity = 1;

// Which random method to use
int randomMethod = 2;

// Whether to use the known ideal solution if possible
bool useIdeal = false;

// The seed to use for randomness
std::string seed = "";

// Force matrix values
std::vector<double> fixedVals;

// What to output
int outputMethod = 1;

// Whether to force the rank of each matrix
bool restrictRankA = false;
bool restrictRankB = false;

// Dot product of two 1D vectors
double dot(real1 a1, real1 a2){
	double toReturn = 0;
	for (int i=0; i<a1.size(); i++){
		toReturn += a1[i]*a2[i];
	}
	return toReturn;
}

// Dot product of two 1D complex vectors
std::complex<double> dot(complex1 a1, complex1 a2){
	std::complex<double> toReturn = 0;
	for (int i=0; i<a1.size(); i++){
		toReturn += a1[i]*std::conj(a2[i]);
	}
	return toReturn;
}

// Get the trace of a matrix (summing the diagonals)
std::complex<double> trace(complex2 mat){
	std::complex<double> sum = 0;
	for (int i=0; i<mat.size(); i++){
		sum += mat[i][i];
	}
	return sum;
}

// Get the inner product of two matrices 
double inner(real2 mat1, real2 mat2){
	double sum = 0;
	for (int i=0; i<mat1.size(); i++){
		for (int j=0; j<mat1[0].size(); j++){
			sum += mat1[i][j] * mat2[i][j];
		}
	}
	return sum;
}

// Get the inner product of two matrices 
std::complex<double> inner(complex2 mat1, complex2 mat2){
	std::complex<double> sum = 0;
	for (int i=0; i<mat1.size(); i++){
		for (int j=0; j<mat1[0].size(); j++){
			sum += mat1[i][j] * std::conj(mat2[i][j]);
		}
	}
	return sum;
}

// Get the inner product of two matrices 
std::complex<double> inner(real2 mat1, complex2 mat2){
	std::complex<double> sum = 0;
	for (int i=0; i<mat1.size(); i++){
		for (int j=0; j<mat1[0].size(); j++){
			sum += mat1[i][j] * std::conj(mat2[i][j]);
		}
	}
	return sum;
}

// Get the inner product of two matrices 
std::complex<double> inner(complex2 mat1, real2 mat2){
	std::complex<double> sum = 0;
	for (int i=0; i<mat1.size(); i++){
		for (int j=0; j<mat1[0].size(); j++){
			sum += mat1[i][j] * std::conj(mat2[i][j]);
		}
	}
	return sum;
}

// Get the inner product of a matrix with a sparse matrix
double inner(real2 mat1, monty::rc_ptr<mosek::fusion::Matrix> mat2){
	double sum = 0;
	int nonZero = mat2->numNonzeros();
	auto subi = monty::new_array_ptr(std::vector<int>(nonZero));
	auto subj = monty::new_array_ptr(std::vector<int>(nonZero));
	auto val = monty::new_array_ptr(std::vector<double>(nonZero));
	mat2->getDataAsTriplets(subi, subj, val);
	for (int i=0; i<val->size(); i++){
		sum += mat1[subi->operator[](i)][subj->operator[](i)] * val->operator[](i);
	}
	return sum;
}

// Transpose a matrix
complex2 transpose(complex2 mat){
	complex2 matTran(mat[0].size(), complex1(mat.size()));
	for (int i=0; i<mat.size(); i++){
		for (int j=0; j<mat[0].size(); j++){
			matTran[j][i] = std::conj(mat[i][j]);
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

// Multiply two matrices
complex2 multiply(complex2 mat1, real2 mat2){

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

// Get the output product of two vectors (assuming second is transposed)
complex2 outer(complex1 mat1, complex1 mat2){

	// Set the dimensions: n x m (x) p x q = np x mq
	complex2 product(mat1.size(), complex1(mat2.size()));

	// Loop over the first matrix
	for (int i=0; i<mat1.size(); i++){

		// And also the second
		for (int j=0; j<mat2.size(); j++){

			// The components of the outer product
			product[i][j] = mat1[i] * mat2[j];

		}

	}

	// Return this much larger matrix
	return product;

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

// Projector operator on two 1D vectors
template<typename type1>
std::vector<type1> proj(std::vector<type1> u, std::vector<type1> v){
	std::vector<type1> toReturn = u;
	type1 factor = dot(u,v) / dot(u,u);
	for (int i=0; i<u.size(); i++){
		toReturn[i] *= factor;
	}
	return toReturn;
}

// Use the Gram-Schmidt process to orthonormalise a set of vectors
void makeOrthonormal(complex2 v, complex2 &u){

	// For each new vector
	for (int i=0; i<u.size(); i++){

		// Start with the original vector
		u[i] = v[i];

		// For each previous vector
		complex1 prev = v[i];
		for (int j=0; j<i; j++){
			complex1 deltaVec = proj(u[j], prev);
			for (int k=0; k<deltaVec.size(); k++){
				u[i][k] -= deltaVec[k];
			}
			prev = u[i];
		}

		// Normalise the vector
		std::complex<double> factor = sqrt(dot(u[i], u[i]));
		for (int k=0; k<u.size(); k++){
			u[i][k] /= factor;
		}

	}

}

// Pretty print a generic 1D vector
template <typename type> void prettyPrint(std::string pre, std::vector<type> arr){

	// Used fixed precision
	std::cout << std::fixed << std::showpos << std::setprecision(precision);

	// For the first line, add the pre text
	std::cout << pre << " { ";

	// For the x values, combine them all on one line
	for (int x=0; x<arr.size(); x++){
		std::cout << arr[x];
		if (x < arr.size()-1){
			std::cout << ", ";
		}
	}

	// Output the row
	std::cout << "}" << std::endl;

	// Reset things for normal output
	std::cout << std::noshowpos;

}

// Pretty print a generic 2D array
template <typename type> void prettyPrint(std::string pre, std::vector<std::vector<type>> arr){

	// Used fixed precision
	std::cout << std::fixed << std::showpos << std::setprecision(precision);

	// Loop over the array
	std::string rowText;
	for (int y=0; y<arr.size(); y++){

		// For the first line, add the pre text
		if (y == 0){
			rowText = pre + "{";

		// Otherwise pad accordingly
		} else {
			rowText = "";
			while (rowText.length() < pre.length()+1){
				rowText += " ";
			}
		}

		// Spacing
		std::cout << rowText << " { ";

		// For the x values, combine them all on one line
		for (int x=0; x<arr[y].size(); x++){
			std::cout << std::setw(5) << arr[y][x];
			if (x < arr[y].size()-1){
				std::cout << ", ";
			}
		}

		// Output the row
		std::cout << "}";
		if (y < arr.size()-1){
			std::cout << ",";
		}
		std::cout << std::endl;

	}

	// Reset things for normal output
	std::cout << std::noshowpos;

}

// Pretty print a complex 2D array
void prettyPrint(std::string pre, complex2 arr){

	// Used fixed precision
	std::cout << std::showpos << std::fixed << std::setprecision(precision);

	// Loop over the array
	std::string rowText;
	for (int y=0; y<arr.size(); y++){

		// For the first line, add the pre text
		if (y == 0){
			rowText = pre + "{";

		// Otherwise pad accordingly
		} else {
			rowText = "";
			while (rowText.length() < pre.length()+1){
				rowText += " ";
			}
		}

		// Spacing
		std::cout << rowText << " { ";

		// For the x values, combine them all on one line
		for (int x=0; x<arr[y].size(); x++){
			std::cout << std::setw(5) << std::real(arr[y][x]) << std::imag(arr[y][x]) << "i ";
			if (x < arr[y].size()-1){
				std::cout << ", ";
			}
		}

		// Output the row
		std::cout << "}";
		if (y < arr.size()-1){
			std::cout << ",";
		} else {
			std::cout << " }";
		}
		std::cout << std::endl;

	}

	// Reset things for normal output
	std::cout << std::noshowpos;

}

// Pretty print a hyperrect
void prettyPrint(std::string pre, hyperRect arr){

	// Print each section
	prettyPrint(pre + ", l ", arr.l);
	prettyPrint(pre + ", L ", arr.L);
	prettyPrint(pre + ", m ", arr.m);
	prettyPrint(pre + ", M ", arr.M);

}

// Branch a hyperrectangle at a certain point
std::vector<hyperRect> branchHyperrectangle(int p, int q, hyperRect &Omega){

	// List of 4 hyperrectangles to return
	std::vector<hyperRect> toReturn(numRects, hyperRect(p, q));

	// Number of non-zero single values
	int K = std::min(p*p, q*q);

	// Determine the index which gives the biggest difference
	int I = 0;
	double bestVal = -10000000;
	double val = 0;
	double vTemp = 0;
	double wTemp = 0;
	for (int i=0; i<K; i++){
		vTemp = Omega.l[i] + (Omega.L[i]-Omega.l[i]) / 2;
		wTemp = Omega.m[i] + (Omega.M[i]-Omega.m[i]) / 2;
		val = vTemp*wTemp - std::max(Omega.m[i]*vTemp + Omega.l[i]*wTemp - Omega.l[i]*Omega.m[i],
									 Omega.M[i]*vTemp + Omega.L[i]*wTemp - Omega.L[i]*Omega.M[i]);
		if (val > bestVal){
			bestVal = val;
			I = i;
		}
	}

	// Split the y hyperrect into a number of subdivisions
	if (numRects > 2){
		double wDelta = (Omega.M[I]-Omega.m[I]) / (numRects/2);
		double vDelta = (Omega.L[I]-Omega.l[I]) / (numRects/2);
		for (int i=0; i<numRects/2; i++){
			toReturn[i] = Omega;
			toReturn[i].m[I] = Omega.m[I] + i*wDelta;
			toReturn[i].M[I] = Omega.m[I] + (i+1)*wDelta;
		}
		for (int i=numRects/2; i<numRects; i++){
			toReturn[i] = Omega;
			toReturn[i].l[I] = Omega.l[I] + i*vDelta;
			toReturn[i].L[I] = Omega.l[I] + (i+1)*vDelta;
		}
	} else {
		double wDelta = (Omega.M[I]-Omega.m[I]) / numRects;
		for (int i=0; i<numRects; i++){
			toReturn[i] = Omega;
			toReturn[i].m[I] = Omega.m[I] + i*wDelta;
			toReturn[i].M[I] = Omega.m[I] + (i+1)*wDelta;
		}
	}

	// Return these hyperrectangles
	return toReturn;

}

// Perform the seesaw method to optimise both A and B 
void seesawExtended(int d, int n){

	// Start the timer 
	auto t1 = std::chrono::high_resolution_clock::now();

	// How many permutations required
	int numPerm = n*(n-1)/2;

	// The inequality value to eventually return
	double finalResult = 0;
	double exact = d*numPerm*sqrt(d*(d-1));
	double delta = exact-finalResult;
	double fromLast = 0;

	// Amount to remove from each
	double sub = sqrt(d*(d-1))*numPerm;

	// How big should the matrices be
	int numMeasureA = d*d*numPerm;
	int numOutcomeA = 3;
	int numMeasureB = n;
	int numOutcomeB = d;

	// The rank for B to try
	restrictRankA = true;
	restrictRankB = true;
	real2 rankA = real2(numOutcomeA*numMeasureA, real1(1, 1));
	for (int i=2; i<numOutcomeA*numMeasureA; i+=numOutcomeA){
		rankA[i][0] = d-2;
	}
	real2 rankB = real2(1, real1(numOutcomeB*numMeasureB, 1));
	auto rankARef = monty::new_array_ptr(rankA);
	auto rankBRef = monty::new_array_ptr(rankB);

	// Create an identity vector
	std::vector<double> identity(d*d, 0.0);
	for (int i=0; i<d; i++){
		identity[i*(d+1)] = 1.0;
	}

	// The arrays to store the operators (real and imaginary separately)
	real2 Ar(numMeasureA*numOutcomeA, real1(d*d));
	real2 Ai(numMeasureA*numOutcomeA, real1(d*d));
	real2 Br(d*d, real1(numMeasureB*numOutcomeB));
	real2 Bi(d*d, real1(numMeasureB*numOutcomeB));

	// Which columns should be identical
	std::vector<std::vector<int>> matchingRows;
	for (int j=0; j<d; j++){
		for (int i=j+1; i<d; i++){
			int downLeft = i-j;
			matchingRows.push_back({j*d+i, (j+downLeft)*d+(i-downLeft)});
		}
	}

	// Set up the random generator
	std::mt19937 generator;
	if (seed.length() > 0){
		std::seed_seq seed1 (seed.begin(), seed.end());
		generator = std::mt19937(seed1);
	} else {
		std::random_device rd;
		generator = std::mt19937(rd());
	}
	std::uniform_real_distribution<double> distribution(-1.0, 1.0);

	// If told to just generate a random symmetric matrix
	if (randomMethod == 1){

		// Randomise B
		for (int x=0; x<numMeasureB; x++){
			for (int a=0; a<numOutcomeB-1; a++){
				for (int j=0; j<d; j++){
					for (int k=0; k<d; k++){

						// Create some random values
						Br[j*d+k][x*numOutcomeB+a] = distribution(generator);

						// Imaginary only on the off-diagonals
						if (j != k){
							Bi[j*d+k][x*numOutcomeB+a] = distribution(generator);
						}

					}
				}
			}
		}

		// Force these to be identical 
		for (int i=0; i<matchingRows.size(); i++){
			for (int j=0; j<numMeasureB*numOutcomeB; j++){
				Br[matchingRows[i][0]][j] = Br[matchingRows[i][1]][j];
				Bi[matchingRows[i][0]][j] = -Bi[matchingRows[i][1]][j];
			}
		}

		// Ensure the trace of each is one
		for (int x=0; x<numMeasureB*numOutcomeB; x++){
			double sum = 0;
			for (int i=0; i<d-1; i++){
				sum += Br[i*(d+1)][x];
			}
			Br[d*d-1][x] = rankB[0][x] - sum;
		}

		// Ensure each measurement sums to the identity
		for (int x=0; x<numMeasureB; x++){
			for (int j=0; j<d*d; j++){
				for (int a=0; a<numOutcomeB-1; a++){
					Br[j][x*numOutcomeB+numOutcomeB-1] -= Br[j][x*numOutcomeB+a];
					Bi[j][x*numOutcomeB+numOutcomeB-1] -= Bi[j][x*numOutcomeB+a];
				}
				Br[j][x*numOutcomeB+numOutcomeB-1] += identity[j];
			}
		}

	// Or if told to use matrices made out of random othonormal projectors
	} else if (randomMethod == 2){

		// For each set of measurements
		for (int x=0; x<numMeasureB; x++){

			// Create some random vectors
			complex2 randVecs(numOutcomeB, complex1(d));
			complex2 normVecs(numOutcomeB, complex1(d));
			for (int b=0; b<numOutcomeB; b++){
				for (int i=0; i<d; i++){
					randVecs[b][i] = std::complex<double>(distribution(generator), distribution(generator));
					if (fixedVals.size() > 1){
						randVecs[b][i] = std::complex<double>(fixedVals[fixedVals.size()-1], fixedVals[fixedVals.size()-1]);
						fixedVals.pop_back();
						fixedVals.pop_back();
					}
				}
			}

			// Use Gram-Schmidt to make these orthonormal
			makeOrthonormal(randVecs, normVecs);

			// Create the matrices from this basis set
			for (int b=0; b<numOutcomeB; b++){
				for (int i=0; i<d; i++){
					for (int j=0; j<d; j++){
						std::complex<double> res = normVecs[b][i]*std::conj(normVecs[b][j]);
						Br[i*d+j][x*numOutcomeB+b] = std::real(res);
						Bi[i*d+j][x*numOutcomeB+b] = std::imag(res);
					}
				}
			}

		}

	}
	
	// The sizes of the variable matrices
	auto dimRefA = monty::new_array_ptr(std::vector<int>({numOutcomeA*numMeasureA, d*d}));
	auto dimRefB = monty::new_array_ptr(std::vector<int>({d*d, numOutcomeB*numMeasureB}));

	// Create the coefficient matrix
	int nextInd = 0;
	real2 C(numMeasureA*numOutcomeA, real1(numMeasureB*numOutcomeB));
	for (int i=0; i<numMeasureB; i++){
		for (int j=i+1; j<numMeasureB; j++){
			for (int b1=0; b1<numOutcomeB; b1++){
				for (int b2=0; b2<numOutcomeB; b2++){
					C[nextInd][i*numOutcomeB+b1] = 1;
					C[nextInd][j*numOutcomeB+b2] = -1;
					C[nextInd+1][i*numOutcomeB+b1] = -1;
					C[nextInd+1][j*numOutcomeB+b2] = 1;
					nextInd += numOutcomeA;
				}
			}
		}
	}
	auto CRef = mosek::fusion::Matrix::sparse(monty::new_array_ptr(C));

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

	// Output before
	if (verbosity >= 2){
		prettyPrint("initial Br = ", Br);
		std::cout << std::endl;
		prettyPrint("initial Bi = ", Bi);
		std::cout << std::endl;
	}

	// ----------------------------
	//    Creating model A
	// ----------------------------
		
	// Create the MOSEK model 
	mosek::fusion::Model::t modelA = new mosek::fusion::Model(); 

	// The matrices to optimise
	mosek::fusion::Variable::t ArOpt = modelA->variable(dimRefA, mosek::fusion::Domain::inRange(-1.0, 1.0));
	mosek::fusion::Variable::t AiOpt = modelA->variable(dimRefA, mosek::fusion::Domain::inRange(-1.0, 1.0));

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

	// The real part should have trace one
	auto sumA = new monty::ndarray<mosek::fusion::Expression::t,1>(monty::shape(d));
	for (int i=0; i<d; i++){
		(*sumA)[i] = ArOpt->slice(columnsStartRefA[i*(d+1)], columnsEndRefA[i*(d+1)]);
	}
	modelA->constraint(mosek::fusion::Expr::add(std::shared_ptr<monty::ndarray<mosek::fusion::Expression::t,1>>(sumA)), mosek::fusion::Domain::equalsTo(rankARef));

	// The imaginary part should have zero on the diags
	for (int i=0; i<d; i++){
		modelA->constraint(AiOpt->slice(columnsStartRefA[i*(d+1)], columnsEndRefA[i*(d+1)]), mosek::fusion::Domain::equalsTo(zeroRefA));
	}

	// Symmetry constraints 
	for (int i=0; i<matchingRows.size(); i++){
		modelA->constraint(mosek::fusion::Expr::sub(ArOpt->slice(columnsStartRefA[matchingRows[i][0]], columnsEndRefA[matchingRows[i][0]]), ArOpt->slice(columnsStartRefA[matchingRows[i][1]], columnsEndRefA[matchingRows[i][1]])), mosek::fusion::Domain::equalsTo(zeroRefA));
		modelA->constraint(mosek::fusion::Expr::add(AiOpt->slice(columnsStartRefA[matchingRows[i][0]], columnsEndRefA[matchingRows[i][0]]), AiOpt->slice(columnsStartRefA[matchingRows[i][1]], columnsEndRefA[matchingRows[i][1]])), mosek::fusion::Domain::equalsTo(zeroRefA));
	}

	// ----------------------------
	//    Creating model B
	// ----------------------------
		
	// Create the MOSEK model 
	mosek::fusion::Model::t modelB = new mosek::fusion::Model(); 

	// The matrices to optimise
	mosek::fusion::Variable::t BrOpt = modelB->variable(dimRefB, mosek::fusion::Domain::inRange(-1.0, 1.0));
	mosek::fusion::Variable::t BiOpt = modelB->variable(dimRefB, mosek::fusion::Domain::inRange(-1.0, 1.0));

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

	// Trace of real should be one
	auto sumB = new monty::ndarray<mosek::fusion::Expression::t,1>(monty::shape(d));
	for (int i=0; i<d; i++){
		(*sumB)[i] = BrOpt->slice(rowsStartRefB[i*(d+1)], rowsEndRefB[i*(d+1)]);
	}
	modelB->constraint(mosek::fusion::Expr::add(std::shared_ptr<monty::ndarray<mosek::fusion::Expression::t,1>>(sumB)), mosek::fusion::Domain::equalsTo(rankBRef));

	// The imaginary part should have zero on the diags
	for (int i=0; i<d; i++){
		modelB->constraint(BiOpt->slice(rowsStartRefB[i*(d+1)], rowsEndRefB[i*(d+1)]), mosek::fusion::Domain::equalsTo(zeroRefB));
	}

	// For each set of measurements, the matrices should sum to the identity
	for (int i=0; i<columnsStartRefB.size(); i+=numOutcomeB){
		modelB->constraint(mosek::fusion::Expr::sum(BrOpt->slice(columnsStartRefB[i], columnsEndRefB[i+numOutcomeB-1]), 1), mosek::fusion::Domain::equalsTo(identityRef));
		modelB->constraint(mosek::fusion::Expr::sum(BiOpt->slice(columnsStartRefB[i], columnsEndRefB[i+numOutcomeB-1]), 1), mosek::fusion::Domain::equalsTo(zero1DRef));
	}

	// Symmetry constraints 
	for (int i=0; i<matchingRows.size(); i++){
		modelB->constraint(mosek::fusion::Expr::sub(BrOpt->slice(rowsStartRefB[matchingRows[i][0]], rowsEndRefB[matchingRows[i][0]]), BrOpt->slice(rowsStartRefB[matchingRows[i][1]], rowsEndRefB[matchingRows[i][1]])), mosek::fusion::Domain::equalsTo(zeroRefB));
		modelB->constraint(mosek::fusion::Expr::add(BiOpt->slice(rowsStartRefB[matchingRows[i][0]], rowsEndRefB[matchingRows[i][0]]), BiOpt->slice(rowsStartRefB[matchingRows[i][1]], rowsEndRefB[matchingRows[i][1]])), mosek::fusion::Domain::equalsTo(zeroRefB));
	}

	// Keep seesawing 
	double prevResult = -1;
	int iter = 0;
	int lessTol = 0;
	for (iter=0; iter<numIters; iter++){

		// ----------------------------
		//    Fixing B, optimising A
		// ----------------------------
		
		// Create references to the fixed matrix
		auto BrRef = monty::new_array_ptr(Br);
		auto BiRef = monty::new_array_ptr(Bi);

		// Set up the objective function 
		modelA->objective(mosek::fusion::ObjectiveSense::Maximize, mosek::fusion::Expr::dot(CRef, mosek::fusion::Expr::sub(mosek::fusion::Expr::mul(ArOpt, BrRef), mosek::fusion::Expr::mul(AiOpt, BiRef))));

		// Solve the SDP
		modelA->solve();
		
		// Extract the results
		try {
			finalResult = modelA->primalObjValue() / 2;
		} catch (mosek::fusion::SolutionError e){
			finalResult = -100;
			std::cout << e.what() << std::endl;
			break;
		}
		delta = exact-finalResult;
		auto tempAr = *(ArOpt->level());
		auto tempAi = *(AiOpt->level());
		int matWidthA = d*d;
		int matHeightA = numMeasureA*numOutcomeA;
		for (int i=0; i<matWidthA*matHeightA; i++){
			Ar[i/matWidthA][i%matWidthA] = tempAr[i];
			Ai[i/matWidthA][i%matWidthA] = tempAi[i];
		}

		// ----------------------------
		//    Fixing A, optimising B
		// ----------------------------

		// Create references to the fixed matrices
		auto ArRef = monty::new_array_ptr(Ar);
		auto AiRef = monty::new_array_ptr(Ai);

		// Set up the objective function
		modelB->objective(mosek::fusion::ObjectiveSense::Maximize, mosek::fusion::Expr::dot(CRef, mosek::fusion::Expr::sub(mosek::fusion::Expr::mul(ArRef, BrOpt), mosek::fusion::Expr::mul(AiRef, BiOpt))));

		// Solve the SDP
		modelB->solve();
		
		// Extract the results
		try {
			finalResult = modelB->primalObjValue() / 2;
		} catch (mosek::fusion::SolutionError e){
			finalResult = -100;
			std::cout << e.what() << std::endl;
			break;
		}
		delta = std::abs(exact-finalResult);
		fromLast = std::abs(finalResult-prevResult);
		int matHeightB = d*d;
		int matWidthB = numMeasureB*numOutcomeB;
		auto tempBr = *(BrOpt->level());
		auto tempBi = *(BiOpt->level());
		for (int i=0; i<matWidthB*matHeightB; i++){
			Br[i/matWidthB][i%matWidthB] = tempBr[i];
			Bi[i/matWidthB][i%matWidthB] = tempBi[i];
		}

		// Output after this section
		if (verbosity > 0){
			std::cout << std::fixed << std::setprecision(9) << "iter " << std::setw(4) << iter << " val " << finalResult << " (" << delta << " from ideal, " << fromLast << " from prev)" << std::endl;

		// If told to give per-iteration graphable output
		} else if (outputMethod == 3){
			std::cout << std::fixed << std::setprecision(9) << std::setw(4) << iter << " " << delta << std::endl;

		// If told to give per-iteration matrix output
		} else if (outputMethod == 6){
			std::cout << "[";
			for (int i=0; i<matWidthB*matHeightB; i++){
				std::cout << tempBr[i] << ", " << tempBi[i] << ", ";
				if (i+1 < matWidthB*matHeightB){
					std::cout << ", ";
				}
			}
			std::cout << "]" << std::endl;
		}

		// If the change is less than the tolerance, increase the count
		if (fromLast < tol){
			lessTol += 1;
		} else {
			lessTol = 0;
		}

		// Once it's been less than the tolerance enough in a row
		if (lessTol > numInRowRequired || delta < tol){
			break;
		}

		// The new is now the old
		prevResult = finalResult;

	}

	// Extract the results from the A matrix
	complex4 A(numMeasureA, complex3(numOutcomeA, complex2(d, complex1(d))));
	for (int x=0; x<numMeasureA; x++){
		for (int a=0; a<numOutcomeA; a++){
			for (int i=0; i<d*d; i++){
				A[x][a][i/d][i%d] = Ar[x*numOutcomeA+a][i] + im*Ai[x*numOutcomeA+a][i];
			}
			if (outputMethod == 1 && verbosity >= 1){
				std::cout << std::endl;
				prettyPrint("A[" + std::to_string(x) + "][" + std::to_string(a) + "] = ", A[x][a]);
				std::cout << trace(A[x][a]) << std::endl;
			}
		}
	}
	
	// Extract the results from the B matrix
	complex4 B(numMeasureB, complex3(numOutcomeB, complex2(d, complex1(d))));
	for (int y=0; y<numMeasureB; y++){
		for (int b=0; b<numOutcomeB; b++){
			for (int i=0; i<d*d; i++){
				B[y][b][i/d][i%d] = Br[i][y*numOutcomeB+b] + im*Bi[i][y*numOutcomeB+b];
			}
			if (outputMethod == 1 && verbosity >= 1){
				std::cout << std::endl;
				prettyPrint("B[" + std::to_string(y) + "][" + std::to_string(b) + "] = ", B[y][b]);
			}
		}
	}

	// For each combination of measurements
	for (int y1=0; y1<numMeasureB; y1++){
		for (int y2=y1+1; y2<numMeasureB; y2++){

			// Calculate the moment matrix between them
			complex2 momentB(numOutcomeB, complex1(numOutcomeB));
			for (int b1=0; b1<numOutcomeB; b1++){
				for (int b2=0; b2<numOutcomeB; b2++){
					for (int i=0; i<d; i++){
						for (int j=0; j<d; j++){
							momentB[b1][b2] += B[y1][b1][i][j]*B[y2][b2][j][i];
						}
					}
				}
			}

			// Output if allowed
			if (outputMethod == 1 && verbosity >= 1){
				std::cout << std::endl;
				prettyPrint("moments " + std::to_string(y1) + " " + std::to_string(y2) + " = ", momentB);
			}

		}
	}

	// Stop the timer 
	auto t2 = std::chrono::high_resolution_clock::now();

	// Output after
	exact = d*numPerm*sqrt(d*(d-1));
	delta = exact-finalResult;
	if (outputMethod == 2){
		std::cout << std::fixed << std::setprecision(9) << delta << std::endl;
	} else if (outputMethod == 7){
		if (d == 2 && n == 4) {
			std::cout << std::fixed << std::setprecision(9) << ((16.726162159 - finalResult) < 1e-5) << std::endl;
		} else {
			std::cout << std::fixed << std::setprecision(9) << (delta < 1e-5) << std::endl;
		}
	} else if (outputMethod == 4){
		std::cout << std::fixed << iter << std::endl;
	} else if (outputMethod == 5){
		std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << std::endl;
	} else if (outputMethod == 1){
		std::cout << std::setprecision(5) << "iterations = " << iter << std::endl;
		std::cout << std::setprecision(5) << "time elapsed = " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << " ms" << std::endl;
		std::cout << std::setprecision(5) << "final result = " << finalResult << " <= " << exact << " (" << delta << " away)" << std::endl;
	}

	// Prevent memory leaks
	modelA->dispose();
	modelB->dispose();

}

// Standard cpp entry point
int main (int argc, char ** argv) {

	// Whether to use the extension for d > 2
	int method = 2;

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
			std::cout << "  Program that uses a seesaw" << std::endl;
			std::cout << "    of a Bell-scenario to" << std::endl;
			std::cout << "       check for MUBs" << std::endl;
			std::cout << "------------------------------" << std::endl;
			std::cout << "                        " << std::endl;
			std::cout << "      main options          " << std::endl;
			std::cout << "-h               show the help" << std::endl;
			std::cout << "-d [int]         set the dimension" << std::endl;
			std::cout << "-n [int]         set the number of measurements" << std::endl;
			std::cout << "                        " << std::endl;
			std::cout << "    tolerance options          " << std::endl;
			std::cout << "-t [dbl]         set the convergence threshold" << std::endl;
			std::cout << "-i [int]         set the iteration limit" << std::endl;
			std::cout << "                        " << std::endl;
			std::cout << "   initialisation options          " << std::endl;
			std::cout << "-k               use a known maximal solution if possible" << std::endl;
			std::cout << "-r               start completely random without G-S" << std::endl;
			std::cout << "-s [str]         set the random seed" << std::endl;
			std::cout << "-f 2*[int] [dbl] set part of the initial array" << std::endl;
			std::cout << "                        " << std::endl;
			std::cout << "      output options          " << std::endl;
			std::cout << "-p [int]         set the output precision" << std::endl;
			std::cout << "-v               verbose output" << std::endl;
			std::cout << "-Z               only output if it reached zero within the tolerance" << std::endl;
			std::cout << "-D               only output the difference to the ideal" << std::endl;
			std::cout << "-I               output for graphing the difference vs iteration" << std::endl;
			std::cout << "-N               only output the number of iterations" << std::endl;
			std::cout << "-T               only output the time taken" << std::endl;
			std::cout << "-M               only output the flattened matrices" << std::endl;
			std::cout << "" << std::endl;
			return 0;

		// Set the number of measurements for Alice
		} else if (arg == "-n") {
			n = std::stoi(argv[i+1]);
			i += 1;

		// Only output the delta
		} else if (arg == "-I") {
			outputMethod = 3;
			verbosity = 0;

		// Only output the time taken
		} else if (arg == "-T") {
			outputMethod = 5;
			verbosity = 0;

		// Only output the time taken
		} else if (arg == "-M") {
			outputMethod = 6;
			verbosity = 0;

		// Use a known ideal if possible
		} else if (arg == "-k") {
			useIdeal = true;

		// Only output the number of iterations required
		} else if (arg == "-N") {
			outputMethod = 4;
			verbosity = 0;

		// If told not to use the G-S method
		} else if (arg == "-r") {
			randomMethod = 1;

		// Only output the delta
		} else if (arg == "-D") {
			outputMethod = 2;
			verbosity = 0;

		// Only output whether it reached zero within the tolerance
		} else if (arg == "-Z") {
			outputMethod = 7;
			verbosity = 0;

		// Set the seed
		} else if (arg == "-s") {
			seed = std::string(argv[i+1]);
			i += 1;

		// Add to the list of fixed values
		} else if (arg == "-f") {
			fixedVals.push_back(std::stod(argv[i+1]));
			i += 1;

		// Set the dimension
		} else if (arg == "-d") {
			d = std::stoi(argv[i+1]);
			i += 1;

		// Set the iteration limit
		} else if (arg == "-i") {
			numIters = std::stoi(argv[i+1]);
			i += 1;

		// Set the verbosity
		} else if (arg == "-v") {
			verbosity = 2;

		// Set the tolerance
		} else if (arg == "-t") {
			tol = std::stod(argv[i+1]);
			epsilon = std::stod(argv[i+1]);
			i += 1;

		// Set the precision
		} else if (arg == "-p") {
			precision = std::stod(argv[i+1]);
			i += 1;

		// DEBUG parameter
		} else if (arg == "-db") {
			numRects = std::stod(argv[i+1]);
			i += 1;

		// Otherwise it's an error
		} else {
			std::cout << "ERROR - unknown argument: \"" << arg << "\"" << std::endl;
			return 1;
		}

	}

	// Perform the seesaw
	seesawExtended(d, n);

}

