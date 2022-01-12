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

// Eigen
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/../unsupported/Eigen/KroneckerProduct>

// MOSEK
#include "fusion.h"

// MPI parallelisation
#include "mpi.h"

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

// Keeping track of MPI things
int procID = 0;
int numProcs = 1;

// Turn an Eigen matrix to a std::vector
complex2 matToVec(Eigen::MatrixXcd mat){
	complex2 data;
	for (int i=0; i<mat.rows(); i++){
		Eigen::VectorXcd row = mat.row(i);
		data.push_back(complex1(row.data(), row.data() + mat.cols()));
	}
	return data;
}

// Turn a std::vector into an Eigen matrix
Eigen::MatrixXcd vecToMat(complex2 data){
	Eigen::MatrixXcd mat(data.size(), data[0].size());
	for (int i=0; i<data.size(); i++){
		mat.row(i) = Eigen::VectorXcd::Map(&data[i][0], data[i].size());
	}
	return mat;
}

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

// Pretty print a generic 2D dense Eigen array
template <typename type>
void prettyPrint(std::string pre, Eigen::Matrix<type, -1, -1> arr){

	// Used fixed precision
	std::cout << std::fixed << std::showpos << std::setprecision(precision);

	// Loop over the array
	std::string rowText;
	for (int y=0; y<arr.rows(); y++){

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
		for (int x=0; x<arr.cols(); x++){
			std::cout << std::setw(5) << arr(y,x);
			if (x < arr.cols()-1){
				std::cout << ", ";
			}
		}

		// Output the row
		std::cout << "}";
		if (y < arr.rows()-1){
			std::cout << ",";
		}
		std::cout << std::endl;

	}

	// Reset things for normal output
	std::cout << std::noshowpos;

}

// Pretty print a generic 2D sparse Eigen array
template <typename type>
void prettyPrint(std::string pre, Eigen::SparseMatrix<type> arr){

	// Extract the dense array and then call the routine as normal
	prettyPrint(pre, Eigen::Matrix<type,-1,-1>(arr));

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

// Use the jointly-constrained bilinear SDP method
// https://arxiv.org/pdf/1808.03182.pdf
void JCB(int d, int n){

	// How many permutations required
	int numPerm = n*(n-1)/2;

	// How many measurements/outcomes for each party
	int numMeasureA = d*d*numPerm;
	int numOutcomeA = 3;
	int numMeasureB = n;
	int numOutcomeB = d;

	// The width/height of the X matrix
	int p = d * numMeasureA * numOutcomeA;

	// The width/height of the Y matrix
	int q = d * numMeasureB * numOutcomeB;

	// Define some useful quantities
	int K = std::min(p*p, q*q);

	// Amount to remove from each to make consistent with the paper
	double sub = sqrt(d*(d-1))*numPerm;

	// The known ideal if there exists MUBs for this problem
	double idealScaled = numPerm*sqrt(d*(d-1));
	double idealRaw = -(idealScaled + sub) * d;

	// Useful constants for constructing the bases
	double oneOverSqrt2 = 1.0 / sqrt(2.0);
	std::complex<double> imagOverSqrt2 = im / sqrt(2.0);

	// Start the timer 
	auto t1 = std::chrono::high_resolution_clock::now();

	// Output basic info
	if (procID == 0){
		std::cout << std::fixed << std::setprecision(precision);
		std::cout << "d = " << d << "  n = " << n << std::endl;
		std::cout << "p = " << p << "  p*p = " << p*p << std::endl;
		std::cout << "q = " << q << "  q*q = " << q*q << std::endl;
		std::cout << "idealRaw = " << idealRaw << std::endl;
		std::cout << "idealScaled = " << idealScaled << std::endl;
	}

	// Assemble eta to be orthonormal and self-adjoint
	if (procID == 0){
		std::cout << "Constructing eta..." << std::endl;
	}
	std::vector<std::vector<Eigen::Triplet<std::complex<double>>>> tripletsEta(p*p, std::vector<Eigen::Triplet<std::complex<double>>>(2));
	std::vector<Eigen::SparseMatrix<std::complex<double>>> etaSparse(p*p);
	int next = 0;
	for (int i=0; i<p; i++){
		for (int j=i; j<p; j++){

			// For the off-diags
			if (i != j){

				// Self-adjoint 1's
				tripletsEta[next][0] = Eigen::Triplet<std::complex<double>>(i, j, oneOverSqrt2);
				tripletsEta[next][1] = Eigen::Triplet<std::complex<double>>(j, i, oneOverSqrt2);
				next += 1;

				// Self-adjoint i's
				tripletsEta[next][0] = Eigen::Triplet<std::complex<double>>(i, j, imagOverSqrt2);
				tripletsEta[next][1] = Eigen::Triplet<std::complex<double>>(j, i, -imagOverSqrt2);
				next += 1;

			// For the diags
			} else {
				tripletsEta[next][0] = Eigen::Triplet<std::complex<double>>(i, i, 1.0);
				next += 1;

			}

		}
	}
	for (int i=0; i<p*p; i++){
		etaSparse[i] = Eigen::SparseMatrix<std::complex<double>>(p, p);
		etaSparse[i].setFromTriplets(tripletsEta[i].begin(), tripletsEta[i].end());
	}

	// Assemble xi to be orthonormal and self-adjoint
	if (procID == 0){
		std::cout << "Constructing xi..." << std::endl;
	}
	std::vector<std::vector<Eigen::Triplet<std::complex<double>>>> tripletsXi(q*q, std::vector<Eigen::Triplet<std::complex<double>>>(2));
	std::vector<Eigen::SparseMatrix<std::complex<double>>> xiSparse(q*q);
	next = 0;
	for (int i=0; i<q; i++){
		for (int j=i; j<q; j++){

			// For the off-diags
			if (i != j){

				// Self-adjoint 1's
				tripletsXi[next][0] = Eigen::Triplet<std::complex<double>>(i, j, oneOverSqrt2);
				tripletsXi[next][1] = Eigen::Triplet<std::complex<double>>(j, i, oneOverSqrt2);
				next += 1;

				// Self-adjoint i's
				tripletsXi[next][0] = Eigen::Triplet<std::complex<double>>(i, j, imagOverSqrt2);
				tripletsXi[next][1] = Eigen::Triplet<std::complex<double>>(j, i, -imagOverSqrt2);
				next += 1;

			// For the diags
			} else {
				tripletsXi[next][0] = Eigen::Triplet<std::complex<double>>(i, i, 1.0);
				next += 1;

			}

		}
	}
	for (int i=0; i<q*q; i++){
		xiSparse[i] = Eigen::SparseMatrix<std::complex<double>>(q, q);
		xiSparse[i].setFromTriplets(tripletsXi[i].begin(), tripletsXi[i].end());
	}

	// The coefficients for the block-diagonals of Q
	if (procID == 0){
		std::cout << "Constructing Q..." << std::endl;
	}
	real1 blockQ(numMeasureA*numOutcomeA*numMeasureB*numOutcomeB);
	int nextInd = 0;
	int perB = numMeasureB*numOutcomeB;
	for (int i=0; i<numMeasureB; i++){
		for (int j=i+1; j<numMeasureB; j++){
			for (int b1=0; b1<numOutcomeB; b1++){
				for (int b2=0; b2<numOutcomeB; b2++){
					blockQ[nextInd+i*numOutcomeB+b1] = 1;
					blockQ[nextInd+j*numOutcomeB+b2] = -1;
					blockQ[nextInd+perB+i*numOutcomeB+b1] = -1;
					blockQ[nextInd+perB+j*numOutcomeB+b2] = 1;
					nextInd += numOutcomeA*perB;
				}
			}
		}
	}

	// Construct Q
	std::vector<Eigen::Triplet<std::complex<double>>> tripletsQ;
	Eigen::SparseMatrix<std::complex<double>> QSparse(p*q, p*q);
	int numB = numMeasureB*numOutcomeB;
	for (int i=0; i<blockQ.size(); i++){
		int topLeftLoc = std::floor(i/numB)*numB*d*d + (i%numB)*d;
		for (int j=0; j<d; j++){
			for (int k=0; k<d; k++){
				tripletsQ.push_back(Eigen::Triplet<std::complex<double>>(topLeftLoc+j*numB*d+j, topLeftLoc+k*numB*d+k, -blockQ[i]));
			}
		}
	}
	QSparse.setFromTriplets(tripletsQ.begin(), tripletsQ.end());

	// Calculate U_{j,k} = tr(Q(eta_j (x) xi_k))
	Eigen::SparseMatrix<double> USparse(p*p, q*q);
	std::vector<Eigen::Triplet<double>> tripletsU;
	if (procID == 0){
		std::cout << "Calculating U from Q... 0%" << std::flush;
	}
	Eigen::SparseMatrix<std::complex<double>> result(p*q, p*q);
	double val;
	for (int j=0; j<p*p; j++){
		if (procID == 0){
			std::cout << "\rCalculating U from Q... " << (100*j) / (p*p) << "%" << std::flush;
		}
		for (int k=0; k<q*q; k++){
			Eigen::KroneckerProductSparse<Eigen::SparseMatrix<std::complex<double>>, Eigen::SparseMatrix<std::complex<double>>> prod(etaSparse[j], xiSparse[k]);
			prod.evalTo(result);
			val = std::real(QSparse.cwiseProduct(result).sum());
			if (std::abs(val) > zeroThresh){
				tripletsU.push_back(Eigen::Triplet<double>(j, k, val));
			}	
		}
	}
	USparse.setFromTriplets(tripletsU.begin(), tripletsU.end());
	if (procID == 0){
		std::cout << std::endl;
	}
	
	// Pre-calculate the decomposition U = S Delta T
	if (procID == 0){
		std::cout << "Calculating decomposition U=S*Delta*T..." << std::endl;
	}
	Eigen::BDCSVD<Eigen::MatrixXd> svd(USparse, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Matrix<double, -1, -1> Delta = svd.singularValues();
	Eigen::SparseMatrix<double> S = svd.matrixU().sparseView();
	Eigen::SparseMatrix<double> T = svd.matrixV().sparseView();

	// Clear some memory
	svd = Eigen::BDCSVD<Eigen::MatrixXd>();

	// For each non-zero S
	if (procID == 0){
		std::cout << "Precalculating S*eta..." << std::endl;
	}
	std::vector<Eigen::SparseMatrix<std::complex<double>>> SEta(p*p, Eigen::SparseMatrix<std::complex<double>>(p,p));
	for (int j1=0; j1<S.outerSize(); ++j1){
		for (Eigen::SparseMatrix<double>::InnerIterator it(S, j1); it; ++it){
			SEta[it.col()] += it.value()*etaSparse[it.row()];
		}
	}

	// Turn it into MOSEK form
	std::vector<mosek::fusion::Matrix::t> SEtarRef(p*p);
	std::vector<mosek::fusion::Matrix::t> SEtaiRef(p*p);
	for (int j=0; j<p*p; j++){

		// Get the lists of locations and values
		int1 nonZeroRows;
		int1 nonZeroCols;
		real1 nonZeroValsr;
		real1 nonZeroValsi;
		for (int i1=0; i1<SEta[j].outerSize(); ++i1){
			for (Eigen::SparseMatrix<std::complex<double>>::InnerIterator it(SEta[j], i1); it; ++it){
				int j = it.row();
				int i = it.col();
				if (j < (std::floor(i / d) * d + d) && j >= std::floor(i / d) * d){
					nonZeroRows.push_back(j % d);
					nonZeroCols.push_back(i);
					nonZeroValsr.push_back(std::real(it.value()));
					nonZeroValsi.push_back(std::imag(it.value()));
				}
			}
		}

		// Stop it from being completely "empty"
		if (nonZeroRows.size() == 0){
			nonZeroRows.push_back(0);
			nonZeroCols.push_back(0);
			nonZeroValsr.push_back(0);
			nonZeroValsi.push_back(0);
		}

		// Make the sparse matrix from this data
		SEtarRef[j] = mosek::fusion::Matrix::sparse(d, p, monty::new_array_ptr(nonZeroRows), monty::new_array_ptr(nonZeroCols), monty::new_array_ptr(nonZeroValsr));
		SEtaiRef[j] = mosek::fusion::Matrix::sparse(d, p, monty::new_array_ptr(nonZeroRows), monty::new_array_ptr(nonZeroCols), monty::new_array_ptr(nonZeroValsi));

	}

	// For each non-zero T
	if (procID == 0){
		std::cout << "Precalculating T*Xi..." << std::endl;
	}
	std::vector<Eigen::SparseMatrix<std::complex<double>>> TXi(q*q, Eigen::SparseMatrix<std::complex<double>>(q,q));
	for (int j1=0; j1<T.outerSize(); ++j1){
		for (Eigen::SparseMatrix<double>::InnerIterator it(T, j1); it; ++it){
			TXi[it.col()] += it.value()*xiSparse[it.row()];
		}
	}

	// Turn it into MOSEK form
	std::vector<mosek::fusion::Matrix::t> TXirRef(q*q);
	std::vector<mosek::fusion::Matrix::t> TXiiRef(q*q);
	for (int j=0; j<q*q; j++){

		// Get the lists of locations and values
		int1 nonZeroRows;
		int1 nonZeroCols;
		real1 nonZeroValsr;
		real1 nonZeroValsi;
		for (int i1=0; i1<TXi[j].outerSize(); ++i1){
			for (Eigen::SparseMatrix<std::complex<double>>::InnerIterator it(TXi[j], i1); it; ++it){
				int j = it.row();
				int i = it.col();
				if (j < (std::floor(i / d) * d + d) && j >= std::floor(i / d) * d){
					nonZeroRows.push_back(j % d);
					nonZeroCols.push_back(i);
					nonZeroValsr.push_back(std::real(it.value()));
					nonZeroValsi.push_back(std::imag(it.value()));
				}
			}
		}

		// Make the sparse matrix from this data
		TXirRef[j] = mosek::fusion::Matrix::sparse(d, q, monty::new_array_ptr(nonZeroRows), monty::new_array_ptr(nonZeroCols), monty::new_array_ptr(nonZeroValsr));
		TXiiRef[j] = mosek::fusion::Matrix::sparse(d, q, monty::new_array_ptr(nonZeroRows), monty::new_array_ptr(nonZeroCols), monty::new_array_ptr(nonZeroValsi));

	}

	// Turn it into MOSEK form
	int1 nonZeroRows;
	int1 nonZeroCols;
	real1 nonZeroValsr;
	real1 nonZeroValsi;
	for (int j=0; j<p*p; j++){

		// For the non-zero elements of SEta
		for (int i1=0; i1<SEta[j].outerSize(); ++i1){
			for (Eigen::SparseMatrix<std::complex<double>>::InnerIterator it(SEta[j], i1); it; ++it){

				// Even then, only the dxd diagonal blocks
				int j = it.row();
				int i = it.col();
				if (j < (std::floor(i / d) * d + d) && j >= std::floor(i / d) * d){
					nonZeroRows.push_back(j);
					nonZeroCols.push_back((j % d)*p + i);
					nonZeroValsr.push_back(std::real(it.value()));
					nonZeroValsi.push_back(std::imag(it.value()));
				}

			}
		}

	}

	// Make the sparse matrix from this data
	mosek::fusion::Matrix::t XToxr = mosek::fusion::Matrix::sparse(p*p, p*d, monty::new_array_ptr(nonZeroRows), monty::new_array_ptr(nonZeroCols), monty::new_array_ptr(nonZeroValsr));
	mosek::fusion::Matrix::t XToxi = mosek::fusion::Matrix::sparse(p*p, p*d, monty::new_array_ptr(nonZeroRows), monty::new_array_ptr(nonZeroCols), monty::new_array_ptr(nonZeroValsi));

	// Reference to the zero/identity matrix for MOSEK
	real2 zero(d, real1(d, 0.0));
	real2 identity(d, real1(d, 0.0));
	for (int i=0; i<d; i++){
		identity[i][i] = 1.0;
	}
	auto identityRef = monty::new_array_ptr(identity);
	auto zeroRef = monty::new_array_ptr(zero);

	// Dimensions of X and Y for MOSEK
	auto dimXRef = monty::new_array_ptr(std::vector<int>({d, p}));
	auto dimYRef = monty::new_array_ptr(std::vector<int>({d, q}));

	// Locations of the start/end of each section for MOSEK
	std::vector<std::shared_ptr<monty::ndarray<int,1>>> startX;
	std::vector<std::shared_ptr<monty::ndarray<int,1>>> endX;
	for (int i=0; i<p; i+=d){
		startX.push_back(monty::new_array_ptr(std::vector<int>({0, i})));
		endX.push_back(monty::new_array_ptr(std::vector<int>({d, i+d})));
	}
	std::vector<std::shared_ptr<monty::ndarray<int,1>>> startY;
	std::vector<std::shared_ptr<monty::ndarray<int,1>>> endY;
	for (int i=0; i<q; i+=d){
		startY.push_back(monty::new_array_ptr(std::vector<int>({0, i})));
		endY.push_back(monty::new_array_ptr(std::vector<int>({d, i+d})));
	}

	// Exact x solution for d2n2
	complex3 XTest;
	//XTest.push_back({ { +1.0+0.0i , +0.0-0.0i },
					  //{ +0.0+0.0i , +0.0+0.0i } });
	//XTest.push_back({ { +0.0+0.0i , +0.0-0.0i },
					  //{ +0.0+0.0i , +1.0+0.0i } });
	//XTest.push_back({ { +0.0+0.0i , +0.0-0.0i },
					  //{ +0.0+0.0i , +0.0+0.0i } });
	//XTest.push_back({ { +0.5+0.0i , +0.5-0.0i },
					  //{ +0.5+0.0i , +0.5+0.0i } });
	//XTest.push_back({ { +0.5+0.0i , -0.5-0.0i },
					  //{ -0.5+0.0i , +0.5+0.0i } });
	//XTest.push_back({ { +0.0+0.0i , +0.0-0.0i },
					  //{ +0.0+0.0i , +0.0+0.0i } });
	//XTest.push_back({ { +0.052836114238185+0.000000000000000i , +0.219222663560252+0.044563248399554i },
		//{ +0.219222663560252-0.044563248399554i , +0.947163885761815+0.000000000000000i } });
	//XTest.push_back({ { +0.947163885639723+0.000000000000000i , -0.219222663560252-0.044563248399554i },
		//{ -0.219222663560252+0.044563248399554i , +0.052836114360277+0.000000000000000i } });
	//XTest.push_back({ { +0.000000000122091+0.000000000000000i , -0.000000000000000-0.000000000000000i },
		//{ -0.000000000000000+0.000000000000000i , -0.000000000122091+0.000000000000000i } });
	//XTest.push_back({ { +0.540036237537646+0.000000000000000i , -0.019577020193051+0.498009879301547i },
		//{ -0.019577020193051-0.498009879301547i , +0.459963762462354+0.000000000000000i } });
	//XTest.push_back({ { +0.459963762340263+0.000000000000000i , +0.019577020193051-0.498009879301547i },
		//{ +0.019577020193051+0.498009879301547i , +0.540036237659737+0.000000000000000i } });
	//XTest.push_back({ { +0.000000000122091+0.000000000000000i , +0.000000000000000+0.000000000000000i },
		//{ +0.000000000000000-0.000000000000000i , -0.000000000122091+0.000000000000000i } });
	//XTest.push_back({ { +0.459963762340177+0.000000000000000i , +0.019577020193042-0.498009879301540i },
		//{ +0.019577020193042+0.498009879301540i , +0.540036237659823+0.000000000000000i } });
	//XTest.push_back({ { +0.540036237537731+0.000000000000000i , -0.019577020193042+0.498009879301540i },
		//{ -0.019577020193042-0.498009879301540i , +0.459963762462269+0.000000000000000i } });
	//XTest.push_back({ { +0.000000000122092+0.000000000000000i , +0.000000000000000+0.000000000000000i },
		//{ +0.000000000000000-0.000000000000000i , -0.000000000122092+0.000000000000000i } });
	//XTest.push_back({ { +0.947163885639720+0.000000000000000i , -0.219222663560255-0.044563248399572i },
		//{ -0.219222663560255+0.044563248399572i , +0.052836114360280+0.000000000000000i } });
	//XTest.push_back({ { +0.052836114238189+0.000000000000000i , +0.219222663560255+0.044563248399572i },
		//{ +0.219222663560255-0.044563248399572i , +0.947163885761811+0.000000000000000i } });
	//XTest.push_back({ { +0.000000000122091+0.000000000000000i , -0.000000000000000-0.000000000000000i },
		//{ -0.000000000000000+0.000000000000000i , -0.000000000122091+0.000000000000000i } });

	// Exact y solution for d2n2
	complex3 YTest;
	if (useIdeal){

		if (d == 2 && n == 2){
			YTest.push_back({ { +1.0+0.0i , +0.0-0.0i },
							  { +0.0+0.0i , +0.0+0.0i } });
			YTest.push_back({ { +0.0+0.0i , +0.0-0.0i },
							  { +0.0+0.0i , +1.0+0.0i } });
			YTest.push_back({ { +0.5+0.0i , +0.5-0.0i },
							  { +0.5+0.0i , +0.5+0.0i } });
			YTest.push_back({ { +0.5+0.0i , -0.5-0.0i },
							  { -0.5+0.0i , +0.5+0.0i } });

		} else if (d == 2 && n == 3){

			YTest.push_back({ { +0.045868809+0.000000000i , -0.180704330-0.105407813i },
            { -0.180704330+0.105407813i , +0.954131191+0.000000000i } });
			YTest.push_back({ { +0.954131191+0.000000000i , +0.180704330+0.105407813i },
            { +0.180704330-0.105407813i , +0.045868809+0.000000000i } });
			YTest.push_back({ { +0.591608307+0.000000000i , -0.398272764+0.288074163i },
            { -0.398272764-0.288074163i , +0.408391693+0.000000000i } });
			YTest.push_back({ { +0.408391693+0.000000000i , +0.398272764-0.288074163i },
            { +0.398272764+0.288074163i , +0.591608307+0.000000000i } });
			YTest.push_back({ { +0.311930722+0.000000000i , +0.242325510+0.394852243i },
            { +0.242325510-0.394852243i , +0.688069278+0.000000000i } });
			YTest.push_back({ { +0.688069278+0.000000000i , -0.242325510-0.394852243i },
            { -0.242325510+0.394852243i , +0.311930722+0.000000000i } });

		} else if (d == 3 && n == 2){

			YTest.push_back({ { +0.155411412+0.000000000i , +0.070472369-0.298607747i , -0.149654327+0.121364526i },
            { +0.070472369+0.298607747i , +0.605701587+0.000000000i , -0.301051778-0.232512496i },
            { -0.149654327-0.121364526i , -0.301051778+0.232512496i , +0.238886994+0.000000000i } });
			YTest.push_back({ { +0.001377017+0.000000000i , -0.001711225+0.020141348i , -0.020897768+0.023017519i },
            { -0.001711225-0.020141348i , +0.296729029+0.000000000i , +0.362641191+0.277062589i },
            { -0.020897768-0.023017519i , +0.362641191-0.277062589i , +0.701893947+0.000000000i } });
			YTest.push_back({ { +0.843211565+0.000000000i , -0.068761144+0.278466399i , +0.170552095-0.144382044i },
            { -0.068761144-0.278466399i , +0.097569377+0.000000000i , -0.061589413-0.044550092i },
            { +0.170552095+0.144382044i , -0.061589413+0.044550092i , +0.059219052+0.000000000i } });
			YTest.push_back({ { +0.099150226+0.000000000i , -0.004650478+0.113922638i , +0.021751797-0.275402114i },
            { -0.004650478-0.113922638i , +0.131114119+0.000000000i , -0.317454566-0.012075319i },
            { +0.021751797+0.275402114i , -0.317454566+0.012075319i , +0.769735648+0.000000000i } });
			YTest.push_back({ { +0.385723058+0.000000000i , -0.426014142+0.023847338i , -0.186971031+0.141159027i },
            { -0.426014142-0.023847338i , +0.471988226+0.000000000i , +0.215228433-0.144344444i },
            { -0.186971031-0.141159027i , +0.215228433+0.144344444i , +0.142288709+0.000000000i } });
			YTest.push_back({ { +0.515126709+0.000000000i , +0.430664620-0.137769975i , +0.165219234+0.134243087i },
            { +0.430664620+0.137769975i , +0.396897648+0.000000000i , +0.102226133+0.156419763i },
            { +0.165219234-0.134243087i , +0.102226133-0.156419763i , +0.087975637+0.000000000i } });

		}

	}
	//YTest.push_back({ { +0.212125948106518+0.000000000000000i , +0.141173539811072-0.383664648148024i },
		//{ +0.141173539811072+0.383664648148024i , +0.787874051893482+0.000000000000000i } });
	//YTest.push_back({ { +0.787874051893482+0.000000000000000i , -0.141173539811072+0.383664648148024i },
		//{ -0.141173539811072-0.383664648148024i , +0.212125948106518+0.000000000000000i } });
	//YTest.push_back({ { +0.844496548846062+0.000000000000000i , -0.168858496036799-0.320638323300013i },
		//{ -0.168858496036799+0.320638323300013i , +0.155503451153938+0.000000000000000i } });
	//YTest.push_back({ { +0.155503451153938+0.000000000000000i , +0.168858496036799+0.320638323300013i },
		//{ +0.168858496036799-0.320638323300013i , +0.844496548846062+0.000000000000000i } });

	// The initial hyperrectangle
	hyperRect D(p, q);

	// Create the MOSEK model for x
	mosek::fusion::Model::t lModel = new mosek::fusion::Model(); 

	// Only use one core
	lModel->setSolverParam("numThreads", 1);

	// The matrices to optimise
	mosek::fusion::Variable::t XrOptL = lModel->variable(dimXRef, mosek::fusion::Domain::inRange(-1.0, 1.0));
	mosek::fusion::Variable::t XiOptL = lModel->variable(dimXRef, mosek::fusion::Domain::inRange(-1.0, 1.0));

	// The parameters which are the only thing changing
	mosek::fusion::Parameter::t lParamr = lModel->parameter(dimXRef);
	mosek::fusion::Parameter::t lParami = lModel->parameter(dimXRef);

	// For each dxd section of X
	for (int i=0; i<startX.size(); i++){

		// Section needs to be semidefinite
		lModel->constraint(mosek::fusion::Expr::vstack(
								mosek::fusion::Expr::hstack(
									XrOptL->slice(startX[i], endX[i]), 
									mosek::fusion::Expr::neg(XiOptL->slice(startX[i], endX[i]))
								), 
								mosek::fusion::Expr::hstack(
									XiOptL->slice(startX[i], endX[i]),
									XrOptL->slice(startX[i], endX[i]) 
								)
						   ), mosek::fusion::Domain::inPSDCone(2*d));

		// Real is symmetric, imag is anti-symmetric
		lModel->constraint(mosek::fusion::Expr::sub(XrOptL->slice(startX[i], endX[i]), mosek::fusion::Expr::transpose(XrOptL->slice(startX[i], endX[i]))), mosek::fusion::Domain::equalsTo(zeroRef));
		lModel->constraint(mosek::fusion::Expr::add(XiOptL->slice(startX[i], endX[i]), mosek::fusion::Expr::transpose(XiOptL->slice(startX[i], endX[i]))), mosek::fusion::Domain::equalsTo(zeroRef));

		// And have trace 1 (apart from the third measure)
		if ((i+1) % 3 != 0){
			lModel->constraint(mosek::fusion::Expr::sum(XrOptL->slice(startX[i], endX[i])->diag()), mosek::fusion::Domain::equalsTo(1.0));
		}

	}

	// Need to sum to the identity
	for (int i=0; i<startX.size(); i+=numOutcomeA){
		auto prods = new monty::ndarray<mosek::fusion::Expression::t,1>(monty::shape(numOutcomeA));
		auto prods2 = new monty::ndarray<mosek::fusion::Expression::t,1>(monty::shape(numOutcomeA));
		for(int j=0; j<numOutcomeA; j++){
			(*prods)[j] = XrOptL->slice(startX[i+j], endX[i+j]);
			(*prods2)[j] = XiOptL->slice(startX[i+j], endX[i+j]);
		}
		lModel->constraint(mosek::fusion::Expr::add(std::shared_ptr<monty::ndarray<mosek::fusion::Expression::t,1>>(prods)), mosek::fusion::Domain::equalsTo(identityRef));
		lModel->constraint(mosek::fusion::Expr::add(std::shared_ptr<monty::ndarray<mosek::fusion::Expression::t,1>>(prods2)), mosek::fusion::Domain::equalsTo(zeroRef));
	}

	// Setup the objective function
	mosek::fusion::Expression::t objectiveExprL = mosek::fusion::Expr::sub(mosek::fusion::Expr::dot(XrOptL, lParamr), mosek::fusion::Expr::dot(XiOptL, lParami));

	// The objective function should be real
	//lModel->constraint(mosek::fusion::Expr::add(mosek::fusion::Expr::dot(XrOptL, lParami), mosek::fusion::Expr::dot(XiOptL, lParamr)), mosek::fusion::Domain::equalsTo(0.0));

	// Exact x solution for d2n2
	for (int i=0; i<XTest.size(); i++){
		int ind = i*d;
		for (int j=0; j<d; j++){
			for (int k=0; k<d; k++){
				if (j != d && k != d){
					lModel->constraint(XrOptL->index(j, ind+k), mosek::fusion::Domain::equalsTo(std::real(XTest[i][j][k])));
					lModel->constraint(XiOptL->index(j, ind+k), mosek::fusion::Domain::equalsTo(std::imag(XTest[i][j][k])));
				}
			}
		}
	}

	// Create the MOSEK model for y
	mosek::fusion::Model::t mModel = new mosek::fusion::Model(); 

	// Only use one core
	mModel->setSolverParam("numThreads", 1);

	// The matrices to optimise
	mosek::fusion::Variable::t YrOptM = mModel->variable(dimYRef, mosek::fusion::Domain::inRange(-1.0, 1.0));
	mosek::fusion::Variable::t YiOptM = mModel->variable(dimYRef, mosek::fusion::Domain::inRange(-1.0, 1.0));

	// The parameters which are the only thing changing
	mosek::fusion::Parameter::t mParamr = mModel->parameter(dimYRef);
	mosek::fusion::Parameter::t mParami = mModel->parameter(dimYRef);

	// For each dxd section of Y
	for (int i=0; i<startY.size(); i++){

		// Section needs to be semidefinite
		mModel->constraint(mosek::fusion::Expr::vstack(
								mosek::fusion::Expr::hstack(
									YrOptM->slice(startY[i], endY[i]), 
									mosek::fusion::Expr::neg(YiOptM->slice(startY[i], endY[i]))
								), 
								mosek::fusion::Expr::hstack(
									YiOptM->slice(startY[i], endY[i]),
									YrOptM->slice(startY[i], endY[i]) 
								)
						   ), mosek::fusion::Domain::inPSDCone(2*d));

		// Real is symmetric, imag is anti-symmetric
		mModel->constraint(mosek::fusion::Expr::sub(YrOptM->slice(startY[i], endY[i]), mosek::fusion::Expr::transpose(YrOptM->slice(startY[i], endY[i]))), mosek::fusion::Domain::equalsTo(zeroRef));
		mModel->constraint(mosek::fusion::Expr::add(YiOptM->slice(startY[i], endY[i]), mosek::fusion::Expr::transpose(YiOptM->slice(startY[i], endY[i]))), mosek::fusion::Domain::equalsTo(zeroRef));

		// And have trace 1
		mModel->constraint(mosek::fusion::Expr::sum(YrOptM->slice(startY[i], endY[i])->diag()), mosek::fusion::Domain::equalsTo(1));

	}

	// Need to sum to the identity
	for (int i=0; i<startY.size(); i+=numOutcomeB){
		auto prods = new monty::ndarray<mosek::fusion::Expression::t,1>(monty::shape(numOutcomeB));
		auto prods2 = new monty::ndarray<mosek::fusion::Expression::t,1>(monty::shape(numOutcomeB));
		for(int j=0; j<numOutcomeB; j++){
			(*prods)[j] = YrOptM->slice(startY[i+j], endY[i+j]);
			(*prods2)[j] = YiOptM->slice(startY[i+j], endY[i+j]);
		}
		mModel->constraint(mosek::fusion::Expr::add(std::shared_ptr<monty::ndarray<mosek::fusion::Expression::t,1>>(prods)), mosek::fusion::Domain::equalsTo(identityRef));
		mModel->constraint(mosek::fusion::Expr::add(std::shared_ptr<monty::ndarray<mosek::fusion::Expression::t,1>>(prods2)), mosek::fusion::Domain::equalsTo(zeroRef));
	}
	
	// Exact y solution for d2n2
	for (int i=0; i<YTest.size(); i++){
		int ind = i*d;
		for (int j=0; j<d; j++){
			for (int k=0; k<d; k++){
				if (j != d && k != d){
					mModel->constraint(YrOptM->index(j, ind+k), mosek::fusion::Domain::equalsTo(std::real(YTest[i][j][k])));
					mModel->constraint(YiOptM->index(j, ind+k), mosek::fusion::Domain::equalsTo(std::imag(YTest[i][j][k])));
				}
			}
		}
	}

	// Setup the objective function
	mosek::fusion::Expression::t objectiveExprM = mosek::fusion::Expr::sub(mosek::fusion::Expr::dot(YrOptM, mParamr), mosek::fusion::Expr::dot(YiOptM, mParami));

	// The objective function should be real
	//mModel->constraint(mosek::fusion::Expr::add(mosek::fusion::Expr::dot(YrOptM, mParami), mosek::fusion::Expr::dot(YiOptM, mParamr)), mosek::fusion::Domain::equalsTo(0.0));

	// Number of cores needs to divide nice
	if (p*p % numProcs != 0){
		std::cerr << "Error - number of cores should divide p*p" << std::endl;
	}
	if (q*q % numProcs != 0){
		std::cerr << "Error - number of cores should divide q*q" << std::endl;
	}
	
	// Get the X sections 
	if (procID == 0){
		std::cout << "Bounding hyperrect for X... 0%" << std::flush;
	}
	int pPerCore = (p*p) / numProcs;
	real1 locall(pPerCore);
	real1 localL(pPerCore);
	for (int j=procID*pPerCore; j<(procID+1)*pPerCore; j++){

		// Progress indicator
		if (procID == 0){
			std::cout << "\rBounding hyperrect for X... " << (100*j) / (pPerCore) << "%" << std::flush;
		}

		// Set the param for this j
		lParamr->setValue(SEtarRef[j]->getDataAsArray());
		lParami->setValue(SEtaiRef[j]->getDataAsArray());

		// Minimise the object function
		lModel->objective(mosek::fusion::ObjectiveSense::Minimize, objectiveExprL);
		lModel->solve();
		locall[j%pPerCore] = lModel->primalObjValue();

		// Maximise the object function
		lModel->objective(mosek::fusion::ObjectiveSense::Maximize, objectiveExprL);
		lModel->solve();
		localL[j%pPerCore] = lModel->primalObjValue();

	}
	if (procID == 0){
		std::cout << std::endl;
	}

	// Sync the results between cores
	MPI_Allgather(&locall[0], pPerCore, MPI_DOUBLE, &D.l[0], pPerCore, MPI_DOUBLE, MPI_COMM_WORLD);
	MPI_Allgather(&localL[0], pPerCore, MPI_DOUBLE, &D.L[0], pPerCore, MPI_DOUBLE, MPI_COMM_WORLD);

	// Get the Y sections
	if (procID == 0){
		std::cout << "Bounding hyperrect for Y... 0%" << std::flush;
	}
	int qPerCore = (q*q) / numProcs;
	real1 localm(qPerCore);
	real1 localM(qPerCore);
	for (int k=procID*qPerCore; k<(procID+1)*qPerCore; k++){

		// Progress indicator
		if (procID == 0){
			std::cout << "\rBounding hyperrect for Y... " << (100*k) / (q*q) << "%" << std::flush;
		}

		// Set the param for this k
		mParamr->setValue(TXirRef[k]->getDataAsArray());
		mParami->setValue(TXiiRef[k]->getDataAsArray());

		// Minimise the object function
		mModel->objective(mosek::fusion::ObjectiveSense::Minimize, objectiveExprM);
		mModel->solve();
		localm[k%qPerCore] = mModel->primalObjValue();

		// Extract the Y values just to see
		if (procID == 0 && verbosity >= 2){
			auto tempYr = *(YrOptM->level());
			auto tempYi = *(YiOptM->level());
			complex2 Y(d, complex1(q));
			for (int i=0; i<q*d; i++){
				Y[i/q][i%q] = tempYr[i] + im*tempYi[i];
			}
			prettyPrint("Y after rect m = ", Y);
			std::cout << std::endl;
		}

		// Maximise the object function
		mModel->objective(mosek::fusion::ObjectiveSense::Maximize, objectiveExprM);
		mModel->solve();
		localM[k%qPerCore] = mModel->primalObjValue();

	}
	if (procID == 0){
		std::cout << std::endl;
	}

	// Sync the results between cores
	MPI_Allgather(&localm[0], qPerCore, MPI_DOUBLE, &D.m[0], qPerCore, MPI_DOUBLE, MPI_COMM_WORLD);
	MPI_Allgather(&localM[0], qPerCore, MPI_DOUBLE, &D.M[0], qPerCore, MPI_DOUBLE, MPI_COMM_WORLD);

	// Prevent memory leaks
	lModel->dispose();
	mModel->dispose();

	// Output various things
	if (procID == 0 && verbosity >= 2){
		std::cout << std::endl;
		prettyPrint("initial hyperrect", D);
		std::cout << std::endl;
	}

	// Setup the MOSEK model
	mosek::fusion::Model::t model = new mosek::fusion::Model(); 

	// Only use one core
	model->setSolverParam("numThreads", 1);

	// The matrices to optimise
	mosek::fusion::Variable::t XrOpt = model->variable(dimXRef, mosek::fusion::Domain::inRange(-1.0, 1.0));
	mosek::fusion::Variable::t XiOpt = model->variable(dimXRef, mosek::fusion::Domain::inRange(-1.0, 1.0));
	mosek::fusion::Variable::t YrOpt = model->variable(dimYRef, mosek::fusion::Domain::inRange(-1.0, 1.0));
	mosek::fusion::Variable::t YiOpt = model->variable(dimYRef, mosek::fusion::Domain::inRange(-1.0, 1.0));
	mosek::fusion::Variable::t rOpt = model->variable(K, mosek::fusion::Domain::unbounded());

	// The parameter values
	std::vector<mosek::fusion::Parameter::t> lParams(p*p);
	std::vector<mosek::fusion::Parameter::t> LParams(p*p);
	for (int i=0; i<p*p; i++){
		lParams[i] = model->parameter();
		LParams[i] = model->parameter();
	}
	std::vector<mosek::fusion::Parameter::t> mParams(q*q);
	std::vector<mosek::fusion::Parameter::t> MParams(q*q);
	for (int i=0; i<q*q; i++){
		mParams[i] = model->parameter();
		MParams[i] = model->parameter();
	}
	std::vector<mosek::fusion::Parameter::t> GmParams(K);
	std::vector<mosek::fusion::Parameter::t> GMParams(K);
	std::vector<mosek::fusion::Parameter::t> HlParams(K);
	std::vector<mosek::fusion::Parameter::t> HLParams(K);
	std::vector<mosek::fusion::Parameter::t> slmParams(K);
	std::vector<mosek::fusion::Parameter::t> sLMParams(K);
	for (int i=0; i<K; i++){
		HlParams[i] = model->parameter();
		HLParams[i] = model->parameter();
		GmParams[i] = model->parameter();
		GMParams[i] = model->parameter();
		slmParams[i] = model->parameter();
		sLMParams[i] = model->parameter();
	}

	// Exact x solution for d2n2
	for (int i=0; i<XTest.size(); i++){
		int ind = i*d;
		for (int j=0; j<d; j++){
			for (int k=0; k<d; k++){
				if (j != d && k != d){
					model->constraint(XrOpt->index(j, ind+k), mosek::fusion::Domain::equalsTo(std::real(XTest[i][j][k])));
					model->constraint(XiOpt->index(j, ind+k), mosek::fusion::Domain::equalsTo(std::imag(XTest[i][j][k])));
				}
			}
		}
	}

	// Exact y solution for d2n2
	for (int i=0; i<YTest.size(); i++){
		int ind = i*d;
		for (int j=0; j<d; j++){
			for (int k=0; k<d; k++){
				if (j != d && k != d){
					model->constraint(YrOpt->index(j, ind+k), mosek::fusion::Domain::equalsTo(std::real(YTest[i][j][k])));
					model->constraint(YiOpt->index(j, ind+k), mosek::fusion::Domain::equalsTo(std::imag(YTest[i][j][k])));
				}
			}
		}
	}

	// Objective function 
	model->objective(mosek::fusion::ObjectiveSense::Minimize, mosek::fusion::Expr::sum(rOpt));
	
	// Sections of X need to be semidefinite
	for (int i=0; i<startX.size(); i++){
		model->constraint(mosek::fusion::Expr::vstack(
								mosek::fusion::Expr::hstack(
									XrOpt->slice(startX[i], endX[i]), 
									mosek::fusion::Expr::neg(XiOpt->slice(startX[i], endX[i]))
								), 
								mosek::fusion::Expr::hstack(
									XiOpt->slice(startX[i], endX[i]),
									XrOpt->slice(startX[i], endX[i]) 
								)
						   ), mosek::fusion::Domain::inPSDCone(2*d));

		// And have trace 1 (apart from the third measure)
		if ((i+1) % 3 != 0){
			model->constraint(mosek::fusion::Expr::sum(XrOpt->slice(startX[i], endX[i])->diag()), mosek::fusion::Domain::equalsTo(1));
		}

		// Real is symmetric, imag is anti-symmetric
		model->constraint(mosek::fusion::Expr::sub(XrOpt->slice(startX[i], endX[i]), mosek::fusion::Expr::transpose(XrOpt->slice(startX[i], endX[i]))), mosek::fusion::Domain::equalsTo(zeroRef));
		model->constraint(mosek::fusion::Expr::add(XiOpt->slice(startX[i], endX[i]), mosek::fusion::Expr::transpose(XiOpt->slice(startX[i], endX[i]))), mosek::fusion::Domain::equalsTo(zeroRef));

	}

	// Sections of Y need to be semidefinite
	for (int i=0; i<startY.size(); i++){
		model->constraint(mosek::fusion::Expr::vstack(
								mosek::fusion::Expr::hstack(
									YrOpt->slice(startY[i], endY[i]), 
									mosek::fusion::Expr::neg(YiOpt->slice(startY[i], endY[i]))
								), 
								mosek::fusion::Expr::hstack(
									YiOpt->slice(startY[i], endY[i]),
									YrOpt->slice(startY[i], endY[i]) 
								)
						   ), mosek::fusion::Domain::inPSDCone(2*d));

		// And have trace 1
		model->constraint(mosek::fusion::Expr::sum(YrOpt->slice(startY[i], endY[i])->diag()), mosek::fusion::Domain::equalsTo(1));
		
		// Real is symmetric, imag is anti-symmetric
		model->constraint(mosek::fusion::Expr::sub(YrOpt->slice(startY[i], endY[i]), mosek::fusion::Expr::transpose(YrOpt->slice(startY[i], endY[i]))), mosek::fusion::Domain::equalsTo(zeroRef));
		model->constraint(mosek::fusion::Expr::add(YiOpt->slice(startY[i], endY[i]), mosek::fusion::Expr::transpose(YiOpt->slice(startY[i], endY[i]))), mosek::fusion::Domain::equalsTo(zeroRef));

	}

	// X needs to sum to the identity
	for (int i=0; i<startX.size(); i+=numOutcomeA){
		auto prods = new monty::ndarray<mosek::fusion::Expression::t,1>(monty::shape(numOutcomeA));
		auto prods2 = new monty::ndarray<mosek::fusion::Expression::t,1>(monty::shape(numOutcomeA));
		for(int j=0; j<numOutcomeA; j++){
			(*prods)[j] = XrOpt->slice(startX[i+j], endX[i+j]);
			(*prods2)[j] = XiOpt->slice(startX[i+j], endX[i+j]);
		}
		model->constraint(mosek::fusion::Expr::add(std::shared_ptr<monty::ndarray<mosek::fusion::Expression::t,1>>(prods)), mosek::fusion::Domain::equalsTo(identityRef));
		model->constraint(mosek::fusion::Expr::add(std::shared_ptr<monty::ndarray<mosek::fusion::Expression::t,1>>(prods2)), mosek::fusion::Domain::equalsTo(zeroRef));
	}

	// Y needs to sum to the identity
	for (int i=0; i<startY.size(); i+=numOutcomeB){
		auto prods = new monty::ndarray<mosek::fusion::Expression::t,1>(monty::shape(numOutcomeB));
		auto prods2 = new monty::ndarray<mosek::fusion::Expression::t,1>(monty::shape(numOutcomeB));
		for(int j=0; j<numOutcomeB; j++){
			(*prods)[j] = YrOpt->slice(startY[i+j], endY[i+j]);
			(*prods2)[j] = YiOpt->slice(startY[i+j], endY[i+j]);
		}
		model->constraint(mosek::fusion::Expr::add(std::shared_ptr<monty::ndarray<mosek::fusion::Expression::t,1>>(prods)), mosek::fusion::Domain::equalsTo(identityRef));
		model->constraint(mosek::fusion::Expr::add(std::shared_ptr<monty::ndarray<mosek::fusion::Expression::t,1>>(prods2)), mosek::fusion::Domain::equalsTo(zeroRef));
	}

	// Bound x by the hyperrectangle
	for (int j=0; j<p*p; j++){
		model->constraint(mosek::fusion::Expr::sub(mosek::fusion::Expr::sub(mosek::fusion::Expr::dot(XrOpt, SEtarRef[j]), mosek::fusion::Expr::dot(XiOpt, SEtaiRef[j])), lParams[j]), mosek::fusion::Domain::greaterThan(0.0));
		model->constraint(mosek::fusion::Expr::sub(mosek::fusion::Expr::sub(mosek::fusion::Expr::dot(XrOpt, SEtarRef[j]), mosek::fusion::Expr::dot(XiOpt, SEtaiRef[j])), LParams[j]), mosek::fusion::Domain::lessThan(0.0));
		//model->constraint(mosek::fusion::Expr::add(mosek::fusion::Expr::dot(XiOpt, SEtarRef[j]), mosek::fusion::Expr::dot(XrOpt, SEtaiRef[j])), mosek::fusion::Domain::equalsTo(0.0));
	}
	
	// Bound y by the hyperrectangle
	for (int j=0; j<q*q; j++){
		model->constraint(mosek::fusion::Expr::sub(mosek::fusion::Expr::sub(mosek::fusion::Expr::dot(YrOpt, TXirRef[j]), mosek::fusion::Expr::dot(YiOpt, TXiiRef[j])), mParams[j]), mosek::fusion::Domain::greaterThan(0.0));
		model->constraint(mosek::fusion::Expr::sub(mosek::fusion::Expr::sub(mosek::fusion::Expr::dot(YrOpt, TXirRef[j]), mosek::fusion::Expr::dot(YiOpt, TXiiRef[j])), MParams[j]), mosek::fusion::Domain::lessThan(0.0));
		//model->constraint(mosek::fusion::Expr::add(mosek::fusion::Expr::dot(YiOpt, TXirRef[j]), mosek::fusion::Expr::dot(YrOpt, TXiiRef[j])), mosek::fusion::Domain::equalsTo(0.0));
	}

	// Combined constraint with r
	for (int j=0; j<K; j++){

		std::cout << j << " " << SEtarRef[j]->numRows() << " x " << SEtarRef[j]->numColumns() << " " << SEtarRef[j]->numNonzeros() << std::endl;
		std::cout << j << " " << SEtaiRef[j]->numRows() << " x " << SEtaiRef[j]->numColumns() << " " << SEtaiRef[j]->numNonzeros() << std::endl;

		// For l and m
		model->constraint(
							mosek::fusion::Expr::sub(
								mosek::fusion::Expr::sub(
									mosek::fusion::Expr::add(
										mosek::fusion::Expr::mul(
											GmParams[j], 
											mosek::fusion::Expr::sub(
												mosek::fusion::Expr::dot(XrOpt, SEtarRef[j]), 
												mosek::fusion::Expr::dot(XiOpt, SEtaiRef[j])
											)
										),
										mosek::fusion::Expr::mul(
											HlParams[j], 
											mosek::fusion::Expr::sub(
												mosek::fusion::Expr::dot(YrOpt, TXirRef[j]), 
												mosek::fusion::Expr::dot(YiOpt, TXiiRef[j]) 
											)
										)
									),
									rOpt->index(j)
								),
								slmParams[j]
							), 
							mosek::fusion::Domain::lessThan(0.0)
						);
		
		// For L and M
		//model->constraint(
							//mosek::fusion::Expr::sub(
								//mosek::fusion::Expr::sub(
									//mosek::fusion::Expr::add(
										//mosek::fusion::Expr::mul(
											//GMParams[j], 
											//mosek::fusion::Expr::sub(
												//mosek::fusion::Expr::dot(XrOpt, SEtarRef[j]), 
												//mosek::fusion::Expr::dot(XiOpt, SEtaiRef[j])
											//)
										//),
										//mosek::fusion::Expr::mul(
											//HLParams[j], 
											//mosek::fusion::Expr::sub(
												//mosek::fusion::Expr::dot(YrOpt, TXirRef[j]), 
												//mosek::fusion::Expr::dot(YiOpt, TXiiRef[j]) 
											//)
										//)
									//),
									//rOpt->index(j)
								//),
								//sLMParams[j]
							//), 
							//mosek::fusion::Domain::lessThan(0.0)
						//);

	}

	// Get the initial value for the upper/lower bounds
	if (procID == 0){
		std::cout << "Calculating initial bounds..." << std::endl;
	}

	// Set the parameters
	for (int i=0; i<p*p; i++){
		lParams[i]->setValue(D.l[i]);
		LParams[i]->setValue(D.L[i]);
	}
	for (int i=0; i<q*q; i++){
		mParams[i]->setValue(D.m[i]);
		MParams[i]->setValue(D.M[i]);
	}
	for (int i=0; i<K; i++){
		HlParams[i]->setValue(std::real(Delta(i,0)*D.l[i]));
		HLParams[i]->setValue(std::real(Delta(i,0)*D.L[i]));
		GmParams[i]->setValue(std::real(Delta(i,0)*D.m[i]));
		GMParams[i]->setValue(std::real(Delta(i,0)*D.M[i]));
		slmParams[i]->setValue(std::real(Delta(i,0)*D.l[i]*D.m[i]));
		sLMParams[i]->setValue(std::real(Delta(i,0)*D.L[i]*D.M[i]));
	}

	// Solve 
	model->solve();

	// Init things here to prevent re-init each iterations
	hyperRect Omega(p, q);
	std::vector<hyperRect> localRects(numRects, hyperRect(p, q));
	std::vector<hyperRect> globalRects(numRects*numProcs, hyperRect(p, q));
	real1 localLowers(numRects, 0.0);
	real1 localUppers(numRects, 0.0);
	real1 globalLowers(numRects*numProcs, 0.0);
	real1 globalUppers(numRects*numProcs, 0.0);
	real2 xs(numRects, real1(p*p));
	real2 ys(numRects, real1(q*q));
	int newLoc = -1;

	// Init things here to prevent re-init each iterations
	// Extract the data
	localLowers[0] = model->primalObjValue();
	auto tempXr = *(XrOpt->level());
	auto tempXi = *(XiOpt->level());
	auto tempYr = *(YrOpt->level());
	auto tempYi = *(YiOpt->level());
	auto tempR = *(rOpt->level());
	real2 Xr(d, real1(p));
	real2 Xi(d, real1(p));
	real2 Yr(d, real1(q));
	real2 Yi(d, real1(q));
	for (int i=0; i<p*d; i++){
		Xr[i/p][i%p] = tempXr[i];
		Xi[i/p][i%p] = tempXi[i];
	}
	for (int i=0; i<q*d; i++){
		Yr[i/q][i%q] = tempYr[i];
		Yi[i/q][i%q] = tempYi[i];
	}

	// Convert X to x
	for (int i=0; i<p*p; i++){
		xs[0][i] = inner(Xr, SEtarRef[i]) - inner(Xi, SEtaiRef[i]);
	}

	// Convert Y to y
	for (int i=0; i<q*q; i++){
		ys[0][i] = inner(Yr, TXirRef[i]) - inner(Yi, TXiiRef[i]);
	}

	// Calculate the upper bound from these
	localUppers[0] = 0;
	for (int j=0; j<K; j++){
		localUppers[0] += std::real(Delta(j,0)*xs[0][j]*ys[0][j]);
	}

	// Verbose output
	if (procID == 0 && verbosity >= 2){

		// Also get the complex X and Y
		complex2 X(d, complex1(p));
		for (int i=0; i<d; i++){
			for (int j=0; j<p; j++){
				X[i][j] = Xr[i][j] + im*Xi[i][j];
			}
		}
		complex2 Y(d, complex1(q));
		for (int i=0; i<d; i++){
			for (int j=0; j<q; j++){
				Y[i][j] = Yr[i][j] + im*Yi[i][j];
			}
		}

		// Matrix outputs
		std::cout << std::endl;
		prettyPrint("X = ", X);
		std::cout << std::endl;
		prettyPrint("Y = ", Y);
		std::cout << std::endl;
		prettyPrint("x = ", xs[0]);
		std::cout << std::endl;
		prettyPrint("y = ", ys[0]);
		std::cout << std::endl;

	}

	// Output initial bounds
	if (procID == 0){
		std::cout << "Raw bounds: " << localLowers[0] << " < raw < " << localUppers[0] << std::endl;
		std::cout << "Scaled bounds: " << -localUppers[0]/d-sub  << " < scaled < " << -localLowers[0]/d-sub << std::endl;
	}

	// Keep track of the remaining hyperrects and their bounds
	std::vector<hyperRect> P = {D};
	std::vector<double> lowerBounds = {localLowers[0]};
	double bestLowerBound = localLowers[0];
	double bestUpperBound = localUppers[0];

	// Create enough branches for the number of cores
	while (P.size() < numProcs){

		// Branch the first element
		localRects = branchHyperrectangle(p, q, P[0]);

		// Remove this rect 
		P.erase(P.begin(), P.begin()+1);
		lowerBounds.erase(lowerBounds.begin(), lowerBounds.begin()+1);

		// Add these new rects to the list
		for (int j=0; j<numRects; j++){
			P.insert(P.begin(), localRects[j]);
			lowerBounds.insert(lowerBounds.begin(), -1000000);
		}

	}

	// Keep looping until any of stopping criteria are met
	int iter = 0;
	bool shouldStop = false;
	while (!shouldStop){

		// Create the new hyperrectangles
		for (int i=0; i<numProcs; i++){
			localRects = branchHyperrectangle(p, q, P[i]);
			for (int j=0; j<numRects; j++){
				globalRects[i*numRects+j] = localRects[j];
			}
		}

		// Remove the rects used to create them
		P.erase(P.begin(), P.begin()+numProcs);
		lowerBounds.erase(lowerBounds.begin(), lowerBounds.begin()+numProcs);

		// This core should only do part the work
		for (int j=0; j<numRects; j++){
			 localRects[j] = globalRects[procID*numRects+j];
		}

		// For each of this core's hyperrects
		for (int j=0; j<numRects; j++){

			// Set the parameters
			for (int i=0; i<p*p; i++){
				lParams[i]->setValue(localRects[j].l[i]);
				LParams[i]->setValue(localRects[j].L[i]);
			}
			for (int i=0; i<q*q; i++){
				mParams[i]->setValue(localRects[j].m[i]);
				MParams[i]->setValue(localRects[j].M[i]);
			}
			for (int i=0; i<K; i++){
				HlParams[i]->setValue(std::real(Delta(i,0)*localRects[j].l[i]));
				HLParams[i]->setValue(std::real(Delta(i,0)*localRects[j].L[i]));
				GmParams[i]->setValue(std::real(Delta(i,0)*localRects[j].m[i]));
				GMParams[i]->setValue(std::real(Delta(i,0)*localRects[j].M[i]));
				slmParams[i]->setValue(std::real(Delta(i,0)*localRects[j].l[i]*localRects[j].m[i]));
				sLMParams[i]->setValue(std::real(Delta(i,0)*localRects[j].L[i]*localRects[j].M[i]));
			}

			// Solve 
			model->solve();

			// Extract the data
			try {
				localLowers[j] = model->primalObjValue();
			} catch (mosek::fusion::SolutionError e){
				localLowers[j] = 10000;
				localUppers[j] = 10000;
				continue;
			}
			auto tempXr = *(XrOpt->level());
			auto tempXi = *(XiOpt->level());
			auto tempYr = *(YrOpt->level());
			auto tempYi = *(YiOpt->level());
			for (int i=0; i<p*d; i++){
				Xr[i/p][i%p] = tempXr[i];
				Xi[i/p][i%p] = tempXi[i];
			}
			for (int i=0; i<q*d; i++){
				Yr[i/q][i%q] = tempYr[i];
				Yi[i/q][i%q] = tempYi[i];
			}

			// Convert X to x
			for (int i=0; i<p*p; i++){
				xs[j][i] = inner(Xr, SEtarRef[i]) - inner(Xi, SEtaiRef[i]);
			}

			// Convert Y to y
			for (int i=0; i<q*q; i++){
				ys[j][i] = inner(Yr, TXirRef[i]) - inner(Yi, TXiiRef[i]);
			}

			// Calculate the upper bound from these
			localUppers[j] = 0;
			for (int i=0; i<K; i++){
				localUppers[j] += std::real(Delta(i,0)*xs[j][i]*ys[j][i]);
			}

			// Verbose output
			if (procID == 0 && verbosity >= 2){

				// Output the hyperrect
				prettyPrint("hyperrect " + std::to_string(j), localRects[j]);

				// Also get the complex X and Y
				complex2 X(d, complex1(p));
				for (int i=0; i<d; i++){
					for (int j=0; j<p; j++){
						X[i][j] = Xr[i][j] + im*Xi[i][j];
					}
				}
				complex2 Y(d, complex1(q));
				for (int i=0; i<d; i++){
					for (int j=0; j<q; j++){
						Y[i][j] = Yr[i][j] + im*Yi[i][j];
					}
				}

				// Matrix outputs
				std::cout << std::endl;
				prettyPrint("X = ", X);
				std::cout << std::endl;
				prettyPrint("Y = ", Y);
				std::cout << std::endl;
				prettyPrint("x = ", xs[j]);
				std::cout << std::endl;
				prettyPrint("y = ", ys[j]);
				std::cout << std::endl;

			}

		}

		// Collect all the results
		MPI_Allgather(&localLowers[0], numRects, MPI_DOUBLE, &globalLowers[0], numRects, MPI_DOUBLE, MPI_COMM_WORLD);
		MPI_Allgather(&localUppers[0], numRects, MPI_DOUBLE, &globalUppers[0], numRects, MPI_DOUBLE, MPI_COMM_WORLD);

		// Iteration header
		if (procID == 0){
			std::cout << "-------------------------------------" << std::endl;
			std::cout << "        Iteration: " << iter << std::endl;
			std::cout << "-------------------------------------" << std::endl;

			// For each of the results, output the bounds
			for (int j=0; j<numRects*numProcs; j++){
				std::cout << "For hyperrect " << j << ": " << globalLowers[j] << " " << globalUppers[j] << ", volume = " << globalRects[j].getVolume() << std::endl;
			}

		}

		// For each of the results
		for (int j=0; j<numRects*numProcs; j++){

			// Is it the new best upper?
			if (globalUppers[j] < bestUpperBound){
				bestUpperBound = globalUppers[j];
			}

			// If the lower bound is a valid overall lower bound
			if (globalLowers[j] <= bestUpperBound){

				// Figure out where in the queue it should go
				newLoc = lowerBounds.size();
				for (int i=0; i<lowerBounds.size(); i++){
					if (globalLowers[j] < lowerBounds[i]){
						newLoc = i;
						break;
					}
				}

				// Place it into the queue
				P.insert(P.begin()+newLoc, globalRects[j]);
				lowerBounds.insert(lowerBounds.begin()+newLoc, globalLowers[j]);

			}

		}

		// The lowest bound of the whole set
		if (lowerBounds.size() > 0){
			bestLowerBound = lowerBounds[0];
		}

		// Output the best so far
		if (procID == 0){
			std::cout << "Raw bounds: " << bestLowerBound << " < raw < " << bestUpperBound << std::endl;
			std::cout << "Scaled bounds: " << -bestUpperBound/d-sub  << " < scaled < " << -bestLowerBound/d-sub << std::endl;
			std::cout << "Size of space: " << P.size() << std::endl;
		}

		// Sync whether all cores should now stop
		shouldStop = bestUpperBound - bestLowerBound < epsilon || P.size() == 0 || iter > numIters;

		// Iteration finished
		iter += 1;

	}

	// Stop the timer 
	auto t2 = std::chrono::high_resolution_clock::now();

	// Return the best
	if (procID == 0){
		std::cout << "-------------------------------------" << std::endl;
		std::cout << "    Final results" << std::endl;
		std::cout << "-------------------------------------" << std::endl;
		std::cout << "Raw bounds: " << bestLowerBound << " < raw < " << bestUpperBound << std::endl;
		std::cout << "Scaled bounds: " << -bestUpperBound/d-sub  << " < scaled < " << -bestLowerBound/d-sub << std::endl;
		std::cout << "Iterations required: " << iter << std::endl;
		std::cout << "Time required: " << std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count() << " s" << std::endl;
	}

}

// Perform the seesaw method to optimise both A and B 
void seesawExtended(int d, int n){

	// Start the timer 
	auto t1 = std::chrono::high_resolution_clock::now();

	// How many permutations required
	int numPerm = n*(n-1)/2;

	// The inequality value to eventually return
	double finalResult = 0;
	double exact = numPerm*sqrt(d*(d-1));
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

	// If told to use the ideal TODO
	if (useIdeal && d == 2 && n == 4) {

		// Mate's tetrahedron vectors
		double o = 1 / std::sqrt(3);
		complex2 idealBlochs = {{o, o, o}, {o, -o, -o}, {-o, o, -o}, {-o, -o, o}};

		// Useful definitions
		Eigen::MatrixXcd PauliX = Eigen::MatrixXcd::Zero(2, 2);
		PauliX(0, 1) = 1;
		PauliX(1, 0) = 1;
		Eigen::MatrixXcd PauliY = Eigen::MatrixXcd::Zero(2, 2);
		PauliY(0, 1) = -1i;
		PauliY(1, 0) = 1i;
		Eigen::MatrixXcd PauliZ = Eigen::MatrixXcd::Zero(2, 2);
		PauliZ(0, 0) = 1;
		PauliZ(1, 1) = -1;
		Eigen::MatrixXcd id = Eigen::MatrixXcd::Identity(2, 2);

		// Use these to generate the ideal B's
		Eigen::MatrixXcd B = Eigen::MatrixXcd::Zero(2, 2);
		int index = 0;
		for (int i=0; i<idealBlochs.size(); i++) {
			B = 0.5*(id + idealBlochs[i][0]*PauliX + idealBlochs[i][1]*PauliY + idealBlochs[i][2]*PauliZ);
			for (int j=0; j<d; j++){
				for (int k=0; k<d; k++){
					Br[j*d+k][index] = std::real(B(j,k));
					Bi[j*d+k][index] = std::imag(B(j,k));
				}
			}
			index += 1;
			B = id - B;
			for (int j=0; j<d; j++){
				for (int k=0; k<d; k++){
					Br[j*d+k][index] = std::real(B(j,k));
					Bi[j*d+k][index] = std::imag(B(j,k));
				}
			}
			index += 1;
		}

	// If told to just generate a random symmetric matrix
	} else if (randomMethod == 1){

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

	prettyPrint("C = ", C); // TODO

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
		prettyPrint("before Br = ", Br);
		std::cout << std::endl;
		prettyPrint("before Bi = ", Bi);
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
			finalResult = modelA->primalObjValue() / d - sub;
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
			finalResult = modelB->primalObjValue() / d - sub;
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

	// Check sections of objective TODO
	std::complex<double> tot = 0;
	int ind = 0;
	for (int y1=0; y1<numMeasureB; y1++){
		for (int y2=y1+1; y2<numMeasureB; y2++){
			for (int b1=0; b1<numOutcomeB; b1++){
				for (int b2=0; b2<numOutcomeB; b2++){
					Eigen::MatrixXcd BMinus = vecToMat(B[y1][b1]) - vecToMat(B[y2][b2]);
					Eigen::MatrixXcd AMinus = vecToMat(A[ind][0]) - vecToMat(A[ind][1]);
					std::complex<double> val = (AMinus.transpose()*BMinus).trace();
					tot += val;
					std::cout << val << std::endl;
					ind++;
				}
			}
		}
	}
	std::cout << tot / std::complex<double>(2*d) << std::endl;

	// Stop the timer 
	auto t2 = std::chrono::high_resolution_clock::now();

	// Output after
	exact = numPerm*sqrt(d*(d-1));
	delta = exact-finalResult;
	if (outputMethod == 2){
		std::cout << std::fixed << std::setprecision(9) << delta << std::endl;
	} else if (outputMethod == 7){
		std::cout << std::fixed << std::setprecision(9) << (delta < 1e-5) << std::endl;
	} else if (outputMethod == 4){
		std::cout << std::fixed << iter << std::endl;
	} else if (outputMethod == 5){
		std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << std::endl;
	} else if (outputMethod == 1){
		std::cout << std::setprecision(5) << "finished in " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << " ms" << std::endl;
		std::cout << std::setprecision(5) << "final result = " << finalResult << " <= " << exact << " (" << delta << " away)" << std::endl;
	}

	// Prevent memory leaks
	modelA->dispose();
	modelB->dispose();

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
		prettyPrint("C = ", C);
		std::cout << std::endl;
		prettyPrint("Ar =", Ar);
		std::cout << std::endl;
		prettyPrint("Ai =", Ai);
		std::cout << std::endl;
	}

	// The rank for A to try
	restrictRankA = true;
	restrictRankB = false;
	real2 rankA = real2(numOutcomeA*numMeasureA, real1(1, d/2.0));
	real2 rankB = real2(1, real1(numOutcomeB*numMeasureB, d/2.0));

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
		if (verbosity > 0){
			std::cout << std::fixed << std::setprecision(5) << "iter " << std::setw(3) << iter << " after B opt " << finalResult << std::endl;
		}

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
		if (verbosity > 0){
			std::cout << std::fixed << std::setprecision(5) << "iter " << std::setw(3) << iter << " after A opt " << finalResult << std::endl;
		}

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
		prettyPrint("Ar =", Ar);
		std::cout << std::endl;
		prettyPrint("Ai =", Ai);
		std::cout << std::endl;
		prettyPrint("Br =", Br);
		std::cout << std::endl;
		prettyPrint("Bi =", Bi);
		std::cout << std::endl;
	}

	// Extract the results from the A matrix
	complex4 A(numMeasureA, complex3(numOutcomeA, complex2(d, complex1(d))));
	for (int x=0; x<numMeasureA; x++){
		for (int a=0; a<numOutcomeA; a++){
			for (int i=0; i<d*d; i++){
				A[x][a][i/d][i%d] = Ar[x*numOutcomeA+a][i] + im*Ai[x*numOutcomeA+a][i];
			}
			if (verbosity > 2){
				prettyPrint("A[" + std::to_string(x) + "][" + std::to_string(a) + "] = ", A[x][a]);
				std::cout << std::endl;
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
			if (verbosity >= 2){
				prettyPrint("B[" + std::to_string(y) + "][" + std::to_string(b) + "] = ", B[y][b]);
				std::cout << std::endl;
			}
		}
	}

	// Output the results
	std::cout << std::setprecision(5) << "final result = " << finalResult << " <= " << (numMeasureB/2)*2*root2 << std::endl;

}

// Standard cpp entry point
int main (int argc, char ** argv) {

	// Start MPI and get info
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &procID);

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
			std::cout << "     method options          " << std::endl;
			std::cout << "-c               use the CHSH method" << std::endl;
			std::cout << "-e               use the extended method" << std::endl;
			std::cout << "-j               use the JCB method" << std::endl;
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

		// Use the standard method
		} else if (arg == "-c") {
			method = 1;

		// Use the JCB method
		} else if (arg == "-j") {
			method = 3;

		// Use the extended method
		} else if (arg == "-e") {
			method = 2;

		// Otherwise it's an error
		} else {
			std::cout << "ERROR - unknown argument: \"" << arg << "\"" << std::endl;
			return 1;
		}

	}

	// Perform the seesaw
	if (method == 2){
		seesawExtended(d, n);
	} else if (method == 3){
		JCB(d, n);
	} else {
		seesaw(d, n);
	}

	// Finish with MPI
	MPI_Finalize();

}

