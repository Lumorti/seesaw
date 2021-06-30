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
};

// Useful values
const double root2 = sqrt(2.0);
const std::complex<double> im = sqrt(std::complex<double>(-1.0));

// How many decimals to output for the matrices
int precision = 2;

// Seesaw iterations
int numIters = 100000;

// Convergence criteria
double tol = 1e-8;
int numInRowRequired = 10;

// How much to output (0 == none, 1 == normal, 2 == extra)
int verbosity = 1;

// Which random method to use
int randomMethod = 2;

// The seed to use for randomness
std::string seed = "";

// Force matrix values
std::vector<double> fixedVals;

// What to output
int outputMethod = 1;

// Whether to force the rank of each matrix
bool restrictRankA = false;
bool restrictRankB = false;

// Turn an Eigen matix to a std::vector
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
std::complex<double> inner(complex2 mat1, complex2 mat2){
	std::complex<double> sum = 0;
	for (int i=0; i<mat1.size(); i++){
		for (int j=0; j<mat1[0].size(); j++){
			sum += mat1[i][j] * mat2[i][j];
		}
	}
	return sum;
}

// Get the inner product of two matrices 
std::complex<double> inner(real2 mat1, complex2 mat2){
	std::complex<double> sum = 0;
	for (int i=0; i<mat1.size(); i++){
		for (int j=0; j<mat1[0].size(); j++){
			sum += mat1[i][j] * mat2[i][j];
		}
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

// Pretty print a generic 1D vector with length n 
template <typename type> void prettyPrint(std::string pre, std::vector<type> arr){

	// Used fixed precision
	std::cout << std::fixed << std::setprecision(precision);

	// For the first line, add the pre text
	std::cout << pre << " | ";

	// For the x values, combine them all on one line
	for (int x=0; x<arr.size(); x++){
		std::cout << std::setw(5) << arr[x] << " ";
	}

	// Output the row
	std::cout << "|" << std::endl;

}

// Pretty print a generic 2D array with width w and height h
template <typename type> void prettyPrint(std::string pre, std::vector<std::vector<type>> arr){

	// Used fixed precision
	std::cout << std::fixed << std::setprecision(precision);

	// Loop over the array
	std::string rowText;
	for (int y=0; y<arr.size(); y++){

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
		for (int x=0; x<arr[y].size(); x++){
			std::cout << std::setw(5) << arr[y][x] << " ";
		}

		// Output the row
		std::cout << "|" << std::endl;

	}

}

// Pretty print a complex 2D array with width w and height h
void prettyPrint(std::string pre, complex2 arr){

	// Used fixed precision
	std::cout << std::fixed << std::setprecision(precision);

	// Loop over the array
	std::string rowText;
	for (int y=0; y<arr.size(); y++){

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
		for (int x=0; x<arr[y].size(); x++){
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

// Pretty print a hyperrect
void prettyPrint(std::string pre, hyperRect arr){

	// Print each section
	prettyPrint(pre + ", l ", arr.l);
	prettyPrint(pre + ", L ", arr.L);
	prettyPrint(pre + ", m ", arr.m);
	prettyPrint(pre + ", M ", arr.M);

}

// Get the initial hyperrectangle bounding the set
hyperRect boundingRectangle(int p, int q, int d, int numOutcomeA, int numOutcomeB, complex2 &S, complex2 &T, complex3 &eta, complex3 &xi){

	// Hyperrect to construct 
	hyperRect toReturn(p, q);

	// Dimensions of X and Y for MOSEK
	auto dimXRef = monty::new_array_ptr(std::vector<int>({p, p}));
	auto dimYRef = monty::new_array_ptr(std::vector<int>({q, q}));

	// Locations of the start/end of each section for MOSEK
	std::vector<std::shared_ptr<monty::ndarray<int,1>>> startX;
	std::vector<std::shared_ptr<monty::ndarray<int,1>>> endX;
	for (int i=0; i<p; i+=d){
		startX.push_back(monty::new_array_ptr(std::vector<int>({i, i})));
		endX.push_back(monty::new_array_ptr(std::vector<int>({i+d, i+d})));
	}
	std::vector<std::shared_ptr<monty::ndarray<int,1>>> startY;
	std::vector<std::shared_ptr<monty::ndarray<int,1>>> endY;
	for (int i=0; i<q; i+=d){
		startY.push_back(monty::new_array_ptr(std::vector<int>({i, i})));
		endY.push_back(monty::new_array_ptr(std::vector<int>({i+d, i+d})));
	}

	// Reference to the zero/identity matrix for MOSEK
	real2 zero(d, real1(d, 0.0));
	real2 identity(d, real1(d, 0.0));
	for (int i=0; i<d; i++){
		identity[i][i] = 1.0;
	}
	auto identityRef = monty::new_array_ptr(identity);
	auto zeroRef = monty::new_array_ptr(zero);

	// Cached array sizes for MOSEK
	auto zero2DRefX = monty::new_array_ptr(real2(p, real1(p)));
	auto zero2DRefY = monty::new_array_ptr(real2(q, real1(q)));

	// Get the X sections
	for (int j=0; j<p*p; j++){

		// Generate C = sum_k S_{kj} eta_k^T
		real2 Cr(p, real1(p, 0.0));
		real2 Ci(p, real1(p, 0.0));
		for (int k=0; k<p*p; k++){
			for (int l=0; l<p; l++){
				for (int m=0; m<p; m++){
					Cr[l][m] += std::real(S[k][j] * eta[k][l][m]);
					Ci[l][m] += std::imag(S[k][j] * eta[k][l][m]);
				}
			}
		}
		auto CrRef = monty::new_array_ptr(Cr);
		auto CiRef = monty::new_array_ptr(Ci);

		// Create the MOSEK model 
		mosek::fusion::Model::t lModel = new mosek::fusion::Model(); 

		// The matrices to optimise
		mosek::fusion::Variable::t XrOpt = lModel->variable(dimXRef, mosek::fusion::Domain::symmetric(mosek::fusion::Domain::inRange(-1.0, 1.0)));
		mosek::fusion::Variable::t XiOpt = lModel->variable(dimXRef, mosek::fusion::Domain::inRange(-1.0, 1.0));

		// Sections need to be semidefinite
		for (int i=0; i<startX.size(); i++){
			lModel->constraint(mosek::fusion::Expr::vstack(
									mosek::fusion::Expr::hstack(
										XrOpt->slice(startX[i], endX[i]), 
										mosek::fusion::Expr::neg(XiOpt->slice(startX[i], endX[i]))
									), 
									mosek::fusion::Expr::hstack(
										XiOpt->slice(startX[i], endX[i]),
										XrOpt->slice(startX[i], endX[i]) 
									)
							   ), mosek::fusion::Domain::inPSDCone(2*d));

			// And be anti-symmetric
			lModel->constraint(mosek::fusion::Expr::add(XiOpt->slice(startX[i], endX[i]), mosek::fusion::Expr::transpose(XiOpt->slice(startX[i], endX[i]))), mosek::fusion::Domain::equalsTo(zeroRef));

			// And have trace 1
			//lModel->constraint(mosek::fusion::Expr::sum(XrOpt->slice(startX[i], endX[i])->diag()), mosek::fusion::Domain::equalsTo(1));
			//lModel->constraint(mosek::fusion::Expr::sum(XiOpt->slice(startX[i], endX[i])->diag()), mosek::fusion::Domain::equalsTo(0));

		}

		// Need to sum to the identity
		for (int i=0; i<startX.size(); i+=numOutcomeA){
			auto prods = new monty::ndarray<mosek::fusion::Expression::t,1>(monty::shape(numOutcomeA));
			auto prods2 = new monty::ndarray<mosek::fusion::Expression::t,1>(monty::shape(numOutcomeA));
			for(int j=0; j<numOutcomeA; j++){
				(*prods)[j] = XrOpt->slice(startX[i+j], endX[i+j]);
				(*prods2)[j] = XiOpt->slice(startX[i+j], endX[i+j]);
			}
			lModel->constraint(mosek::fusion::Expr::add(std::shared_ptr<monty::ndarray<mosek::fusion::Expression::t,1>>(prods)), mosek::fusion::Domain::equalsTo(identityRef));
			lModel->constraint(mosek::fusion::Expr::add(std::shared_ptr<monty::ndarray<mosek::fusion::Expression::t,1>>(prods2)), mosek::fusion::Domain::equalsTo(zeroRef));
		}

		// Specify the zero elements of X
		for (int i=0; i<p; i++){
			for (int j=0; j<p; j++){
				if (j > i+d || j < i){
					lModel->constraint(XrOpt->index(i,j), mosek::fusion::Domain::equalsTo(0));
					lModel->constraint(XiOpt->index(i,j), mosek::fusion::Domain::equalsTo(0));
					lModel->constraint(XrOpt->index(j,i), mosek::fusion::Domain::equalsTo(0));
					lModel->constraint(XiOpt->index(j,i), mosek::fusion::Domain::equalsTo(0));
				}
			}
		}

		// Setup the objective function
		mosek::fusion::Expression::t objectiveExpr = mosek::fusion::Expr::sub(mosek::fusion::Expr::dot(XrOpt, CrRef), mosek::fusion::Expr::dot(XiOpt, CiRef));

		// The objective function should be real
		lModel->constraint(mosek::fusion::Expr::add(mosek::fusion::Expr::dot(XrOpt, CiRef), mosek::fusion::Expr::dot(XiOpt, CrRef)), mosek::fusion::Domain::equalsTo(0));

		// Minimise the object function
		lModel->objective(mosek::fusion::ObjectiveSense::Minimize, objectiveExpr);
		lModel->solve();
		toReturn.l[j] = lModel->primalObjValue();

		// Extract the X values just to see TODO
		auto tempXr = *(XrOpt->level());
		auto tempXi = *(XiOpt->level());
		complex2 X(p, complex1(p));
		for (int i=0; i<p*p; i++){
			X[i/p][i%p] = tempXr[i] + im*tempXi[i];
		}
		prettyPrint("X after rect = ", X);
		std::cout << std::endl;

		// Maximise the object function
		lModel->objective(mosek::fusion::ObjectiveSense::Maximize, objectiveExpr);
		lModel->solve();
		toReturn.L[j] = lModel->primalObjValue();

	}

	// Get the Y sections
	for (int k=0; k<q*q; k++){

		// Generate C = sum_j T_{jk} xi_j^T
		real2 Cr(q, real1(q, 0.0));
		real2 Ci(q, real1(q, 0.0));
		for (int j=0; j<q*q; j++){
			for (int l=0; l<q; l++){
				for (int m=0; m<q; m++){
					Cr[l][m] += std::real(T[j][k] * xi[j][l][m]);
					Ci[l][m] += std::imag(T[j][k] * xi[j][l][m]);
				}
			}
		}
		auto CrRef = monty::new_array_ptr(Cr);
		auto CiRef = monty::new_array_ptr(Ci);

		// Create the MOSEK model 
		mosek::fusion::Model::t mModel = new mosek::fusion::Model(); 

		// The matrices to optimise
		mosek::fusion::Variable::t YrOpt = mModel->variable(dimYRef, mosek::fusion::Domain::symmetric(mosek::fusion::Domain::inRange(-1.0, 1.0)));
		mosek::fusion::Variable::t YiOpt = mModel->variable(dimYRef, mosek::fusion::Domain::inRange(-1.0, 1.0));

		// Sections need to be semidefinite
		for (int i=0; i<startY.size(); i++){
			mModel->constraint(mosek::fusion::Expr::vstack(
									mosek::fusion::Expr::hstack(
										YrOpt->slice(startY[i], endY[i]), 
										mosek::fusion::Expr::neg(YiOpt->slice(startY[i], endY[i]))
									), 
									mosek::fusion::Expr::hstack(
										YiOpt->slice(startY[i], endY[i]),
										YrOpt->slice(startY[i], endY[i]) 
									)
							   ), mosek::fusion::Domain::inPSDCone(2*d));

			// And be anti-symmetric
			mModel->constraint(mosek::fusion::Expr::add(YiOpt->slice(startY[i], endY[i]), mosek::fusion::Expr::transpose(YiOpt->slice(startY[i], endY[i]))), mosek::fusion::Domain::equalsTo(zeroRef));

			// And have trace 1
			//lModel->constraint(mosek::fusion::Expr::sum(XrOpt->slice(startX[i], endX[i])->diag()), mosek::fusion::Domain::equalsTo(1));
			//lModel->constraint(mosek::fusion::Expr::sum(XiOpt->slice(startX[i], endX[i])->diag()), mosek::fusion::Domain::equalsTo(0));

		}

		// Need to sum to the identity
		for (int i=0; i<startY.size(); i+=numOutcomeB){
			auto prods = new monty::ndarray<mosek::fusion::Expression::t,1>(monty::shape(numOutcomeB));
			auto prods2 = new monty::ndarray<mosek::fusion::Expression::t,1>(monty::shape(numOutcomeB));
			for(int j=0; j<numOutcomeB; j++){
				(*prods)[j] = YrOpt->slice(startY[i+j], endY[i+j]);
				(*prods2)[j] = YiOpt->slice(startY[i+j], endY[i+j]);
			}
			mModel->constraint(mosek::fusion::Expr::add(std::shared_ptr<monty::ndarray<mosek::fusion::Expression::t,1>>(prods)), mosek::fusion::Domain::equalsTo(identityRef));
			mModel->constraint(mosek::fusion::Expr::add(std::shared_ptr<monty::ndarray<mosek::fusion::Expression::t,1>>(prods2)), mosek::fusion::Domain::equalsTo(zeroRef));
		}

		// Specify the zero elements of Y
		for (int i=0; i<q; i++){
			for (int j=0; j<q; j++){
				if (j > i+d || j < i){
					mModel->constraint(YrOpt->index(i,j), mosek::fusion::Domain::equalsTo(0));
					mModel->constraint(YiOpt->index(i,j), mosek::fusion::Domain::equalsTo(0));
					mModel->constraint(YrOpt->index(j,i), mosek::fusion::Domain::equalsTo(0));
					mModel->constraint(YiOpt->index(j,i), mosek::fusion::Domain::equalsTo(0));
				}
			}
		}

		// The objective function should be real
		mModel->constraint(mosek::fusion::Expr::add(mosek::fusion::Expr::dot(YrOpt, CiRef), mosek::fusion::Expr::dot(YiOpt, CrRef)), mosek::fusion::Domain::equalsTo(0));

		// Setup the objective function
		mosek::fusion::Expression::t objectiveExpr = mosek::fusion::Expr::sub(mosek::fusion::Expr::dot(YrOpt, CrRef), mosek::fusion::Expr::dot(YiOpt, CiRef));

		// Minimise the object function
		mModel->objective(mosek::fusion::ObjectiveSense::Minimize, objectiveExpr);
		mModel->solve();
		toReturn.m[k] = mModel->primalObjValue();

		// Extract the Y values just to see TODO
		auto tempYr = *(YrOpt->level());
		auto tempYi = *(YiOpt->level());
		complex2 Y(q, complex1(q));
		for (int i=0; i<q*q; i++){
			Y[i/q][i%q] = tempYr[i] + im*tempYi[i];
		}
		prettyPrint("Y after rect = ", Y);
		std::cout << std::endl;

		// Maximise the object function
		mModel->objective(mosek::fusion::ObjectiveSense::Maximize, objectiveExpr);
		mModel->solve();
		toReturn.M[k] = mModel->primalObjValue();

	}

	// Return the final hyperrectangle
	return toReturn;

}

// Compute the upper and lower bounds for a hyperrectangle
void computeBounds(int p, int q, int d, int numOutcomeA, int numOutcomeB, complex2 &S, complex2 &Delta, complex2 &T, complex3 &eta, complex3 &xi, hyperRect &rect, double &lowerBound, double &upperBound, real1 &x, real1 &y){

	// Define some useful quantities
	int K = std::min(p*p, q*q);

	// Reference to the zero/identity matrix for MOSEK
	real2 zero(d, real1(d, 0.0));
	real2 identity(d, real1(d, 0.0));
	for (int i=0; i<d; i++){
		identity[i][i] = 1.0;
	}
	auto identityRef = monty::new_array_ptr(identity);
	auto zeroRef = monty::new_array_ptr(zero);

	// Assemble the combined S and Eta matrices 
	std::vector<std::shared_ptr<monty::ndarray<double,2>>> SEtarRef;
	std::vector<std::shared_ptr<monty::ndarray<double,2>>> SEtaiRef;
	for (int j=0; j<p*p; j++){
		real2 SEtar(p, real1(p));
		real2 SEtai(p, real1(p));
		for (int k=0; k<p*p; k++){
			for (int i1=0; i1<p; i1++){
				for (int i2=0; i2<p; i2++){
					SEtar[i1][i2] += std::real(S[k][j]*eta[k][i1][i2]);
					SEtai[i1][i2] += std::imag(S[k][j]*eta[k][i1][i2]);
				}
			}
		}
		SEtarRef.push_back(monty::new_array_ptr(SEtar));
		SEtaiRef.push_back(monty::new_array_ptr(SEtai));
	}

	// Assemble the combined T and Xi matrices
	std::vector<std::shared_ptr<monty::ndarray<double,2>>> TXirRef;
	std::vector<std::shared_ptr<monty::ndarray<double,2>>> TXiiRef;
	for (int j=0; j<q*q; j++){
		real2 TXir(q, real1(q));
		real2 TXii(q, real1(q));
		for (int k=0; k<q*q; k++){
			for (int i1=0; i1<q; i1++){
				for (int i2=0; i2<q; i2++){
					TXir[i1][i2] += std::real(T[k][j]*xi[k][i1][i2]);
					TXii[i1][i2] += std::imag(T[k][j]*xi[k][i1][i2]);
				}
			}
		}
		TXirRef.push_back(monty::new_array_ptr(TXir));
		TXiiRef.push_back(monty::new_array_ptr(TXii));
	}

	// Assemble the G matrices
	std::vector<std::shared_ptr<monty::ndarray<double,2>>> GlrRef;
	std::vector<std::shared_ptr<monty::ndarray<double,2>>> GliRef;
	std::vector<std::shared_ptr<monty::ndarray<double,2>>> GLrRef;
	std::vector<std::shared_ptr<monty::ndarray<double,2>>> GLiRef;
	for (int j=0; j<K; j++){
		real2 Glr(p, real1(p));
		real2 Gli(p, real1(p));
		real2 GLr(p, real1(p));
		real2 GLi(p, real1(p));
		for (int k=0; k<p*p; k++){
			for (int i1=0; i1<p; i1++){
				for (int i2=0; i2<p; i2++){
					Glr[i1][i2] += std::real(Delta[j][0]*rect.m[j]*S[k][j]*eta[k][i1][i2]);
					Gli[i1][i2] += std::imag(Delta[j][0]*rect.m[j]*S[k][j]*eta[k][i1][i2]);
					GLr[i1][i2] += std::real(Delta[j][0]*rect.M[j]*S[k][j]*eta[k][i1][i2]);
					GLi[i1][i2] += std::imag(Delta[j][0]*rect.M[j]*S[k][j]*eta[k][i1][i2]);
				}
			}
		}
		GlrRef.push_back(monty::new_array_ptr(Glr));
		GliRef.push_back(monty::new_array_ptr(Gli));
		GLrRef.push_back(monty::new_array_ptr(GLr));
		GLiRef.push_back(monty::new_array_ptr(GLi));
	}

	// Assemble the H matrices
	std::vector<std::shared_ptr<monty::ndarray<double,2>>> HmrRef;
	std::vector<std::shared_ptr<monty::ndarray<double,2>>> HmiRef;
	std::vector<std::shared_ptr<monty::ndarray<double,2>>> HMrRef;
	std::vector<std::shared_ptr<monty::ndarray<double,2>>> HMiRef;
	for (int j=0; j<K; j++){
		real2 Hmr(q, real1(q));
		real2 Hmi(q, real1(q));
		real2 HMr(q, real1(q));
		real2 HMi(q, real1(q));
		for (int l=0; l<q*q; l++){
			for (int i1=0; i1<q; i1++){
				for (int i2=0; i2<q; i2++){
					Hmr[i1][i2] += std::real(Delta[j][0]*rect.m[j]*S[l][j]*eta[l][i1][i2]);
					Hmi[i1][i2] += std::imag(Delta[j][0]*rect.m[j]*S[l][j]*eta[l][i1][i2]);
					HMr[i1][i2] += std::real(Delta[j][0]*rect.M[j]*S[l][j]*eta[l][i1][i2]);
					HMi[i1][i2] += std::imag(Delta[j][0]*rect.M[j]*S[l][j]*eta[l][i1][i2]);
				}
			}
		}
		HmrRef.push_back(monty::new_array_ptr(Hmr));
		HmiRef.push_back(monty::new_array_ptr(Hmi));
		HMrRef.push_back(monty::new_array_ptr(HMr));
		HMiRef.push_back(monty::new_array_ptr(HMi));
	}

	// Assemble the s vectors
	std::vector<double> slm;
	std::vector<double> sLM;
	for (int j=0; j<K; j++){
		slm.push_back(std::real(Delta[j][0]*rect.l[j]*rect.m[j]));
		sLM.push_back(std::real(Delta[j][0]*rect.L[j]*rect.M[j]));
	}

	// Dimensions of X and Y for MOSEK
	auto dimXRef = monty::new_array_ptr(std::vector<int>({p, p}));
	auto dimYRef = monty::new_array_ptr(std::vector<int>({q, q}));

	// Locations of the start/end of each section for MOSEK
	std::vector<std::shared_ptr<monty::ndarray<int,1>>> startX;
	std::vector<std::shared_ptr<monty::ndarray<int,1>>> endX;
	for (int i=0; i<p; i+=d){
		startX.push_back(monty::new_array_ptr(std::vector<int>({i, i})));
		endX.push_back(monty::new_array_ptr(std::vector<int>({i+d, i+d})));
	}
	std::vector<std::shared_ptr<monty::ndarray<int,1>>> startY;
	std::vector<std::shared_ptr<monty::ndarray<int,1>>> endY;
	for (int i=0; i<q; i+=d){
		startY.push_back(monty::new_array_ptr(std::vector<int>({i, i})));
		endY.push_back(monty::new_array_ptr(std::vector<int>({i+d, i+d})));
	}

	// Create the MOSEK model
	mosek::fusion::Model::t model = new mosek::fusion::Model(); 
	
	// The matrices to optimise
	mosek::fusion::Variable::t XrOpt = model->variable(dimXRef, mosek::fusion::Domain::symmetric(mosek::fusion::Domain::inRange(-1.0, 1.0)));
	mosek::fusion::Variable::t XiOpt = model->variable(dimXRef, mosek::fusion::Domain::inRange(-1.0, 1.0));
	mosek::fusion::Variable::t YrOpt = model->variable(dimYRef, mosek::fusion::Domain::symmetric(mosek::fusion::Domain::inRange(-1.0, 1.0)));
	mosek::fusion::Variable::t YiOpt = model->variable(dimYRef, mosek::fusion::Domain::inRange(-1.0, 1.0));
	mosek::fusion::Variable::t rOpt = model->variable(K, mosek::fusion::Domain::inRange(-1.0, 1.0));

	// Objective function TODO max or min
	model->objective(mosek::fusion::ObjectiveSense::Maximize, mosek::fusion::Expr::sum(rOpt));
	
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

		// And have trace 1
		//model->constraint(mosek::fusion::Expr::sum(XrOpt->slice(startX[i], endX[i])->diag()), mosek::fusion::Domain::equalsTo(1));
		//model->constraint(mosek::fusion::Expr::sum(XiOpt->slice(startX[i], endX[i])->diag()), mosek::fusion::Domain::equalsTo(0));

		// The imaginary part should be anti-symmetric
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
		//model->constraint(mosek::fusion::Expr::sum(YrOpt->slice(startY[i], endY[i])->diag()), mosek::fusion::Domain::equalsTo(1));
		//model->constraint(mosek::fusion::Expr::sum(YiOpt->slice(startY[i], endY[i])->diag()), mosek::fusion::Domain::equalsTo(0));
		
		// The imaginary part should be anti-symmetric
		model->constraint(mosek::fusion::Expr::add(YiOpt->slice(startY[i], endY[i]), mosek::fusion::Expr::transpose(YiOpt->slice(startY[i], endY[i]))), mosek::fusion::Domain::equalsTo(zeroRef));

	}

	// X needs to sum to the identity
	for (int i=0; i<startX.size(); i+=numOutcomeA){

		// Real part should be one
		auto prods = new monty::ndarray<mosek::fusion::Expression::t,1>(monty::shape(numOutcomeA));
		for(int j=0; j<numOutcomeA; j++){
			(*prods)[j] = XrOpt->slice(startX[i+j], endX[i+j]);
		}
		model->constraint(mosek::fusion::Expr::add(std::shared_ptr<monty::ndarray<mosek::fusion::Expression::t,1>>(prods)), mosek::fusion::Domain::equalsTo(identityRef));

		// Imag part should be zero
		auto prods2 = new monty::ndarray<mosek::fusion::Expression::t,1>(monty::shape(numOutcomeA));
		for(int j=0; j<numOutcomeA; j++){
			(*prods2)[j] = XiOpt->slice(startX[i+j], endX[i+j]);
		}
		model->constraint(mosek::fusion::Expr::add(std::shared_ptr<monty::ndarray<mosek::fusion::Expression::t,1>>(prods2)), mosek::fusion::Domain::equalsTo(zeroRef));

	}

	// Y needs to sum to the identity
	for (int i=0; i<startY.size(); i+=numOutcomeB){

		// Real part should be one
		auto prods = new monty::ndarray<mosek::fusion::Expression::t,1>(monty::shape(numOutcomeB));
		for(int j=0; j<numOutcomeB; j++){
			(*prods)[j] = YrOpt->slice(startY[i+j], endY[i+j]);
		}
		model->constraint(mosek::fusion::Expr::add(std::shared_ptr<monty::ndarray<mosek::fusion::Expression::t,1>>(prods)), mosek::fusion::Domain::equalsTo(identityRef));

		// Imag part should be zero
		auto prods2 = new monty::ndarray<mosek::fusion::Expression::t,1>(monty::shape(numOutcomeB));
		for(int j=0; j<numOutcomeB; j++){
			(*prods2)[j] = YiOpt->slice(startY[i+j], endY[i+j]);
		}
		model->constraint(mosek::fusion::Expr::add(std::shared_ptr<monty::ndarray<mosek::fusion::Expression::t,1>>(prods2)), mosek::fusion::Domain::equalsTo(zeroRef));

	}

	// Specify the zero elements of X
	for (int i=0; i<p; i++){
		for (int j=0; j<p; j++){
			if (j > i+d || j < i){
				model->constraint(XrOpt->index(i,j), mosek::fusion::Domain::equalsTo(0));
				model->constraint(XiOpt->index(i,j), mosek::fusion::Domain::equalsTo(0));
			}
		}
	}

	// Specify the zero elements of Y
	for (int i=0; i<q; i++){
		for (int j=0; j<q; j++){
			if (j > i+d || j < i){
				model->constraint(YrOpt->index(i,j), mosek::fusion::Domain::equalsTo(0));
				model->constraint(YiOpt->index(i,j), mosek::fusion::Domain::equalsTo(0));
			}
		}
	}

	// Special X constraints
	for (int j=0; j<p*p; j++){
		model->constraint(mosek::fusion::Expr::sub(mosek::fusion::Expr::dot(XrOpt, SEtarRef[j]), mosek::fusion::Expr::dot(XiOpt, SEtaiRef[j])), mosek::fusion::Domain::inRange(rect.l[j], rect.L[j]));
		model->constraint(mosek::fusion::Expr::add(mosek::fusion::Expr::dot(XiOpt, SEtarRef[j]), mosek::fusion::Expr::dot(XrOpt, SEtaiRef[j])), mosek::fusion::Domain::equalsTo(0));
	}
	
	// Special Y constraints
	for (int k=0; k<q*q; k++){
		model->constraint(mosek::fusion::Expr::sub(mosek::fusion::Expr::dot(YrOpt, TXirRef[k]), mosek::fusion::Expr::dot(YiOpt, TXiiRef[k])), mosek::fusion::Domain::inRange(rect.m[k], rect.M[k]));
		model->constraint(mosek::fusion::Expr::add(mosek::fusion::Expr::dot(YiOpt, TXirRef[k]), mosek::fusion::Expr::dot(YrOpt, TXiiRef[k])), mosek::fusion::Domain::equalsTo(0));
	}

	// Combined constraint TODO plus or minus
	for (int j=0; j<K; j++){

		// For l and m
		model->constraint(mosek::fusion::Expr::sub(
							 mosek::fusion::Expr::add(
								mosek::fusion::Expr::sub(mosek::fusion::Expr::dot(XrOpt, GlrRef[j]), 
														 mosek::fusion::Expr::dot(XiOpt, GliRef[j])), 
								mosek::fusion::Expr::sub(mosek::fusion::Expr::dot(YrOpt, HmrRef[j]), 
														 mosek::fusion::Expr::dot(YiOpt, HmiRef[j]))), 
						  rOpt->index(j)), mosek::fusion::Domain::lessThan(slm[j]));

		// Imag of this should be zero
		model->constraint(mosek::fusion::Expr::add(
							mosek::fusion::Expr::add(mosek::fusion::Expr::dot(XrOpt, GliRef[j]), 
													 mosek::fusion::Expr::dot(XiOpt, GlrRef[j])), 
							mosek::fusion::Expr::add(mosek::fusion::Expr::dot(YrOpt, HmiRef[j]), 
													 mosek::fusion::Expr::dot(YiOpt, HmrRef[j]))), 
						  mosek::fusion::Domain::equalsTo(0));

		// For L and M
		model->constraint(mosek::fusion::Expr::sub(
							 mosek::fusion::Expr::add(
								mosek::fusion::Expr::sub(mosek::fusion::Expr::dot(XrOpt, GLrRef[j]), 
														 mosek::fusion::Expr::dot(XiOpt, GLiRef[j])), 
								mosek::fusion::Expr::sub(mosek::fusion::Expr::dot(YrOpt, HMrRef[j]), 
														 mosek::fusion::Expr::dot(YiOpt, HMiRef[j]))), 
						  rOpt->index(j)), mosek::fusion::Domain::lessThan(sLM[j]));

		// Imag of this should be zero
		model->constraint(mosek::fusion::Expr::add(
								mosek::fusion::Expr::sub(mosek::fusion::Expr::dot(XrOpt, GLiRef[j]), 
														 mosek::fusion::Expr::dot(XiOpt, GLrRef[j])), 
								mosek::fusion::Expr::sub(mosek::fusion::Expr::dot(YrOpt, HMiRef[j]), 
														 mosek::fusion::Expr::dot(YiOpt, HMrRef[j]))), 
						  mosek::fusion::Domain::equalsTo(0));

	}

	// Solve 
	model->solve();

	// Extract the data
	lowerBound = model->primalObjValue();
	auto tempXr = *(XrOpt->level());
	auto tempXi = *(XiOpt->level());
	auto tempYr = *(YrOpt->level());
	auto tempYi = *(YiOpt->level());
	auto tempR = *(rOpt->level());
	complex2 X(p, complex1(p));
	complex2 Y(q, complex1(q));
	real1 r(K);
	for (int i=0; i<p*p; i++){
		X[i/p][i%p] = tempXr[i] + im*tempXi[i];
	}
	for (int i=0; i<q*q; i++){
		Y[i/q][i%q] = tempYr[i] + im*tempYi[i];
	}
	for (int i=0; i<K; i++){
		r[i] = tempR[i];
	}

	// Convert these to the vector rep
	for (int i=0; i<p*p; i++){
		complex2 SEta(p, complex1(p));
		for (int k=0; k<p*p; k++){
			for (int i1=0; i1<p; i1++){
				for (int i2=0; i2<q; i2++){
					SEta[i1][i2] += S[k][i]*eta[k][i1][i2];
				}
			}
		}
		x[i] = std::real(inner(X, SEta));
	}
	for (int i=0; i<q*q; i++){
		complex2 TXi(q, complex1(q));
		for (int k=0; k<q*q; k++){
			for (int i1=0; i1<q; i1++){
				for (int i2=0; i2<q; i2++){
					TXi[i1][i2] += T[k][i]*xi[k][i1][i2];
				}
			}
		}
		y[i] = std::real(inner(Y, TXi));
	}

	std::cout << std::endl;
	prettyPrint("X = ", X);
	std::cout << std::endl;
	prettyPrint("Y = ", Y);
	std::cout << std::endl;
	prettyPrint("x = ", x);
	std::cout << std::endl;
	prettyPrint("y = ", y);
	std::cout << std::endl;
	prettyPrint("r = ", r);

	// Calculate the upper bound from this too
	upperBound = 0;
	for (int j=0; j<K; j++){
		upperBound += std::real(Delta[j][0]*x[j]*y[j]);
	}

}

// Branch a hyperrectangle at a certain point
std::vector<hyperRect> branchHyperrectangle(int p, int q, hyperRect &Omega, real1 &v, real1 &w){

	// List of 4 hyperrectangles to return
	std::vector<hyperRect> toReturn(4, hyperRect(p, q));

	// Number of non-zero single values
	int K = std::min(p*p, q*q);

	// Determine the index which gives the biggest difference
	int I = 0;
	double bestVal = -10000000;
	for (int i=0; i<K; i++){
		double val = std::max(Omega.m[i]*v[i] + Omega.l[i]*w[i] - Omega.l[i]*Omega.m[i],
			                           Omega.M[i]*v[i] + Omega.L[i]*w[i] - Omega.L[i]*Omega.M[i]) - v[i]*w[i];
		if (std::abs(val) > std::abs(bestVal)){
			bestVal = val;
			I = i;
		}
	}

	// Rectangles are the same apart from the special index I
	toReturn[0] = Omega;
	toReturn[1] = Omega;
	toReturn[2] = Omega;
	toReturn[3] = Omega;

	// For the first hyperrectangle
	toReturn[0].l[I] = Omega.l[I];
	toReturn[0].L[I] = v[I];
	toReturn[0].m[I] = Omega.m[I];
	toReturn[0].M[I] = w[I];

	// For the second hyperrectangle
	toReturn[1].l[I] = v[I];
	toReturn[1].L[I] = Omega.L[I];
	toReturn[1].m[I] = Omega.m[I];
	toReturn[1].M[I] = w[I];

	// For the third hyperrectangle
	toReturn[2].l[I] = v[I];
	toReturn[2].L[I] = Omega.L[I];
	toReturn[2].m[I] = w[I];
	toReturn[2].M[I] = Omega.M[I];

	// For the fourth hyperrectangle
	toReturn[3].l[I] = Omega.l[I];
	toReturn[3].L[I] = v[I];
	toReturn[3].m[I] = w[I];
	toReturn[3].M[I] = Omega.M[I];

	// Return these 4 hyperrectangles
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

	// The bases of these matrices 
	complex3 eta(p*p, complex2(p, complex1(p)));
	complex3 xi(q*q, complex2(q, complex1(q)));

	// Assemble eta to be orthonormal and self-adjoint
	std::cout << "Constructing eta..." << std::endl;
	int next = 0;
	for (int i=0; i<p; i++){
		for (int j=i; j<p; j++){

			// Self-adjoint 1's
			eta[next][i][j] = 1;
			eta[next][j][i] = 1;
			next += 1;

			// Self-adjoint i's
			if (i != j){
				eta[next][i][j] = im;
				eta[next][j][i] = -im;
				next += 1;
			}

		}
	}

	// Assemble xi to be orthonormal and self-adjoint
	std::cout << "Constructing xi..." << std::endl;
	next = 0;
	for (int i=0; i<q; i++){
		for (int j=i; j<q; j++){

			// Self-adjoint 1's
			xi[next][i][j] = 1;
			xi[next][j][i] = 1;
			next += 1;

			// Self-adjoint i's
			if (i != j){
				xi[next][i][j] = im;
				xi[next][j][i] = -im;
				next += 1;
			}

		}
	}

	// The matrix defining the entire problem 
	real2 Q(p*q, real1(p*q, 0.0));

	// The coefficients for the block-diagonals of Q
	std::cout << "Constructing Q..." << std::endl;
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
	for (int i=0; i<blockQ.size(); i++){
		for (int j1=0; j1<d; j1++){
			for (int j2=0; j2<d; j2++){
				for (int k1=0; k1<d; k1++){
					for (int k2=0; k2<d; k2++){
						if (j1 == j2 && k1 == k2){
							Q[i*d*d+j1*d+j2][i*d*d+k1*d+k2] = blockQ[i];
						}
					}
				}
			}
		}
	}

	// Calculate U_{j,k} = tr(Q(eta_j (x) xi_k))
	std::cout << "Calculating U from Q..." << std::endl;
	complex2 U(p*p, complex1(q*q));
	for (int j=0; j<p*p; j++){
		for (int k=0; k<q*q; k++){
			U[j][k] = inner(Q, outer(eta[j], xi[k]));
		}
	}
	
	// Pre-calculate the decomposition U = S Delta T
	std::cout << "Pre-calculating decomposition..." << std::endl;
	Eigen::BDCSVD<Eigen::MatrixXcd> svd(vecToMat(U), Eigen::ComputeFullU | Eigen::ComputeFullV);
	complex2 Delta = matToVec(svd.singularValues());
	complex2 S = matToVec(svd.matrixU());
	complex2 T = matToVec(svd.matrixV());

	// The initial hyperrectangle
	std::cout << "Calculating initial hyperrrectangle..." << std::endl;
	hyperRect D(p, q);
	D = boundingRectangle(p, q, d, numOutcomeA, numOutcomeB, S, T, eta, xi);

	// Output various things TODO
	std::cout << "p = " << p << std::endl;
	std::cout << "q = " << q << std::endl;
	std::cout << "eta shape = " << eta.size() << " x " << eta[0].size() << " x " << eta[0][0].size() << std::endl;
	std::cout << "xi shape = " << xi.size() << " x " << xi[0].size() << " x " << xi[0][0].size() << std::endl;
	std::cout << "Q shape = " << Q.size() << " x " << Q[0].size() << std::endl;
	std::cout << "U shape = " << U.size() << " x " << U[0].size() << std::endl;
	std::cout << "Delta shape = " << Delta.size() << " x " << Delta[0].size() << std::endl;
	std::cout << "S shape = " << S.size() << " x " << S[0].size() << std::endl;
	std::cout << "T shape = " << T.size() << " x " << T[0].size() << std::endl;
	std::cout << std::endl;
	prettyPrint("blockQ = ", blockQ);
	std::cout << std::endl;
	prettyPrint("Q = ", Q);
	std::cout << std::endl;
	prettyPrint("U = ", U);
	std::cout << std::endl;
	prettyPrint("Delta = ", Delta);
	std::cout << std::endl;
	prettyPrint("S = ", S);
	std::cout << std::endl;
	prettyPrint("T = ", T);
	std::cout << std::endl;
	prettyPrint("initial hyperrect", D);
	std::cout << std::endl;

	// The tolerance until deemed to have converged
	double epsilon = 1e-5;

	// Init things here to prevent re-init each iterations
	double lowerBound = 0;
	double upperBound = 0;
	hyperRect Omega(p, q);
	std::vector<hyperRect> newRects;
	real1 x(p*p);
	real1 y(q*q);
	int newLoc = -1;

	// Get the initial value for the upper/lower bounds
	std::cout << "Calculating initial bounds..." << std::endl;
	computeBounds(p, q, d, numOutcomeA, numOutcomeB, S, Delta, T, eta, xi, D, lowerBound, upperBound, x, y);
	std::cout << "For initial hyperrect: " << lowerBound << " " << upperBound << std::endl;
	
	// Keep track of the remaining hyperrects and their bounds
	std::vector<hyperRect> P = {D};
	real2 xCoords = {x};
	real2 yCoords = {y};
	std::vector<double> lowerBounds = {lowerBound};
	double bestLowerBound = lowerBound;
	double bestUpperBound = upperBound;

	// Keep looping until the bounds match
	int iter = 0;
	while (bestUpperBound - bestLowerBound > epsilon){

		// Choose the lower-bounding hyperrectangle
		Omega = P[0];
		x = xCoords[0];
		y = yCoords[0];

		// Create the four new hyperrectangles
		newRects = branchHyperrectangle(p, q, Omega, x, y);

		// Remove the current rect
		P.erase(P.begin(), P.begin()+1);
		xCoords.erase(xCoords.begin(), xCoords.begin()+1);
		yCoords.erase(yCoords.begin(), yCoords.begin()+1);
		lowerBounds.erase(lowerBounds.begin(), lowerBounds.begin()+1);

		// Per-iteration output
		std::cout << "-------------------------------------" << std::endl;
		std::cout << "        Iteration: " << iter << std::endl;
		std::cout << "-------------------------------------" << std::endl;

		// For each of the new hyperrectangles
		for (int j=0; j<4; j++){

			// Get the bounds
			std::cout << "Computing bounds for hyperrect " << j << "..." << std::endl;
			computeBounds(p, q, d, numOutcomeA, numOutcomeB, S, Delta, T, eta, xi, newRects[j], lowerBound, upperBound, x, y);
			std::cout << "For hyperrect " << j << ": " << lowerBound << " " << upperBound << std::endl;

			// Is it the new best?
			if (lowerBound > bestLowerBound){
				bestLowerBound = lowerBound;
			}
			if (upperBound < bestUpperBound){
				bestUpperBound = upperBound;
			}

			// Place it into the queue
			newLoc = lowerBounds.size();
			for (int i=0; i<lowerBounds.size(); i++){
				if (lowerBound < lowerBounds[i]){
					newLoc = i;
					break;
				}
			}
			P.insert(P.begin()+newLoc, newRects[j]);
			xCoords.insert(xCoords.begin()+newLoc, x);
			yCoords.insert(yCoords.begin()+newLoc, y);
			lowerBounds.insert(lowerBounds.begin()+newLoc, lowerBound);

		}

		// Output the best so far
		std::cout << "Lower bound: " << bestLowerBound << std::endl;
		std::cout << "Upper bound: " << bestUpperBound << std::endl;

		// Iteration finished
		iter += 1;

	}

	// Return the best
	std::cout << "-------------------------------------" << std::endl;
	std::cout << "    Final results" << std::endl;
	std::cout << "-------------------------------------" << std::endl;
	std::cout << "Lower bound: " << bestLowerBound << std::endl;
	std::cout << "Upper bound: " << bestUpperBound << std::endl;

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

	// If told to just generate a random symteric matrix
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
			if (verbosity >= 2){
				std::cout << std::endl;
				prettyPrint("A[" + std::to_string(x) + "][" + std::to_string(a) + "] = ", A[x][a]);
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
			if (verbosity >= 2){
				std::cout << std::endl;
				prettyPrint("moments " + std::to_string(y1) + " " + std::to_string(y2) + " = ", momentB);
			}

		}
	}

	// Stop the timer 
	auto t2 = std::chrono::high_resolution_clock::now();

	// Output after
	exact = numPerm*sqrt(d*(d-1));
	delta = exact-finalResult;
	if (outputMethod == 2){
		std::cout << std::fixed << std::setprecision(9) << delta << std::endl;;
	} else if (outputMethod == 4){
		std::cout << std::fixed << iter << std::endl;;
	} else if (outputMethod == 5){
		std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << std::endl;
	} else if (outputMethod == 1){
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
			std::cout << "-h               show the help" << std::endl;
			std::cout << "-d [int]         set the dimension" << std::endl;
			std::cout << "-n [int]         set the number of measurements" << std::endl;
			std::cout << "-t [dbl]         set the convergence threshold" << std::endl;
			std::cout << "-i [int]         set the iteration limit" << std::endl;
			std::cout << "-v               verbose output" << std::endl;
			std::cout << "-c               use the CHSH method" << std::endl;
			std::cout << "-e               use the extended method" << std::endl;
			std::cout << "-j               use the JCB method" << std::endl;
			std::cout << "-r               start completely random without G-S" << std::endl;
			std::cout << "-s [str]         set the random seed" << std::endl;
			std::cout << "-f [int*2] [dbl] set part of the initial array" << std::endl;
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
			d = std::stoi(argv[i+1]);
			i += 1;

		// Set the verbosity
		} else if (arg == "-v") {
			verbosity = 2;

		// Set the tolerance
		} else if (arg == "-t") {
			tol = std::stod(argv[i+1]);
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

}

