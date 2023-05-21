// Finds error for fit function by sampling parameter space according to covariance matrix and optimal params,
// then takes standard deviation of sampled fit within region to estimate fit error

#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <chrono>
#include <fstream>
#include "calc_delta.cuh"

using namespace Eigen;

MatrixXd param_samples(int n, int m, MatrixXd cov, VectorXd means) 
{
	// create random number generator with normal distribution
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0.0, 1.0);

	// generate matrix of n samples of m variables
	MatrixXd samples(n, m);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			samples(i, j) = distribution(generator);
		}
	}

	// Perform Cholesky decomposition of covariance matrix
	LLT<MatrixXd> lltOfCov(cov);
	MatrixXd U = lltOfCov.matrixU();
	//std::cout << "U:\n" << U << std::endl;

	// multiply samples matrix by matrix cov U 
	MatrixXd result = samples * U;

	// add mean to result 
	result.rowwise() += means.transpose();

	return result;
}

// calculates delta vector containing sums of fit within region for sampled params
__global__ void calculateDelta(int n, int i0, int j0, int i1, int j1, double* params, double* delta) 
{
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	if (k < n) {
		double sum = 0.0;
		for (int i = i0; i < i1; i++) {
			for (int j = j0; j < j1; j++) {
				double value = plane(i, j, params[0*n + k], params[1*n + k], params[2*n + k]) + gaussian(i, j, params[3*n + k], params[4*n + k], params[5*n + k]); 
				sum += value;
			}
		}
		delta[k] = sum;
	}   
}

// necessary for calculating d0 on cpu
double calc_d0(int i0, int j0, int i1, int j1, double b0, double b1, double b2, double A, double sigx, double sigy)
{
	double x0 = 156.;
	double y0 = 136.;
	double sum = 0;
	for (int i = i0; i < i1; i++) {
		for (int j = j0; j < j1; j++) {
			double plane = b0 + b1*i + b2*j;
			double gauss = A * exp(-pow(i-x0,2.)/(2.*pow(sigx,2.)) - pow(j-y0,2.)/(2.*pow(sigy,2.)));
			sum += (plane + gauss);
		}
	}
	return sum;
}


int main(int argc, char** argv) {
	// Region bounds, (x0, y0, x1, y1)
	const VectorXd PR = Vector4d(128, 104, 184, 168);
	const VectorXd SR = Vector4d(115, 87,  201, 185);
	const VectorXd BR = Vector4d(73 , 66,  294, 195);
	const VectorXd FR = Vector4d(120, 100, 190, 170);

	if (argc != 3) {
		std::cerr << "Usage: " << argv[0] << " <n> <cov##.dat>" << std::endl;
		return 1;
	}
	int n = std::stoi(argv[1]);
	if (n <= 0) {
		std::cerr << "Error: n must be a positive integer." << std::endl;
		return 1;
	}
	std::cout << "n = " << n << std::endl;
	int m = 6;  // number of variables 

	std::string filename(argv[2]);
	std::ifstream infile(filename);

	// define covariance matrix, means for sample params from input data file 
	MatrixXd cov(m, m);
	VectorXd means(m);
	std::string line;
	int row = 0;
	while (std::getline(infile, line)) {
		std::istringstream iss(line);
		double value;
		int col = 0;
		while (iss >> value) {
			// assumes that datafile has 2 lines of header and 1 line separating covariance matrix and means column vector
			if (row-2 < m) {
				cov(row-2, col) = value;
			} else {
				means(row-3-m) = value;
			}
			col ++;
		}
		row ++;
	}
	std::cout << "input data: " << filename << std::endl;
	std::cout << "analyzing region: FR" << std::endl;	// manually change this line, cuda call, and d0 calculation  
	std::cout << "cov:\n" << cov << std::endl;
	std::cout << "means:\n" << means.transpose() << std::endl;

	// get samples
	MatrixXd params(n, m);
	params = param_samples(n, m, cov, means); 

	// perform some cuda magic:
	double* delta = new double[n];

	double* d_params;
	cudaMalloc(&d_params, params.size() * sizeof(double));
	cudaMemcpy(d_params, params.data(), params.size() * sizeof(double), cudaMemcpyHostToDevice);

	double* d_delta;
	cudaMalloc((void**)&d_delta, n * sizeof(double));

	int blockSize = 256;
	int numBlocks = (n + blockSize - 1) / blockSize;

	// find standard deviation of regional average for sampled parameters
	auto start = std::chrono::high_resolution_clock::now();	// start timer
	calculateDelta<<<numBlocks, blockSize>>>(n, int(FR(0)), int(FR(1)), int(FR(2)), int(FR(3)), d_params, d_delta);

	cudaMemcpy(delta, d_delta, n * sizeof(double), cudaMemcpyDeviceToHost);
	auto end = std::chrono::high_resolution_clock::now();	// end timer
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start); 

	std::cout << "Time elapsed: " << duration.count()/1000. << " seconds.\n" << std::endl;

	cudaFree(d_params);
	cudaFree(d_delta);

	// find mean and standard deviation of delta
	Map<VectorXd> vec(delta, n);
	double mu = vec.mean();
	double variance = ((vec.array() - mu).square().sum()) / (vec.size() - 1);
	double std_dev = sqrt(variance);
	std::cout << "mean: " << mu << std::endl;
	std::cout << "standard deviation: " << std_dev << std::endl;

	// find d0 for optimal parameters
	double d0 = calc_d0(int(FR(0)), int(FR(1)), int(FR(2)), int(FR(3)), means[0], means[1], means[2], means[3], means[4], means[5]); 
	std::cout << "d0: " << d0 << "\n\n"; 

	// Write delta to a file
	std::ofstream outfile;
	outfile.open("delta.dat");
	outfile << "# delta values\n";
	outfile << vec;
	outfile.close();

	delete[] delta;

	return 0;
}
