#pragma once

#include "../gemm.hh"

#include <random>

namespace cpu {

/** An abstract class for GEMM BLAS kernels. */
		template <typename T>
		class sp_gemm : public ::gemm<T> {
		public:
				using ::gemm<T>::gemm;
				using ::gemm<T>::m_;
				using ::gemm<T>::n_;
				using ::gemm<T>::k_;
				using ::gemm<T>::A_;
				using ::gemm<T>::B_;
				using ::gemm<T>::C_;

		public:
			/** Initialise the required data structures. */
			virtual void initialise(int n, double sparsity, bool binary = false) {
				n_ = n;

				A_ = (T*)malloc(sizeof(T) * n_ * n_);
				B_ = (T*)malloc(sizeof(T) * n_ * n_);
				C_ = (T*)malloc(sizeof(T) * n_ * n_);

				// Set initial values to 0
				for (int i = 0; i < (n_ * n_); i++) {
					A_[i] = 0.0;
					B_[i] = 0.0;
				}

				// Random number generator objects for use in descent
				std::default_random_engine gen;
				gen.seed(std::chrono::system_clock::now()
								         .time_since_epoch().count());
				std::uniform_real_distribution<double> dist(0.0, 1.0);

				// Work out number of edges needed to achieve target sparsity
				int edges = 1 + (int) (n * n * (1 - sparsity));

				// Initialise the matrices
				// Using a=0.45 and b=c=0.22 as default probabilities
				for (int i = 0; i < edges; i++) {
					while (!rMat(A_, n, 0, n - 1, 0, n - 1,
					             0.45, 0.22, 0.22,
					             &gen, dist, false)) {}
					while (!rMat(B_, n, 0, n - 1, 0, n - 1,
					             0.45, 0.22, 0.22,
					             &gen, dist, false)) {}
				}
			}

			private:
				bool rMat(T* M, int n, int x1, int x2, int y1, int y2,
					        float a, float b, float c, std::default_random_engine* gen,
					        std::uniform_real_distribution<double> dist, bool bin) {
					// If a 1x1 submatrix, then add an edge and return out
					if (x1 >= x2 && y1 >= y2) {
						if (abs(M[(y1 * n) + x1]) > 0.1) {
							return false;
						} else {
							// Add 1.0 if this is a binary graph, and a random real number otherwise
							M[(int) (y1 * n) + x1] = (bin) ? 1.0 : (((rand() % 10000) /
											100.0) - 50.0);
							return true;
						}
					} else {
						// Divide up the matrix
						int xMidPoint = x1 + floor((x2 - x1) / 2);
						int yMidPoint = y1 + floor((y2 - y1) / 2);

						// ToDo -- add some noise to these values between iterations
						float newA = a;
						float newB = b;
						float newC = c;

						// Work out which quarter to recurse into
						// There are some ugly ternary operators here to avoid going out of bounds in the edge case
						// that we are already at 1 width or 1 height
						float randomNum = dist(*gen);
						if (randomNum < a) {
							return rMat(M, n, x1, xMidPoint, y1, yMidPoint,
							            newA, newB, newC, gen, dist, bin);
						} else if (randomNum < (a + b)) {
							return rMat(M, n, ((xMidPoint < x2) ? xMidPoint + 1 : xMidPoint), x2, y1, yMidPoint,
							            newA, newB, newC, gen, dist, bin);
						} else if (randomNum < (a + b + c)) {
							return rMat(M, n, x1, xMidPoint, ((yMidPoint < y2) ? yMidPoint + 1 : yMidPoint), y2,
							            newA, newB, newC, gen, dist, bin);
						} else {
							return rMat(M, n, ((xMidPoint < x2) ? xMidPoint + 1 : xMidPoint), x2,
							            ((yMidPoint < y2) ? yMidPoint + 1 : yMidPoint), y2, newA, newB, newC,
							            gen, dist, bin);
						}
					}
					return true;
				}
				/** Do any necessary cleanup (free pointers, close library handles, etc.)
				 * after Kernel has been called. */
				void postCallKernelCleanup() {
					free(A_);
					free(B_);
					free(C_);
				}
		};
}  // namespace cpu