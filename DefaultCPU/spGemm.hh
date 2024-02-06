#pragma once

#include <time.h>

#include <random>
#include <vector>
#include <chrono>

#include "../include/CPU/spGemm.hh"
#include "../include/utilities.hh"

// ToDo: allow for symmetric matrices (undirected graphs)

namespace cpu {

#if defined CPU_DEFAULT
/** A class for spGEMM CPU BLAS kernels. */
    template <typename T>
    class spGemm_cpu : public spGemm<T> {
    public:
        using spGemm<T>::spGemm;
        using spGemm<T>::n_;

        /** Initialise the required data structures. */
        virtual void initialise(int n, double sparsity, bool binary) override {
            n_ = n;
            isBinary_ = binary;

            A_.assign(n * n, 0.0);
            B_.assign(n * n, 0.0);
            C_.reserve(n * n);

            // Random number generator objects for use in descent
            std::default_random_engine gen;
            gen.seed(std::chrono::system_clock::now()
                             .time_since_epoch().count());
            std::uniform_real_distribution<double> dist(0.0, 1.0);

            // Work out number of edges needed to achieve target sparsity
            int edges = 1 + (int)(n * n * (1 - sparsity));

            // Initialise the matrices
            // Using a=0.45 and b=c=0.22 as default probabilities
            for (int i = 0; i < edges; i++) {
                while (!rMat(&A_, n, 0, n-1, 0, n-1,
                             0.45, 0.22, 0.22,
                             &gen, dist)) {}
                while (!rMat(&B_, n, 0, n-1, 0, n-1,
                             0.45, 0.22, 0.22,
                             &gen, dist)) {}
            }
        }

    private:
        bool rMat(std::vector<T>* M, int n, int x1, int x2, int y1, int y2,
                  float a, float b, float c, std::default_random_engine* gen,
                  std::uniform_real_distribution<double> dist) {
            // If a 1x1 submatrix, then add an edge and return out
            if (x1 >= x2 && y1 >= y2) {
                if (abs(M->at((y1 * n) + x1)) > 0.1) {
                    return false;
                } else {
                    // Add 1.0 if this is a binary graph, and a random real number otherwise
                    M->at((int) (y1 * n) + x1) = (isBinary_) ? 1.0 : (((rand() % 10000) / 100.0) - 50.0);
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
                                newA, newB, newC, gen, dist);
                } else if (randomNum < (a + b)) {
                    return rMat(M, n, ((xMidPoint < x2) ? xMidPoint + 1 : xMidPoint), x2, y1, yMidPoint,
                                newA, newB, newC, gen, dist);
                } else if (randomNum < (a + b + c)) {
                    return rMat(M, n, x1, xMidPoint, ((yMidPoint < y2) ? yMidPoint + 1 : yMidPoint), y2,
                                newA, newB, newC, gen, dist);
                } else {
                    return rMat(M, n, ((xMidPoint < x2) ? xMidPoint + 1 : xMidPoint), x2,
                                ((yMidPoint < y2) ? yMidPoint + 1 : yMidPoint), y2, newA, newB, newC,
                                gen, dist);
                }
            }
        }

        /** Make a class to the BLAS Library Kernel. */
        virtual void callKernel() override {
            /** A naive implementation of a SpGEMM. Alpha and Beta are always 1 and 0
             * respectively.
             * Operation takes the form of C[N,N] = A[N,N] * B[N,N].
             * A return value is required to ensure that the compiler does not optimise
             * away this function. */
            int x, y, z;
            T acc;
            for (x = 0; x < n_; x++) {
                for (y = 0; y < n_; y++) {
                    acc = 0.0;
                    for (z = 0; z < n_; z++) {
                        acc += A_[x * n_ + z] * B_[z * n_ + y];
                    }
                    C_[x * n_ + y] = acc;
                }
            }
        }

        /** Call the extern consume() function. */
        virtual void callConsume() override {
            consume((void*)A_.data(), (void*)B_.data(), (void*)C_.data());
        }

        void printGraph(std::vector<T> M, int n) {
            std::cout << "____Printing Matrix____" << std::endl;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    std::cout << M[(i * n) + j] << " ";
                }
                std::cout << std::endl;
            }
        }

        /** Input matrix A. */
        std::vector<T> A_;

        /** Input matrix B. */
        std::vector<T> B_;

        /** Input matrix C. */
        std::vector<T> C_;

        bool isBinary_;

    };

#endif
}  // namespace cpu