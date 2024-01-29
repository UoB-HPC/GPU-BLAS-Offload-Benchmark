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
        virtual void initialise(int n, double sparsity) override {
            n_ = n;

            A_.reserve(n * n);
            B_.reserve(n * n);
            C_.reserve(n * n);

            // Random number generator objects for use in descent
            std::default_random_engine gen;
            std::uniform_real_distribution<double> dist(0.0, 1.0);

            // Initialise the matrices
            // Using a=0.45 and b=c=0.22 as default probabilities
            int edges = (int)(n * n * (1 - sparsity));
            for (int i = 0; i < edges; i++) {
                while (!rMat(&A_, n, 0, n-1, 0, n-1,
                             0.45, 0.22, 0.22,
                             gen, dist)) {}
                while (!rMat(&B_, n, 0, n-1, 0, n-1,
                             0.45, 0.22, 0.22,
                             gen, dist)) {}
            }
        }

    private:
        bool rMat(std::vector<T>* M, int n, int x1, int x2, int y1, int y2,
                  float a, float b, float c, std::default_random_engine gen,
                  std::uniform_real_distribution<double> dist) {
            // If a 1x1 submatrix, then add an edge and return out
            if (x1 >= x2 && y1 <= y2) {
                if (M[(y1 * n) + x1] == 1) {
                    return false;
                } else {
                    M[(y1 * n) + x1] = 1;
                    return true;
                }
            } else {
                // Divide up the matrix
                int xMidPoint = x1 + (int)((x2 - x1) / 2);
                int yMidPoint = y1 + (int)((y2 - y1) / 2);
                // ToDo - consider if need to check for non-square matrices
                // Introduce some noise to the quarter probabilities
                float newA = a + (-0.01 + (dist(gen) * 0.02));
                float newB = b + (-0.01 + (dist(gen) * 0.02));
                float newC = c + (-0.01 + (dist(gen) * 0.02));
                // Make sure noise doesn't make impossible probabilities
                if ((newA + newB + newC) > 0.98 ||
                    newA < 0.02 || newB < 0.02 || newC < 0.02) {
                    newA = 0.45;
                    newB = 0.22;
                    newC = 0.22;
                }
                // Work out which quarter to recurse into
                float randomNum = dist(gen);
                if (randomNum < a) {
                    return rMat(M, n, x1, xMidPoint, y1, yMidPoint,
                                newA, newB, newC);
                } else if (randomNum < (a + b)) {
                    return rMat(M, n, xMidPoint, x2, y1, yMidPoint,
                                newA, newB, newC);
                } else if (randomNum < (a + b + c)) {
                    return rMat(M, n, x1, xMidPoint, yMidPoint, y2,
                                newA, newB, newC);
                } else {
                    return rMat(M, n, xMidPoint, x2, yMidPoint, y2,
                                newA, newB, newC);
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

        /** Input matrix A. */
        std::vector<T> A_;

        /** Input matrix B. */
        std::vector<T> B_;

        /** Input matrix C. */
        std::vector<T> C_;

    };
#endif
}  // namespace cpu