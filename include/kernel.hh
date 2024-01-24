#pragma once

#include <chrono>

/** A generic abstract class defining the operation of timing a BLAS kernel for
 * n iterations. */
template <typename T>
class kernel {
 public:
  kernel(const int iters) : iterations_(iters) {}

  /** Call the BLAS kernel n times, with 1 warmup run.
   * Returns the time elapsed for n BLAS calls in seconds. */
  double compute() {
    // Warmup run
    callKernel();
    callConsume();

    // Start timer
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime =
        std::chrono::high_resolution_clock::now();

    // Perform all SGEMM iterations
    for (int i = 0; i < iterations_; i++) {
      callKernel();
      // Post iteration consume - ensures naive kernel isn't optimised away
      callConsume();
    }

    // Stop Timer
    std::chrono::time_point<std::chrono::high_resolution_clock> endTime =
        std::chrono::high_resolution_clock::now();
    // Get time elapsed in seconds
    std::chrono::duration<double> time_s = endTime - startTime;
    return time_s.count();
  }

 private:
  /** Make a call to the BLAS Library Kernel. */
  virtual void callKernel() = 0;

  /** Call the extern consume() function. */
  virtual void callConsume() = 0;

 protected:
  /** The number of iterations to perform per problem size. */
  const int iterations_;
};
