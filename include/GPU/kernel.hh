#pragma once

#include <chrono>

namespace gpu {

/** A generic abstract class defining the operation of timing a BLAS kernel for
 * n iterations. */
template <typename T>
class kernel {
 public:
  kernel(const int iters) : iterations_(iters) {}

  /** Call the BLAS kernel n times, with 1 warmup run.
   * Returns the time elapsed for n BLAS calls in seconds. */
  double compute() {
    // Start timer
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime =
        std::chrono::high_resolution_clock::now();

    // Perform all GPU BLAS calls & callConsume()
    callKernel(iterations_);

    // Stop Timer
    std::chrono::time_point<std::chrono::high_resolution_clock> endTime =
        std::chrono::high_resolution_clock::now();
    // Get time elapsed in seconds
    std::chrono::duration<double> time_s = endTime - startTime;
    return time_s.count();
  }

 private:
  /** Make a call to the BLAS Library Kernel. */
  virtual void callKernel(const int iterations) = 0;

  /** Call the extern consume() function. */
  virtual void callConsume() = 0;

  /** The number of iterations to perform per problem size. */
  const int iterations_;
};
}  // namespace gpu