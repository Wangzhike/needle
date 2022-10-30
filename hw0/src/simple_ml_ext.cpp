#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <functional>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    // auto print_matrix = [](const float* matrix, size_t nr, size_t nc, size_t r, size_t n) {
    //   for (size_t i = r; i < r + n && i < nr; ++i) {
    //     for (size_t j = 0; j < nc; ++j) {
    //       std::cout << matrix[i*nc + j] << ",";
    //     }
    //     std::cout << std::endl;
    //   }
    //   std::cout << std::endl;
    // };

    auto matmul = [] (const float* ma, const float* mb, float* mc,
        size_t m, size_t n, size_t k) {
      /** Matrix Multiply
       * @param ma: input, size (m, n)
       * @param mb: input, size (n, k)
       * @param mc: output, size (m, k)
       */
       for (size_t i = 0; i < m; ++i) {
         for (size_t j = 0; j < k; ++j) {
           mc[i*k+j] = 0;
           for (size_t l = 0; l < n; ++l) {
             mc[i*k+j] += ma[i*n+l] * mb[l*k+j];
           }
         }
       }
    };

    int* Y = new int[m*k];
    for (size_t i = 0; i < m; ++i) {
      size_t label = static_cast<size_t>(y[i]);
      for (size_t j = 0; j < k; ++j) {
        if (label == j) {
          Y[i*k+j] = 1;
        } else {
          Y[i*k+j] = 0;
        }
      }
    }

    size_t n_batch = (m / batch) + 1;
    /// loop over minibatch
    for (size_t i = 0; i < n_batch; ++i) {
      size_t start = i * batch;
      size_t end = start + batch < m ? start + batch : m;
      if (start >= end) {
        break;
      }
      size_t mini_batch = end - start;
      const float* X_batch = X + start * n;
      const int* Y_batch = Y + start * k;
      float* Z_batch = new float[mini_batch*k];

      /// forward propagation
      /// Z = np.matmul(X, theta)
      matmul(X_batch, theta, Z_batch, mini_batch, n, k);
      // for (size_t r = 0; r < mini_batch; ++r) {
      //   for (size_t c = 0; c < k; ++c) {
      //     Z_batch[r*k+c] =0;
      //     for (size_t p = 0; p < n; ++p) {
      //       Z_batch[r*k+c] += X_batch[r*n+p] * theta[p*k+c];
      //     }
      //   }
      // }

      /// Z = np.exp(Z)
      for (size_t r = 0; r < mini_batch; ++r) {
        for (size_t c = 0; c < k; ++c) {
          Z_batch[r*k+c] = std::exp(Z_batch[r*k+c]);
        }
      }
      /// Z = normalize(Z)
      for (size_t r =0; r < mini_batch; ++r) {
        float Z_batch_sum = 0;
        for (size_t c = 0; c < k; ++c) {
          Z_batch_sum += Z_batch[r*k+c];
        }
        for (size_t c = 0; c < k;++c) {
          Z_batch[r*k+c] = Z_batch[r*k+c] / Z_batch_sum;
        }
      }

      /// backward propagation
      // /// Z - Y
      // float* Z_Y_batch = new float[mini_batch*k];
      // for (size_t r = 0; r < mini_batch; ++r) {
      //   for (size_t c = 0; c < k; ++c) {
      //     Z_Y_batch[r*k+c] = Z_batch[r*k+c] - Y_batch[r*k+c];
      //   }
      // }
      // /// X_batch_T
      // float* X_batch_T = new float[n*mini_batch];
      // for (size_t r = 0; r < n; ++r) {
      //   for (size_t c = 0; c < mini_batch; ++c) {
      //     X_batch_T[r*mini_batch+c] = X_batch[c*n+r];
      //   }
      // }

      /// d_theta
      float* d_theta = new float[n*k];
      // matmul(X_batch_T, Z_Y_batch, d_theta, n, mini_batch, k);
      // for (size_t r = 0; r < n; ++r) {
      //   for (size_t c = 0; c < k; ++c) {
      //     d_theta[r*k+c] = d_theta[r*k+c] / mini_batch;
      //   }
      // }
      for (size_t r = 0; r < n; ++r) {
        for (size_t c = 0; c < k; ++c) {
          d_theta[r*k+c] = 0;
          for (size_t p = 0; p < mini_batch; ++p) {
            d_theta[r*k+c] += X_batch[p*n+r] * (Z_batch[p*k+c] - Y_batch[p*k+c]);
          }
          d_theta[r*k+c] = d_theta[r*k+c] / mini_batch;
        }
      }

      /// SGD
      for (size_t r = 0; r < n; ++r) {
        for (size_t c = 0; c < k; ++c) {
          theta[r*k+c] -= lr * d_theta[r*k+c];
        }
      }
      delete[] d_theta;
      // delete[] X_batch_T;
      // delete[] Z_Y_batch;
      delete[] Z_batch;
    }
    delete[] Y;
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
