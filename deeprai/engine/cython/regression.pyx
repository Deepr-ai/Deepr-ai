import numpy as np
cimport numpy as np

cpdef linear_regression(np.ndarray[np.float64_t, ndim=1]x_vals, np.ndarray[np.float64_t, ndim=1]y_vals):
    cdef double x_mean, y_mean, numerator, denominator
    cdef int n_samples = x_vals.shape[0]

    # Calculate the mean of x and y
    x_mean = np.mean(x_vals)
    y_mean = np.mean(y_vals)

    # Calculate the numerator and denominator of the slope
    numerator = np.dot(x_vals, y_vals) - n_samples * x_mean * y_mean
    denominator = np.dot(x_vals, x_vals) - n_samples * x_mean * x_mean

    # Calculate the slope and intercept
    cdef double slope = numerator / denominator
    cdef double intercept = y_mean - slope * x_mean

    return slope, intercept


cpdef poly_regression(np.ndarray[np.float64_t, ndim=1] x_vals, np.ndarray[np.float64_t, ndim=1] y_vals,
                      int max_degree=10):
    cdef int n = x_vals.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] X = np.ones((n, max_degree + 1))
    cdef np.ndarray[np.float64_t, ndim=1] beta = np.zeros(max_degree + 1)
    cdef np.ndarray[np.float64_t, ndim=1] residuals = np.zeros(n)
    cdef int best_degree = 0
    cdef double best_rsq = 0.0

    # Create the design matrix X
    for degree in range(1, max_degree + 1):
        X[:, degree] = x_vals ** degree

        # Compute the OLS regression coefficients and residuals
        beta[:degree + 1] = np.linalg.inv(X[:, :degree + 1].T.dot(X[:, :degree + 1])).dot(
            X[:, :degree + 1].T.dot(y_vals))
        residuals = y_vals - X[:, :degree + 1].dot(beta[:degree + 1])

        # Compute the R-squared of the regression
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)
        rsq = 1.0 - ss_res / ss_tot
        # Update the best degree and R-squared if necessary
        if rsq > best_rsq:
            best_degree = degree
            best_rsq = rsq

    # Compute the OLS regression coefficients and return them
    beta[:best_degree + 1] = np.linalg.inv(X[:, :best_degree + 1].T.dot(X[:, :best_degree + 1])).dot(
        X[:, :best_degree + 1].T.dot(y_vals))
    return [int(round(beta[i])) for i in range(best_degree + 1)][::-1]



cpdef sine_regression(np.ndarray[np.float64_t, ndim=1] x_vals, np.ndarray[np.float64_t, ndim=1] y_vals):
    cdef double pi = np.pi
    cdef double amp, freq, phase, offset, residual, sine_val
    cdef int n = x_vals.shape[0]
    cdef int i

    # Define the sum variables
    cdef double sum_x = 0.0
    cdef double sum_y = 0.0
    cdef double sum_x2 = 0.0
    cdef double sum_xy = 0.0
    cdef double sum_sin = 0.0
    cdef double sum_cos = 0.0

    # Calculate the sums
    for i in range(n):
        sum_x += x_vals[i]
        sum_y += y_vals[i]
        sum_x2 += x_vals[i]**2
        sum_xy += x_vals[i] * y_vals[i]
        sum_sin += np.sin(x_vals[i])
        sum_cos += np.cos(x_vals[i])

    # Calculate the regression coefficients
    freq = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
    amp = (sum_y - freq * sum_x) / n
    offset = (sum_cos * sum_y - sum_sin * sum_xy) / (n * sum_cos**2 - sum_sin**2)
    phase = np.arctan(-sum_sin / sum_cos)


    return amp, freq, phase, offset