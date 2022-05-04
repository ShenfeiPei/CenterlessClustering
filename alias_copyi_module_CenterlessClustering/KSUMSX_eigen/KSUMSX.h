#ifndef _KSUMSX_H
#define _KSUMSX_H

#include <iostream>
#include <algorithm>
#include <fstream>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include <numeric>
#include <functional>
#include <set>
#include <ctime>
#include <chrono>
#include "Eigen400/Eigen/Dense"
#include "Eigen400/Eigen/Core"

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::VectorXd;
using Eigen::RowVectorXd;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matdr;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Matdc;
typedef RowVectorXd Vecdr;
typedef VectorXd Vecdc;

class KSUMSX{
public:
    int N = 0;
    int dim = 0;
    int c_true = 0;
    int debug = 0;

    // X: (dim, N) ColMajor
    Matdr X;
    Vecdr xnorm;

    // S: (c, dim) RowMajor
    Matdr S;
    Vecdc n;
    Vecdc v;

    std::vector<std::vector<int>> Y;
    std::vector<double> time_arr;
    std::vector<int> iter_arr;

    KSUMSX();
    KSUMSX(std::vector<std::vector<double>> &X, int c_true, int debug);
    ~KSUMSX();

    void init(std::vector<int> &y);
    void opt(std::vector<std::vector<int>> &Y, int block_size, int ITER);
    int opt_once(std::vector<int> &y, int block_size, int ITER);

    int update_parallel(std::vector<int> &y, int block_size, int ITER);
};
#endif
