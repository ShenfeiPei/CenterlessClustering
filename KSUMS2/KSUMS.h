#ifndef _KSUMS_H
#define _KSUMS_H

#include <iostream>
#include <algorithm>
#include <fstream>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include <numeric>
#include <functional>
#include <ctime>
#include <chrono>
#include "Keep_order.h"

using namespace std;

class KSUMS{
public:
    int N = 0;
    int c_true = 0;
    bool debug;

    vector<vector<int>> NN;
    vector<vector<double>> NND;
    vector<vector<int>> Y;
    vector<int> y;
    vector<double> time_arr;

    double t = 0;
    double sigma = 0;

    vector<double> hi;
    vector<int> hi_TF;
    vector<int> hi_count;

    vector<int> knn2c;

    double max_d = 0;

    Keep_order KO;

    KSUMS();
    KSUMS(std::vector<std::vector<int>> &NN, std::vector<std::vector<double>> &NND, int c_true, bool debug, double max_dd);
    ~KSUMS ();

    double maximum_2Dvec(vector<vector<double>> &Vec);
    void symmetry(vector<vector<int>> &NN, vector<vector<double>> &NND);
    // void init();
    void opt(int rep, int ITER, int our_init);
    double opt_once(int ITER, int our_init);
    void init();
    int find_c_new(int sam_i);

};
#endif
