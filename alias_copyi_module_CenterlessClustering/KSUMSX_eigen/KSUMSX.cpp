#include "KSUMSX.h"

KSUMSX::KSUMSX(){}

KSUMSX::KSUMSX(std::vector<std::vector<double>> &X, int c_true, int debug){
    this->N = X.size();
    this->dim = X[0].size();
    this->c_true = c_true;
    this->debug = debug;

    this->X.resize(dim, N);
    for (int i = 0; i < N; i++){
        this->X.col(i) = VectorXd::Map(&X[i][0], X[i].size());
    }
    this->xnorm.resize(N);

    S.resize(c_true, dim);
    n.resize(c_true);
    v.resize(c_true);

    // Matdr Ar(3, 2);
    // Ar << 1, 2,
    //      3, 4,
    //      5, 6;

    // Matdc Ac(3, 2);
    // Ac << 1, 2,
    //     3, 4,
    //     5, 6;

    // Matdc Mc(3, 3);
    // Mc << 1, 2, 4,
    //       4, 5, 7,
    //       4, 5, 7;

    // VectorXd vc(3);
    // vc << 1, 2, 3;

    // RowVectorXd vr(3);
    // vr << 2, 4, 6;

    // std::cout << "Ar * Mc.col(0)" << std::endl;
    // std::cout << Ar * Mc.col(0) << std::endl;

    // std::cout << "Ar * Ar.row(0)" << std::endl;
    // std::cout << Ar * Ar.row(0) << std::endl;

    // std::cout << "Ar * Ac.row(0)" << std::endl;
    // std::cout << Ar * Ac.row(0) << std::endl;

    // std::cout << "Ar * Ar.row(0).t" << std::endl;
    // std::cout << Ar * Ar.row(0).transpose() << std::endl;

    // std::cout << "vc * vr + vc" << std::endl;
    // Mc(Eigen::all, Eigen::seq(0, 2)) = (vc * vr).colwise() + vc;
    // std::cout << Mc << std::endl;

        //   n * xnorm(i:j) + v - 2 *  S  *  X(i:j) 
        //   .   . . .        .       ...     ...
        //   .                .       ...     ...
}

KSUMSX::~KSUMSX() {}

void KSUMSX::init(std::vector<int> &y){
    // init y
    std::generate(y.begin(), y.end(), [=](){return rand() % c_true;});

    // init n, v, S
    n.setZero();
    v.setZero();
    S.setZero();
    int tmp_c;
    for (int i = 0; i < N; i++){
        tmp_c = y[i];
        n(tmp_c) ++;
        v(tmp_c) += xnorm(i);
        S.row(tmp_c) += X.col(i);
    }

}

int KSUMSX::opt_once(std::vector<int> &y, int block_size, int ITER){

    Eigen::setNbThreads(12);

    init(y);

    // std::chrono::time_point<std::chrono::steady_clock> t1;
    // std::chrono::time_point<std::chrono::steady_clock> t2;

    // t1 = std::chrono::steady_clock::now();
    int Iter = update_parallel(y, block_size, ITER);
    if (debug == 1){
        std::cout << "ITER = " << Iter << std::endl;
    }
    // t2 = std::chrono::steady_clock::now();
    // double time_1 = std::chrono::duration<double>(t2 - t1).count();
    // std::cout << "parallel time " << time_1 << std::endl;

    return Iter;
}

void KSUMSX::opt(std::vector<std::vector<int>> &Y, int block_size, int ITER){
    int rep = Y.size();
    this->Y = Y;

    std::chrono::time_point<std::chrono::steady_clock> t1;
    std::chrono::time_point<std::chrono::steady_clock> t2;

    t1 = std::chrono::steady_clock::now();
    #pragma omp parallel for
    for (int i = 0; i < N; i++){
        xnorm[i] = X.col(i).squaredNorm();
    }
    t2 = std::chrono::steady_clock::now();
    double time_1 = std::chrono::duration<double>(t2 - t1).count();
    // std::cout << "xnorm time " << time_1 << std::endl;

    // Y
    time_arr.resize(rep);
    iter_arr.resize(rep);

    for (int rep_i = 0; rep_i < rep; rep_i++){
        t1 = std::chrono::steady_clock::now();
        int iter = opt_once(this->Y[rep_i], block_size, ITER);
        t2 = std::chrono::steady_clock::now();

        iter_arr[rep_i] = iter;
        time_arr[rep_i] = time_1 + std::chrono::duration<double>(t2 - t1).count();
    }
}

int KSUMSX::update_parallel(std::vector<int> &y, int block_size, int ITER){
    int c_old = 0, c_new = 0, converge = 0;
    std::ptrdiff_t c_new_ptr;
    double tmp_d;

    int n_block = N / block_size;
    if (n_block * block_size < N){
        n_block ++;
    }
    int i = 0, j = 0;

    int block_size_true = 0;
    Matdr t(c_true, block_size);

    std::set<int> change_ind;
    std::vector<int> change_ind_vec;
    int Iter;
    for (Iter = 0; Iter < ITER; Iter++){
        converge = 0;
        for (int block = 0; block < n_block; block++){
            i = block * block_size;
            j = (block + 1) * block_size - 1;
            if (j >= N){
                j = N - 1;
            }

            //   n * xnorm(i:j) + v - 2 *  S  *  X(i:j) 
            //   .   . . .        .       ...     ...
            //   .                .       ...     ...
            block_size_true = j - i + 1;

            t(Eigen::all, Eigen::seq(0, block_size_true - 1)) = (n * xnorm(Eigen::seq(i, j)) - 2 * S * X(Eigen::all, Eigen::seq(i, j))).colwise() + v;

            change_ind.clear();

            for (int k = 0; k < block_size_true; k++){

                change_ind_vec.clear();
                change_ind_vec.assign(change_ind.begin(), change_ind.end());
                t(change_ind_vec, k) = n(change_ind_vec) * xnorm(i+k) - 2 * S(change_ind_vec, Eigen::all) * X.col(i+k) + v(change_ind_vec);

                tmp_d = t.col(k).minCoeff(& c_new_ptr);
                c_old = y[i + k];
                c_new = (int) c_new_ptr;

                if (c_old != c_new){
                    change_ind.insert(c_old);
                    change_ind.insert(c_new);

                    converge ++;

                    y[i+k] = c_new;

                    S.row(c_old) -= X.col(i+k);
                    S.row(c_new) += X.col(i+k);

                    v(c_old) -= xnorm(i+k);
                    v(c_new) += xnorm(i+k);

                    n(c_old) --;
                    n(c_new) ++;
                }
            }
        }

        if (converge < N / 1000){
            break;
        }
    }
    return Iter;
}
