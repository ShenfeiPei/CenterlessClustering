#include "KSUMS.h"
#include <chrono>

KSUMS::KSUMS(){}

KSUMS::KSUMS(std::vector<std::vector<int>> &NN, std::vector<std::vector<double>> &NND, int c_true, bool debug, double max_dd){
    this->N = NN.size();
    this->c_true = c_true;
    this->NN = NN;
    this->NND = NND;
    this->debug = debug;

    //check NND
    for (int i = 0; i < N; i++){
        if (NN[i][0] != i){
            std::cout << "Error opening file" << std::endl;
            exit(EXIT_FAILURE);
        }
    }


    hi = vector<double>(c_true, 0);
    hi_TF = vector<int>(c_true, 0);
    hi_count = vector<int>(c_true, 0);
    knn2c = vector<int>(N, 0);

    y = vector<int>(N, 0);

    srand((unsigned)time(NULL));

    if (max_dd > 0){
        this->max_d = max_dd;
    }else{
        this->max_d = maximum_2Dvec(NND);
    }

    symmetry(this->NN, this->NND);

    if (debug){
        cout << "max_d = " << max_d << endl;
    }

}

KSUMS::~KSUMS() {}

double KSUMS::maximum_2Dvec(vector<vector<double>> &Vec){
    int N = Vec.size();
    vector<double> tmp(N, 0);

    for(int i = 0; i < N; i++){
        tmp[i] = *max_element(Vec[i].begin(), Vec[i].end());
    }

    double ret = *max_element(tmp.begin(), tmp.end());
    return ret;
}

void KSUMS::symmetry(vector<vector<int>> &NN, vector<vector<double>> &NND){
    int N = NN.size();
    int knn = NN[0].size();

    vector<vector<int>> RNN;
    vector<vector<double>> RNND;
    RNN.resize(N);
    RNND.resize(N);

    int tmp_j = 0;
    double tmp_d = 0;
    for (int i = 0; i < N; i++){
        for (int k = 0; k < knn; k++){
            tmp_j = NN[i][k];
            tmp_d = NND[i][k];
            RNN[tmp_j].push_back(i);
            RNND[tmp_j].push_back(tmp_d);
        }
    }

    vector<bool> flag(N, false);
    for (int i = 0; i < N; i++){
        for (auto j : NN[i]){
            flag[j] = true;
        }

        for (int k = 0; k < RNN[i].size(); k++){
            tmp_j = RNN[i][k];
            if (flag[tmp_j] == false){

                NN[i].push_back(tmp_j);

                tmp_d = RNND[i][k];
                NND[i].push_back(tmp_d);
            }
        }

        for (int k = 0; k < knn; k++){
            tmp_j = NN[i][k];
            flag[tmp_j] = false;
        }
    }
}

void KSUMS::init(){
    fill(y.begin(), y.end(), -1);
    vector<int> n(NN.size(), 0);

    int n_up = NN.size() / c_true;

    vector<int> tmpL;
    int id = 0;
    int num_clu = 0;
    int flag = 0;

    for (int i = 0; i < NN.size(); i++) if (y[i] == -1){
        tmpL.push_back(i);
        y[i] = num_clu;
        n[num_clu] ++;
        flag = 0;

        while (tmpL.size() > 0 and flag==0){
            id = tmpL.back();
            tmpL.pop_back();

            for (auto ele : NN[i]) if (y[ele] == -1){
                tmpL.push_back(ele);
                y[ele] = num_clu;
                n[num_clu] += 1;

                if (n[num_clu] >= n_up){
                    flag = 1;
                    tmpL.clear();
                    break;
                }
            }
        }
        num_clu ++;
    }


    if (debug){
        cout << "num_clu = " << num_clu << ", (c = " << c_true << ")" << endl;
    }


    if (num_clu > c_true){
        KO = Keep_order(y, N, num_clu);

        vector<int> old2new(num_clu, -1);

        int tmp_old_c, tmp_new_c;
        for (int i = 0; i < num_clu - c_true; i++){
            tmp_old_c = KO.o2c[i];
            old2new[tmp_old_c] = rand() % c_true;
        }

        for (int i = num_clu - c_true; i < num_clu; i++){
            tmp_old_c = KO.o2c[i];
            old2new[tmp_old_c] = i - num_clu + c_true;
        }

        int c_old, c_new;
        for (int i = 0; i < N; i++){
            c_old = y[i];
            c_new = old2new[c_old];
            y[i] = c_new;
        }
    }
}


double KSUMS::opt_once(int ITER, int our_init){
    if (our_init == 1){
        init();
    }else{
        std::generate(y.begin(), y.end(), [&]() {return rand() % c_true;});
    }

    KO = Keep_order(y, N, c_true);
    // std::cout << "hh" << std::endl;

    chrono::milliseconds total_t = chrono::milliseconds(0);
    chrono::time_point<chrono::steady_clock> t1;
    chrono::time_point<chrono::steady_clock> t2;

    t1 = chrono::steady_clock::now();
    int iter = 0, c_old = 0, c_new = 0, converge = 1;

    for (iter = 0; iter < ITER; iter++){
        // std::cout << "iter = " << iter << std::endl;
        converge = 1;
        for (int sam_i = 0; sam_i < N; sam_i++){
            c_old = y[sam_i];

            c_new = find_c_new(sam_i);

            if (c_new != c_old){
                converge = 0;
                y[sam_i] = c_new;
                KO.sub(KO.c2o[c_old]);
                KO.add(KO.c2o[c_new]);
            }
        }
        if (converge == 1){
            break;
        }
    }
    t2 = chrono::steady_clock::now();

    return chrono::duration<double>(t2 - t1).count();
}

void KSUMS::opt(int rep, int ITER, int our_init){
    Y.resize(rep);
    time_arr.resize(rep);

    chrono::milliseconds total_t = chrono::milliseconds(0);
    double time_once;
    for (int rep_i = 0; rep_i < rep; rep_i++){
        time_once = opt_once(ITER, our_init);
        time_arr[rep_i] = time_once;
        Y[rep_i] = y;
    }
}


int KSUMS::find_c_new(int sam_i){

    int c_new, tmp_c, tmp_ni, tmp_nb;
    int h_min_ind = 0;
    double h_min_val = 0;

    for (int k = 0; k < NN[sam_i].size(); k++){
        tmp_nb = NN[sam_i][k];
        knn2c[k] = y[tmp_nb];
    }

    for (int k = 0; k < NN[sam_i].size(); k++){
        tmp_c = knn2c[k];
        hi[tmp_c] = 0;
        hi_count[tmp_c] = 0;
        hi_TF[tmp_c] = 0;
    }

    for (int k = 0; k < NN[sam_i].size(); k++){
        tmp_c = knn2c[k];
        hi[tmp_c] += NND[sam_i][k];
        hi_count[tmp_c] += 1;
    }

    for (int k = 0; k < NN[sam_i].size(); k++){
        tmp_c = knn2c[k];
        if (hi_TF[tmp_c] == 0){
            hi_TF[tmp_c] = 1;

            tmp_ni = KO.o2ni[KO.c2o[tmp_c]];
            hi[tmp_c] += (tmp_ni - hi_count[tmp_c]) * max_d;
        }
    }

    h_min_ind = knn2c[0];
    h_min_val = hi[h_min_ind];
    for (int k = 1; k < NN[sam_i].size(); k++){
        tmp_c = knn2c[k];
        if (hi[tmp_c] < h_min_val){
            h_min_ind = tmp_c;
            h_min_val = hi[h_min_ind];
        }
    }

    if (KO.o2ni[0] * max_d < h_min_val){
        c_new = KO.o2c[0];
    }else{
        c_new = h_min_ind;
    }

    return c_new;
}
