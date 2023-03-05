#ifndef _KEEP_ORDER_H
#define _KEEP_ORDER_H

#include <algorithm>
#include <vector>

class Keep_order{
public:
    int N = 0;
    int c = 0;
    int *o2ni = nullptr;
    int *o2c = nullptr;
    int *c2o = nullptr;

    typedef struct lr_ind_t{
        int l = -1;
        int r = -1;
    }lr_ind;

    lr_ind *ni2o = nullptr;

    Keep_order();
    Keep_order(std::vector<int> &y, int N, int c);
    ~Keep_order();

    void argsort_f(int *v, int n, int *ind);
    void sub(int id);
    void add(int id);
};

#endif // _KEEP_ORDER_H
