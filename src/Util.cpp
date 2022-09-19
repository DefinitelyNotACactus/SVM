//
//  Util.cpp
//  SVM
//
//  Created by David Galvao on 18/09/22.
//

#include "Util.hpp"

#include <math.h>

double dot(const int size, const svm_node *xn, const svm_node *xm) {
    double product = 0;
    for(int i = 0; i < size; i++) {
        product += xn[i].value * xm[i].value;
    }
    return product;
}

double dot(const int size, const double *xn, const double *xm) {
    double product = 0;
    for(int i = 0; i < size; i++) {
        product += xn[i] * xm[i];
    }
    return product;
}

double norm(const int size, const double *x) {
    double sumVector = 0;
    for(int i = 0; i < size; i++) {
        sumVector += x[i] * x[i];
    }
    return sqrt(sumVector);
}

double *diff(const int size, const svm_node *xn, const svm_node *xm) {
    double *diff = new double[size];
    for(int i = 0; i < size; i++) {
        diff[i] = xn[i].value - xm[i].value;
    }
    return diff;
}
