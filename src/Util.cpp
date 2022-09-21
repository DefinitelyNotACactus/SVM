//
//  Util.cpp
//  SVM
//
//  Created by David Galvao on 18/09/22.
//

#include "Util.hpp"

#include <math.h>
#include <random>

std::random_device rd = std::random_device{};
std::default_random_engine rng = std::default_random_engine{ rd() };

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

double accuracyScore(const int size, const double *h, const double *y) {
    int rightPredictions = 0;
    for(int i = 0; i < size; i++) {
        if(h[i] == y[i]) {
            rightPredictions += 1;
        }
    }
    return (double) rightPredictions / (double) size;
}

std::vector<int> randomChoice(const int range, const int samples) {
    if(samples <= 0) {
        return std::vector<int>();
    }
    std::vector<int> v(range);
    std::iota(v.begin(), v.end(), 0);
    if (samples > range) {
        return v;
    }
    std::shuffle(v.begin(), v.end(), rng);
    return std::vector<int>(v.begin(), v.begin() + samples);
}
