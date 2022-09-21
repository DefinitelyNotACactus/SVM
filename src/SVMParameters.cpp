//
//  SVMParameters.cpp
//  SVM
//
//  Created by David Galvao on 18/09/22.
//

#include "SVMParameters.hpp"
#include "Util.hpp"

#include <math.h>
#include <unordered_set>
#include <iostream>

SVMParameters::SVMParameters(const svm_model *model, const svm_problem &problem, bool margin) : model(model), l(model->l), N(model->l), d(problem.d), X(new svm_node*[problem.l]), y(new double[problem.l]), alpha(new double[problem.l]), sumB(0), XSV0(new svm_node*[problem.l]), XSV1(new svm_node*[problem.l]) {
    for(int i = 0; i < problem.l; i++) {
        y[i] = problem.y[i];
        alpha[i] = 0;
        X[i] = new svm_node[d + 1];
        for(int j = 0; j <= d; j++) {
            X[i][j] = problem.x[i][j];
        }
    }
    
    separateVS();
    computeB();
    if (margin) { computeMargin(); }
}

double SVMParameters::kernel(const svm_node *xm, const svm_node *xn) {
    double *vdiff;
    double vdot;
    switch (model->param.kernel_type) {
        case RBF:
            vdiff = diff(d, xn, xm);
            vdot = dot(d, vdiff, vdiff);
            delete [] vdiff;
            return exp(-((double) 1 / (double) d) * pow(sqrt(vdot), 2));
        case POLY:
            return (((double) 1 / (double) d) * dot(d, xn, xm) + pow(model->param.coef0, model->param.degree));
        default:
            return dot(d, xn, xm);
    }
}

void SVMParameters::computeB() {
    sumB = 0;
    for(int i = 0; i < l; i++) {
        sumB += model->sv_coef[0][i] * kernel(model->SV[i], XSV);
    }
}

double SVMParameters::computeLeftHand(const svm_node *xn, double yn) {
    double sumW = 0;
    for(int i = 0; i < l; i++) {
        sumW += model->sv_coef[0][i] * kernel(model->SV[i], xn);
    }
    return yn * (sumW + (double) 1 / (double) ysv - sumB);
}

void SVMParameters::print() {
    for(int i = 0; i < l; i++) {
        std::cout << "Alpha: " << model->sv_coef[0][i] << " - ID: " << model->sv_indices[i] - 1 << "\n";
    }
}

void SVMParameters::computeMargin() {
    double sumMinDist = 0;
    for(int i = 0; i < lenSV0; i++) {
        double minDist = INFINITY;
        for(int j = 0; j < lenSV1; j++) {
            double *svDiff = diff(d, XSV0[i], XSV1[j]);
            double dist = norm(d, svDiff);
            if (minDist > dist) {
                minDist = dist;
            }
            delete [] svDiff;
        }
        sumMinDist += minDist;
    }
    margin = (sumMinDist / (double) lenSV0) / (double) 2;
    std::cout << "Margin: " << margin << "\n";
}

void SVMParameters::separateVS() {
    bool svSelected = false;
    std::unordered_set<int> svNoMargin;
    int backIndexSV0 = 0, backIndexSV1 = 0;
    for(int i = 0; i < N; i++) {
        double alpha_ = abs(model->sv_coef[0][i]);
        int idSV = model->sv_indices[i] - 1;
        alpha[idSV] = alpha_;
        if (alpha_ > 0 and alpha_ < model->param.C) {
            if(!svSelected) {
                XSV = X[idSV];
                ysv = y[idSV];
                svSelected = true;
                continue;
            }
            if (model->sv_coef[0][i] < 0) {
                XSV0[backIndexSV0] = new svm_node[d + 1];
                for(int j = 0; j <= d; j++) {
                    XSV0[backIndexSV0][j] = X[idSV][j];
                }
//                XSV0[backIndexSV0][d] = svm_node();
//                XSV0[backIndexSV0] = X[idSV];
                backIndexSV0++;
            } else {
                XSV1[backIndexSV1] = new svm_node[d + 1];
                for(int j = 0; j <= d; j++) {
                    XSV1[backIndexSV1][j] = X[idSV][j];
                }
//                XSV1[backIndexSV1][d] = svm_node();
//                XSV1[backIndexSV1] = X[idSV];
                backIndexSV1++;
            }
        } else {
            svNoMargin.insert(idSV);
        }
    }
    //svm_node **X = new svm_node*[N - svNoMargin.size()];
    int backIndex = 0;
    //double *y = new double[N - svNoMargin.size()];
    for(int i = 0; i < N; i++) {
        if(svNoMargin.find(i) == svNoMargin.end()) {
            //X[backIndex] = new svm_node[d + 1];
            y[backIndex] = y[i];
            for(int j = 0; j <= d; j++) {
                X[backIndex][j] = X[i][j];
            }
            backIndex++;
        }
    }
    
//    for(int i = 0; i < N; i++) {
//        delete [] this->X[i];
//    }
//    delete [] this->X;
//    delete [] this->y;
    
//    this->X = X;
//    this->y = y;
    N -= svNoMargin.size();
    lenSV0 = backIndexSV0;
    lenSV1 = backIndexSV1;
}
