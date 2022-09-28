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
        X[i] = problem.x[i];
        y[i] = problem.y[i];
        alpha[i] = 0;
    }
    
    separateVS();
    computeB();
    if (margin) { computeMargin(); }
}

SVMParameters::~SVMParameters() {
    delete [] X;
    delete [] y;
    delete [] XSV0;
    delete [] XSV1;
    delete [] alpha;
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
        indicesSV.insert(idSV);
        alpha[idSV] = alpha_;
        if (alpha_ > 0 and alpha_ < model->param.C) {
            if(!svSelected) {
                XSV = X[idSV];
                ysv = y[idSV];
                svSelected = true;
                continue;
            }
            if (model->sv_coef[0][i] < 0) {
                XSV0[backIndexSV0] = X[idSV];
                backIndexSV0++;
            } else {
                XSV1[backIndexSV1] = X[idSV];
                backIndexSV1++;
            }
        } else {
            svNoMargin.insert(idSV);
        }
    }
    int backIndex = 0;
    for(int i = 0; i < N; i++) {
        if(svNoMargin.find(i) == svNoMargin.end()) {
            X[backIndex] = X[i];
            y[backIndex] = y[i];
            backIndex++;
        }
    }
    
    N -= svNoMargin.size();
    lenSV0 = backIndexSV0;
    lenSV1 = backIndexSV1;
}

bool SVMParameters::isSV(int index) {
    return (indicesSV.find(index) != indicesSV.end());// UX[i] is not a SV
}
