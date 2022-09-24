//
//  AbstractSVM.cpp
//  SVM
//
//  Created by David Galvao on 24/09/22.
//

#include "AbstractSVM.hpp"

AbstractSVM::AbstractSVM(const svm_parameter *params) : params(params) { };

double * AbstractSVM::predict(const svm_model *model, const svm_problem &problem) {
    double *predictions = new double[problem.l];
    for(int i = 0; i < problem.l; i++) {
        predictions[i] = predict(model, problem.x[i]);
    }
    return predictions;
}

double * AbstractSVM::predict(const svm_problem &problem) {
    return predict(model, problem);
}

double AbstractSVM::predict(const svm_model *model, const svm_node *xn) {
    return svm_predict(model, xn);
}

double AbstractSVM::predict(const svm_node *xn) {
    return svm_predict(model, xn);
}
