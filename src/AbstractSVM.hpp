//
//  AbstractSVM.hpp
//  SVM
//
//  Created by David Galvao on 24/09/22.
//

#ifndef AbstractSVM_hpp
#define AbstractSVM_hpp

#include "svm.hpp"

class AbstractSVM {
public:
    AbstractSVM(const svm_parameter *);
    
    virtual void fit(const svm_problem &) = 0;
    
    double *predict(const svm_problem &);
    double *predict(const svm_model *, const svm_problem &);
    double predict(const svm_node *);
    double predict(const svm_model *, const svm_node *);
    
protected:
    const svm_parameter *params;
    svm_model *model;
};
#endif /* AbstractSVM_hpp */
