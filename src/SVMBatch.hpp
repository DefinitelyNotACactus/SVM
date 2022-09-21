//
//  SVMBatch.hpp
//  SVM
//
//  Created by David Galvao on 19/09/22.
//

#ifndef SVMBatch_hpp
#define SVMBatch_hpp

#include "svm.hpp"

#include <tuple>

class SVMBatch {
public:
    SVMBatch(svm_parameter *, double, double);
    
    void fit(const svm_problem &);
    double *predict(const svm_problem &);
    double *predict(const svm_model *, const svm_problem &);
    double predict(const svm_node *);
    double predict(const svm_model *, const svm_node *);
    
private:
    svm_parameter *params;
    svm_model *model;
    double pct, epsilon;
    
    std::tuple<svm_node **, int, svm_node **, int> separate_classes(svm_problem &);
};
#endif /* SVMBatch_hpp */
