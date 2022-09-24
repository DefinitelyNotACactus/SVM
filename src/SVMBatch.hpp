//
//  SVMBatch.hpp
//  SVM
//
//  Created by David Galvao on 19/09/22.
//

#ifndef SVMBatch_hpp
#define SVMBatch_hpp

#include "AbstractSVM.hpp"

#include <tuple>

class SVMBatch : public AbstractSVM {
public:
    SVMBatch(svm_parameter *, double, double);
    
    void fit(const svm_problem &);
    
private:
    double pct, epsilon;
    
    std::tuple<svm_node **, int, svm_node **, int> separate_classes(svm_problem &);
};
#endif /* SVMBatch_hpp */
