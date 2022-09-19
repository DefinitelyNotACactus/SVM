//
//  DatasetReader.hpp
//  SVM
//
//  Created by David Galvao on 17/09/22.
//

#ifndef DatasetReader_hpp
#define DatasetReader_hpp

#include <string>
#include <vector>
#include <unordered_map>

struct DataFrame {
    int size;
    std::vector<std::string> columns;
    std::unordered_map<std::string, std::vector<double>> entries;
    
    std::vector<double> operator [] (const std::string &index) const  {
        return entries.at(index);
    }
};

DataFrame readCsv(const std::string &, int indexColumn=0, char delimiter=',');

#endif /* DatasetReader_hpp */
