//
//  DatasetReader.cpp
//  SVM
//
//  Created by David Galvao on 17/09/22.
//

#include <fstream>
#include <iostream>
#include <sstream>

#include "DatasetReader.hpp"

DataFrame readCsv(const std::string &inputFile, int indexColumn, char delimiter) {
    std::ifstream input(inputFile);
    if(!input.is_open()) {
        std::cout << "Error while opening file: " << inputFile << "\n";
        exit(-1);
    }
    DataFrame df;
    std::string line, columnValue;
    int currentRow = 0, currentColumn;
    while(std::getline(input, line)) {
        std::stringstream stream(line);
        currentRow++;
        currentColumn = 0;
        if (currentRow == 1) { // Header
            while(std::getline(stream, columnValue, delimiter)) {
                if(currentColumn == indexColumn) {
                    columnValue = "Index";
                } else if(columnValue == "") {
                    columnValue = "Unnamed: " + std::to_string(currentColumn);
                }
                df.columns.push_back(columnValue);
                currentColumn++;
            }
        } else {
            while(std::getline(stream, columnValue, delimiter)) {
                if(currentColumn != indexColumn) {
                    df.entries[df.columns[currentColumn]].push_back(std::stod(columnValue));
                }
                currentColumn++;
            }
        }
    }
    
    df.size = currentRow - 1;
    
    return df;
}
