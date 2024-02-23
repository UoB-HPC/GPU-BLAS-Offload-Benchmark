#pragma once

#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

class tablePrinter {
 public:
  tablePrinter(const std::string title, const std::vector<std::string>& headers,
               const std::vector<std::vector<std::string>>& data)
      : title_(title), columnHeaders_(headers), rowData_(data) {
    // Check that all rows of data contain correct number of elements
    for (size_t i = 0; i < rowData_.size(); i++) {
      if (rowData_[i].size() != columnHeaders_.size()) {
        std::cerr << "ERROR - Number of elements in table row " << i
                  << " does not match the number of column headers."
                  << std::endl;
        exit(1);
      }
    }
    // Initialise member variables
    calcColumnWidths();
    formatHeader();
    formatRows();
    initCommonStrings();
  }

  /** Print the table to stdout - `padding` refers to the number of tabs each
   * line will be prefixed with. */
  void print(int padding) {
    std::string prefix(padding, '\t');
    std::cout << prefix << title_ << std::endl;
    prefix = std::string(padding + 1, '\t');

    std::cout << prefix << hLine_ << std::endl;
    std::cout << prefix << headerLine_ << std::endl;
    std::cout << prefix << hLineD_ << std::endl;
    for (size_t i = 0; i < rows_.size(); i++) {
      std::cout << prefix << rows_[i] << std::endl;
      std::cout << prefix << hLine_ << std::endl;
    }
    std::cout << std::endl;
  }

 private:
  /** Calculates width of each column. */
  void calcColumnWidths() {
    for (size_t i = 0; i < columnHeaders_.size(); i++) {
      // Find max width from columnHeader and each Row
      size_t maxWidth = columnHeaders_[i].size();
      for (size_t j = 0; j < rowData_.size(); j++) {
        if (rowData_[j][i].size() > maxWidth) maxWidth = rowData_[j][i].size();
      }
      // Once found max width, add 1 padding to either side
      colWidths_.push_back(static_cast<int>(maxWidth + 2));
    }
  }

  /** Format the table's header. */
  void formatHeader() {
    // Initialise stream
    std::stringstream headerStream;
    headerStream << "|";
    for (size_t i = 0; i < columnHeaders_.size(); i++) {
      int suffixPadding = colWidths_[i] - columnHeaders_[i].size() - 1;
      headerStream << " " << columnHeaders_[i]
                   << std::string(suffixPadding, ' ') << "|";
    }
    headerLine_ = headerStream.str();
  }

  /** Format the table's rows. */
  void formatRows() {
    for (size_t i = 0; i < rowData_.size(); i++) {
      std::stringstream rowStream;
      rowStream << "|";
      for (size_t j = 0; j < rowData_[i].size(); j++) {
        int suffixPadding = colWidths_[j] - rowData_[i][j].size() - 1;
        rowStream << " " << rowData_[i][j] << std::string(suffixPadding, ' ')
                  << "|";
      }
      rows_.push_back(rowStream.str());
    }
  }

  /** Initialises common strings. */
  void initCommonStrings() {
    int totalWidth = 1 + 1 +
                     std::accumulate(colWidths_.begin(), colWidths_.end(), 0) +
                     (columnHeaders_.size() - 1);

    hLine_ = std::string(totalWidth, '-');
    hLineD_ = std::string(totalWidth, '=');
  }

  /** The table's title. */
  const std::string title_;

  /** The header names for each column. */
  const std::vector<std::string> columnHeaders_;

  /** The rows of data - items of data per row must match number of columns
   * defined in `columnHeaders_`. */
  const std::vector<std::vector<std::string>> rowData_;

  /** The correctly formatted table header. */
  std::string headerLine_;

  /** The correctly formatted table rows. */
  std::vector<std::string> rows_;

  /** The width of each column - each index corresponds to an index in
   * `columnHeaders_`. */
  std::vector<int> colWidths_;

  /** A horizontal line of single width. */
  std::string hLine_;

  /** A horizontal line of double width. */
  std::string hLineD_;
};