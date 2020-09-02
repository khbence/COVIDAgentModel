#include "progressionType.h"

ProgressionType::ProgressionType(
    const parser::ProgressionDirectory::ProgressionFile& file)
    : ageBegin(file.age[0]), ageEnd(file.age[1]), preCond(file.preCond) {}


bool operator<(const ProgressionType& lhs, const ProgressionType& rhs) {
    if (lhs.ageBegin < rhs.ageBegin) { return lhs.preCond < rhs.preCond; }
    return false;
}

bool operator<(const ProgressionType& lhs,
    const std::pair<unsigned, std::string>& rhs) {
    if (lhs.ageBegin < rhs.first) { return lhs.preCond < rhs.second; }
    return false;
}

bool operator<(const std::pair<unsigned, std::string>& lhs,
    const ProgressionType& rhs) {
    if (lhs.first < rhs.ageBegin) { return lhs.second < rhs.preCond; }
    return false;
}