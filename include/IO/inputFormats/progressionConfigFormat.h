#pragma once
#include "JSONDecoder.h"
#include <vector>
#include "string"

namespace parser {
    struct ProgressionDirectory
        : public jsond::JSONDecodable<ProgressionDirectory> {
        struct ProgressionFile : public jsond::JSONDecodable<ProgressionFile> {
            BEGIN_MEMBER_DECLARATIONS();
            DECODABLE_MEMBER(std::string, fileName);
            DECODABLE_MEMBER(std::vector<int>, age);
            DECODABLE_MEMBER(std::string, preCond);
            END_MEMBER_DECLARATIONS();
        };
        BEGIN_MEMBER_DECLARATIONS();
        DECODABLE_MEMBER(std::vector<ProgressionFile>, transitionMatrices);
        END_MEMBER_DECLARATIONS();
    };

}// namespace parser