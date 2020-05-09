#pragma once
#include "JSONDecoder.h"
#include <vector>
#include <string>

namespace parser {
    struct Locations : public jsond::JSONDecoder<Locations> {
        struct Place : public jsond::JSONDecoder<Place> {
            BEGIN_MEMBER_DECLARATION();
            DECODABLE_MEMBER(int, ID);
            DECODABLE_MEMBER(int, type);
            DECODABLE_MEMBER(std::vector<double>, coordinates);
            DECODABLE_MEMBER(int, area);
            DECODABLE_MEMBER(std::string, state);
            DECODABLE_MEMBER(int, capacity);
            DECODABLE_MEMBER(std::vector<int>, ageInter);
            END_MEMBER_DECLARATIONS();
        };

        BEGIN_MEMBER_DECLARATION();
        DECODABLE_MEMBER(std::vector<Place>, places);
        END_MEMBER_DECLARATIONS();
    };
}// namespace parser