#pragma once
#include "JSONDecoder.h"
#include <vector>
#include <string>

namespace parser {
    struct ConfigRandom : public jsond::JSONDecodable<ConfigRandom> {
        struct LocationTypeChance : public jsond::JSONDecodable<LocationTypeChance> {
            BEGIN_MEMBER_DECLARATIONS();
            DECODABLE_MEMBER(unsigned, value);
            DECODABLE_MEMBER(double, chance);
            END_MEMBER_DECLARATIONS();
        };

        struct PreCondChance : public jsond::JSONDecodable<PreCondChance> {
            BEGIN_MEMBER_DECLARATIONS();
            DECODABLE_MEMBER(unsigned, value);
            DECODABLE_MEMBER(double, chance);
            END_MEMBER_DECLARATIONS();
        };
        struct StateChance : public jsond::JSONDecodable<StateChance> {
            BEGIN_MEMBER_DECLARATIONS();
            DECODABLE_MEMBER(std::string, value);
            DECODABLE_MEMBER(double, chance);
            END_MEMBER_DECLARATIONS();
        };
        struct AgentTypeChance : public jsond::JSONDecodable<AgentTypeChance> {
            BEGIN_MEMBER_DECLARATIONS();
            DECODABLE_MEMBER(unsigned, value);
            DECODABLE_MEMBER(double, chance);
            END_MEMBER_DECLARATIONS();
        };

        BEGIN_MEMBER_DECLARATIONS();
        DECODABLE_MEMBER(double, irregulalLocationChance);
        DECODABLE_MEMBER(std::vector<LocationTypeChance>, locationTypeDistibution);
        DECODABLE_MEMBER(std::vector<PreCondChance>, preCondDistibution);
        DECODABLE_MEMBER(std::vector<StateChance>, stateDistibution);
        DECODABLE_MEMBER(std::vector<AgentTypeChance>, agentTypeDistribution);
        END_MEMBER_DECLARATIONS();
    };
}// namespace parser