#pragma once
#include "JSONDecoder.h"
#include <vector>
#include <string>

namespace parser {
    struct AgentTypes : public jsond::JSONDecodable<AgentTypes> {
        struct Schedule : public jsond::JSONDecodable<Schedule> {
            struct Event : public jsond::JSONDecodable<Event> {
                BEGIN_MEMBER_DECLARATIONS();
                DECODABLE_MEMBER(int, type);
                DECODABLE_MEMBER(double, chance);
                DECODABLE_MEMBER(double, start);
                DECODABLE_MEMBER(double, end);
                END_MEMBER_DECLARATIONS();
            };

            BEGIN_MEMBER_DECLARATIONS();
            DECODABLE_MEMBER(unsigned, ID);
            DECODABLE_MEMBER(std::vector<Event>, schedule);
            END_MEMBER_DECLARATIONS();
        };

        struct Type : public jsond::JSONDecodable<Type> {
            struct ScheduleUnique : public jsond::JSONDecodable<ScheduleUnique> {
                struct Event : public jsond::JSONDecodable<Event> {
                    BEGIN_MEMBER_DECLARATIONS();
                    DECODABLE_MEMBER(int, ID);
                    DECODABLE_MEMBER(int, type);
                    DECODABLE_MEMBER(double, chance);
                    DECODABLE_MEMBER(double, start);
                    DECODABLE_MEMBER(double, end);
                    END_MEMBER_DECLARATIONS();
                };

                BEGIN_MEMBER_DECLARATIONS();
                DECODABLE_MEMBER(std::string, WB);
                DECODABLE_MEMBER(std::string, dayType);
                DECODABLE_MEMBER(std::vector<Event>, schedule);
                END_MEMBER_DECLARATIONS();
            };

            struct ScheduleTypic : public jsond::JSONDecodable<ScheduleTypic> {
                BEGIN_MEMBER_DECLARATIONS();
                DECODABLE_MEMBER(int, ID);
                DECODABLE_MEMBER(std::string, WB);
                DECODABLE_MEMBER(std::string, dayType);
                DECODABLE_MEMBER(int, scheduleID);
                END_MEMBER_DECLARATIONS();
            };

            BEGIN_MEMBER_DECLARATIONS();
            DECODABLE_MEMBER(std::string, name);
            DECODABLE_MEMBER(int, ID);
            DECODABLE_MEMBER(std::vector<ScheduleUnique>, schedulesUnique);
            DECODABLE_MEMBER(std::vector<ScheduleTypic>, schedulesTypic);
            END_MEMBER_DECLARATIONS();
        };

        BEGIN_MEMBER_DECLARATIONS();
        DECODABLE_MEMBER(std::vector<Schedule>, commonSchedules);
        DECODABLE_MEMBER(std::vector<Type>, types);
        END_MEMBER_DECLARATIONS();
    };
}// namespace parser