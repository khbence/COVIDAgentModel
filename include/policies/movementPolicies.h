#pragma once
#include <iostream>
#include "timeHandler.h"
#include "datatypes.h"
#include "cxxopts.hpp"
#include "operators.h"
#include "locationTypesFormat.h"

template<typename SimulationType>
class NoMovement {
public:
    // add program parameters if we need any, this function got called already from Simulation
    static void addProgramParameters(cxxopts::Options& options) {}
    void initializeArgs(const cxxopts::ParseResult& result) {}
    void init(const parser::LocationTypes& data, unsigned cemeteryID) {}

    void planLocations() {}
    void movement(Timehandler simTime, unsigned timeStep) {}
};

template<typename SimulationType>
class DummyMovement {
protected:
    thrust::device_vector<unsigned> stepsUntilMove;

public:
    // add program parameters if we need any, this function got called already from Simulation
    static void addProgramParameters(cxxopts::Options& options) {}
    void initializeArgs(const cxxopts::ParseResult& result) {}
    void init(const parser::LocationTypes& data, unsigned cemeteryID) {}

    void planLocations() {
        auto realThis = static_cast<SimulationType*>(this);
        thrust::device_vector<unsigned>& agentLocations = realThis->agents->location;
        unsigned numberOfAgents = agentLocations.size();

        if (stepsUntilMove.size() == 0) stepsUntilMove.resize(numberOfAgents);
        thrust::fill(stepsUntilMove.begin(), stepsUntilMove.end(), 0u);
    }

    void movement(Timehandler simTime, unsigned timeStep) {
        PROFILE_FUNCTION();
        auto realThis = static_cast<SimulationType*>(this);
        thrust::device_vector<unsigned>& locationAgentList = realThis->locs->locationAgentList;
        thrust::device_vector<unsigned>& locationListOffsets = realThis->locs->locationListOffsets;
        thrust::device_vector<unsigned>& locationIdsOfAgents = realThis->locs->locationIdsOfAgents;
        thrust::device_vector<unsigned>& agentLocations = realThis->agents->location;
        unsigned numberOfAgents = agentLocations.size();
        unsigned numberOfLocations = locationListOffsets.size() - 1;

        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(agentLocations.begin(), stepsUntilMove.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(agentLocations.end(), stepsUntilMove.end())),
            [numberOfLocations] HD(thrust::tuple<unsigned&, unsigned&> tuple) {
                auto& location = thrust::get<0>(tuple);
                auto& stepsUntilMove = thrust::get<1>(tuple);
                if (stepsUntilMove == 0) {
                    location = RandomGenerator::randomUnsigned(numberOfLocations);
                    stepsUntilMove = RandomGenerator::randomUnsigned(144 / 4);// Move 4 times per day on average
                }
                stepsUntilMove--;
            });
        Util::updatePerLocationAgentLists(agentLocations, locationIdsOfAgents, locationAgentList, locationListOffsets);
    }
};

namespace RealMovementOps {
    [[nodiscard]] HD unsigned findActualLocationForType(unsigned agent,
        unsigned locType,
        unsigned long* locationOffsetPtr,
        unsigned* possibleLocationsPtr,
        unsigned* possibleTypesPtr) {
        for (unsigned i = locationOffsetPtr[agent]; i < locationOffsetPtr[agent + 1]; i++) {
            if (locType == possibleTypesPtr[i]) return possibleLocationsPtr[i];
        }
        // printf("locType %d not found for agent %d - locationOffsets: %d-%d\n", locType, agent, locationOffsetPtr[agent],
        // locationOffsetPtr[agent+1]);
        return std::numeric_limits<unsigned>::max();
    }

    template<typename PPState, typename LocationType>
    struct MovementArguments {
        MovementArguments() : simTime(0u) {}
        unsigned* stepsUntilMovePtr;
        PPState* agentStatesPtr;
        unsigned* agentTypesPtr;
        bool* diagnosedPtr;
        bool* quarantinedPtr;
        AgentStats* agentStatsPtr;
        unsigned* eventOffsetPtr;
        AgentTypeList::Event* eventsPtr;
        unsigned* agentLocationsPtr;
        unsigned long* locationOffsetPtr;
        unsigned* possibleLocationsPtr;
        unsigned* possibleTypesPtr;
        bool* locationStatesPtr;
        unsigned* locationCapacitiesPtr;
        unsigned* locationQuarantineUntilPtr;
        unsigned quarantinePolicy;
        Days day;
        unsigned hospitalType;
        unsigned homeType;
        unsigned publicPlaceType;
        unsigned doctorType;
        TimeDay simTime;
        unsigned timeStep;
        unsigned timestamp;
        unsigned tracked;
        unsigned cemeteryLoc;
        unsigned schoolType;
        unsigned workType;
        LocationType *locationTypePtr;
    };

    template<typename PPState, typename LocationType>
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    __device__
#endif
        void
        doMovement(unsigned i,
            MovementArguments<PPState, LocationType> &a) {
        if (a.stepsUntilMovePtr[i] > 0) {
            a.stepsUntilMovePtr[i]--;
            return;
        }

        unsigned& agentType = a.agentTypesPtr[i];
        states::WBStates wBState = a.agentStatesPtr[i].getWBState();
        if (wBState == states::WBStates::D) {// If dead, do not go anywhere
            a.stepsUntilMovePtr[i] = std::numeric_limits<unsigned>::max();
            a.agentLocationsPtr[i] = a.cemeteryLoc;
            return;
        }
        // TODO: during disease progression, move people who just diesd to some specific place

        if (wBState == states::WBStates::S) {// go to hospital if in serious condition
            a.stepsUntilMovePtr[i] = std::numeric_limits<unsigned>::max();
            a.agentLocationsPtr[i] =
                RealMovementOps::findActualLocationForType(i, a.hospitalType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr);
            return;
        }

        // Is agent currently in a place under quarantine
        if (a.quarantinePolicy > 1 && a.timestamp < a.locationQuarantineUntilPtr[a.agentLocationsPtr[i]]) {
            if (a.agentStatsPtr[i].quarantinedTimestamp == 0) {
                a.agentStatsPtr[i].quarantinedTimestamp = a.timestamp;
                a.quarantinedPtr[i] = true;
                if (i == a.tracked)
                    printf("Agent %d of type %d day %d at %d:%d location %d is quarantined, staying at home until %d\n",
                        i,
                        agentType + 1,
                        (int)a.day,
                        a.simTime.getMinutes() / 60,
                        a.simTime.getMinutes() % 60,
                        a.agentLocationsPtr[i],
                        a.locationQuarantineUntilPtr[a.agentLocationsPtr[i]]);
            }
            a.stepsUntilMovePtr[i] = a.locationQuarantineUntilPtr[a.agentLocationsPtr[i]] - a.timestamp;

            // If not home, send home
            unsigned homeLocation =
                RealMovementOps::findActualLocationForType(i, a.homeType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr);
            if (homeLocation != a.agentLocationsPtr[i]) {
                unsigned until = a.locationQuarantineUntilPtr[a.agentLocationsPtr[i]];
                a.agentLocationsPtr[i] = homeLocation;
                // TODO: quarantine whole home??
                // a.locationQuarantineUntilPtr[a.agentLocationsPtr[i]] = until;
            }
            return;
        }

        // Should agent still be quarantined
        if ((a.quarantinePolicy > 0 && (a.diagnosedPtr[i] || (a.timestamp - a.agentStatsPtr[i].diagnosedTimestamp) < 2 * 7 * 24 * 60 / a.timeStep)) //stay home if diagnosed or quarantine has not expired
            || (a.quarantinePolicy > 1 && a.quarantinedPtr[i]
                && (a.timestamp - a.agentStatsPtr[i].quarantinedTimestamp) < 2 * 7 * 24 * 60 / a.timeStep)) {// TODO: specify quarantine length
            if (a.agentStatesPtr[i].getWBState() == states::WBStates::S)// send to hospital
                a.agentLocationsPtr[i] =
                    RealMovementOps::findActualLocationForType(i, a.hospitalType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr);
            else {
                // Quarantine for 2 weeks at home
                a.agentLocationsPtr[i] =
                    RealMovementOps::findActualLocationForType(i, a.homeType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr);
            }
            // if less than 2 weeks since diagnosis/quarantine, stay where agent already is
            a.stepsUntilMovePtr[i] = std::numeric_limits<unsigned>::max();
            return;
        }

        unsigned agentTypeOffset = AgentTypeList::getOffsetIndex(agentType, wBState, a.day);
        unsigned eventsBegin = a.eventOffsetPtr[agentTypeOffset];
        unsigned eventsEnd = a.eventOffsetPtr[agentTypeOffset + 1];

        int activeEventsBegin = -1;
        int activeEventsEnd = -1;

        // Here we assume if multiple events are given for the same timeslot, they all start & end at the same time
        for (unsigned j = eventsBegin; j < eventsEnd; j++) {
            if (a.simTime >= a.eventsPtr[j].start && a.simTime < a.eventsPtr[j].end && activeEventsBegin == -1) activeEventsBegin = j;
            if (a.simTime < a.eventsPtr[j].start) {
                activeEventsEnd = j;
                break;
            }
        }
        if (i == a.tracked)
            printf("Agent %d of type %d day %d at %d:%d WBState %d activeEvents: %d-%d\n",
                i,
                agentType + 1,
                (int)a.day,
                a.simTime.getMinutes() / 60,
                a.simTime.getMinutes() % 60,
                (int)wBState,
                activeEventsBegin,
                activeEventsEnd);

        // Possibilities:
        // 1 both are -1 -> no more events for that day. Should be home if wBState != S, or at hospital if S
        // 2 Begin != -1, End == -1 -> last event for the day. Move there (if needed pick randomly)
        // 3 Begin == -1, End != -1 -> no events right now, but there will be some later
        //      3a if less than 30 mins until next possible event, then stay here
        //      3b if 30-60 to next possible event, should go to public place (type 0)
        //      3c more than 60 mins, then go home
        // 4 neither -1, then pick randomly between one of the events

        // ISSUES:
        // do we forcibly finish at midnight?? What if the duration goes beyond that?
        unsigned newLocationType = std::numeric_limits<unsigned>::max();

        // Case 1
        if (activeEventsBegin == -1 && activeEventsEnd == -1) {
            newLocationType = wBState == states::WBStates::S ? a.hospitalType : a.homeType;// Hostpital if sick, home otherwise
            unsigned myHome =
                RealMovementOps::findActualLocationForType(i, newLocationType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr);
            a.agentLocationsPtr[i] = myHome;
            a.stepsUntilMovePtr[i] = a.simTime.getStepsUntilMidnight(a.timeStep);
            if (i == a.tracked)
                printf(
                    "\tCase 1- moving to locType %d location %d until midnight (for %d steps)\n", newLocationType, myHome, a.stepsUntilMovePtr[i] - 1);
        }
        // Case 2 and 4
        if (activeEventsBegin != -1) {
            unsigned numPotentialEvents = eventsEnd - activeEventsBegin;
            TimeDayDuration basicDuration(0.0);
            if (numPotentialEvents == 1) {
                newLocationType = a.eventsPtr[activeEventsBegin].locationType;
                basicDuration = a.eventsPtr[activeEventsBegin].duration;
            } else {
                double rand = RandomGenerator::randomReal(1.0);
                double threshhold = a.eventsPtr[activeEventsBegin].chance;
                unsigned i = 0;
                while (rand > threshhold && i < numPotentialEvents) {
                    i++;
                    threshhold += a.eventsPtr[activeEventsBegin + i].chance;
                }
                newLocationType = a.eventsPtr[activeEventsBegin + i].locationType;
                basicDuration = a.eventsPtr[activeEventsBegin + i].duration;
            }
            unsigned newLocation =
                RealMovementOps::findActualLocationForType(i, newLocationType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr);
            // Check if location is open/closed. If closed, go home instead
            unsigned wasClosed = std::numeric_limits<unsigned>::max();
            if (a.locationStatesPtr[newLocation] == false) {
                wasClosed = newLocation;
                newLocation = RealMovementOps::findActualLocationForType(i, a.homeType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr);
            }
            a.agentLocationsPtr[i] = newLocation;
            if (activeEventsEnd == -1) {
                if ((a.simTime + basicDuration).isOverMidnight()) {
                    a.stepsUntilMovePtr[i] = a.simTime.getStepsUntilMidnight(a.timeStep);
                } else {
                    // does not last till midnight, but no events afterwards - spend full duration there
                    a.stepsUntilMovePtr[i] = basicDuration.steps(a.timeStep);
                }
            } else {
                // If duration is less then the beginning of the next move window, then spend full duration here
                if (a.simTime + basicDuration < a.eventsPtr[activeEventsEnd].start)
                    a.stepsUntilMovePtr[i] = basicDuration.steps(a.timeStep);
                else {
                    // Otherwise I need to move again randomly between the end of this duration and the end of next movement window
                    TimeDayDuration window = a.eventsPtr[activeEventsEnd].end - (a.simTime + basicDuration);
                    unsigned randExtra = RandomGenerator::randomUnsigned(window.steps(a.timeStep));
                    a.stepsUntilMovePtr[i] = basicDuration.steps(a.timeStep) + randExtra;
                }
            }
            if (i == a.tracked) {
                if (wasClosed == std::numeric_limits<unsigned>::max())
                    printf("\tCase 2&4- moving to locType %d location %d for %d steps\n", newLocationType, newLocation, a.stepsUntilMovePtr[i] - 1);
                else
                    printf("\tCase 2&4- tried moving to locType %d location %d, but was closed, moving home to %d for %d steps\n",
                        newLocationType,
                        wasClosed,
                        newLocation,
                        a.stepsUntilMovePtr[i] - 1);
            }
        }

        // Case 3
        if (activeEventsBegin == -1 && activeEventsEnd != -1) {
            // Randomly decide when the move will occur in the next window:
            TimeDayDuration length = a.eventsPtr[activeEventsEnd].end - a.eventsPtr[activeEventsEnd].start;
            unsigned length_steps = length.steps(a.timeStep);
            unsigned randDelay = RandomGenerator::randomUnsigned(length_steps);
            a.stepsUntilMovePtr[i] = (a.eventsPtr[activeEventsEnd].start - a.simTime).steps(a.timeStep) + randDelay;
            unsigned timeLeft = a.stepsUntilMovePtr[i];
            // Case 3.a -- less than 30 mins -> stay here
            if (timeLeft < TimeDayDuration(0.3).steps(a.timeStep)) {
                if (i == a.tracked) printf("\tCase 3a- staying in place for %d steps\n", a.stepsUntilMovePtr[i] - 1);
                // Do nothing - location stays the same
            } else if (timeLeft < TimeDayDuration(1.0).steps(a.timeStep)) {
                newLocationType = a.publicPlaceType;
                unsigned myPublicPlace =
                    RealMovementOps::findActualLocationForType(i, a.publicPlaceType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr);
                a.agentLocationsPtr[i] = myPublicPlace;
                if (i == a.tracked)
                    printf("\tCase 3b- moving to public Place type 1 location %d for %d steps\n", myPublicPlace, a.stepsUntilMovePtr[i] - 1);
            } else {
                newLocationType = a.homeType;
                unsigned myHome = RealMovementOps::findActualLocationForType(i, a.homeType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr);
                a.agentLocationsPtr[i] = myHome;
                if (i == a.tracked) printf("\tCase 3c- moving to home type 2 location %d for %d steps\n", myHome, a.stepsUntilMovePtr[i] - 1);
            }
        }

        // Diagnosis-related operations
        if (newLocationType == a.hospitalType || newLocationType == a.doctorType) {
            // If previously undiagnosed and
            if (!a.diagnosedPtr[i] && a.agentStatesPtr[i].isInfectious()) {
                a.diagnosedPtr[i] = true;
                a.quarantinedPtr[i] = true;
                a.agentStatsPtr[i].diagnosedTimestamp = a.timestamp;
                a.agentStatsPtr[i].quarantinedTimestamp = a.timestamp;

                if (a.agentStatesPtr[i].getWBState() == states::WBStates::S)// send to hospital
                    a.agentLocationsPtr[i] =
                        RealMovementOps::findActualLocationForType(i, a.hospitalType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr);
                else {
                    // Quarantine for 2 weeks at home
                    a.agentLocationsPtr[i] =
                        RealMovementOps::findActualLocationForType(i, a.homeType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr);
                    if (a.quarantinePolicy > 0) {
                        a.stepsUntilMovePtr[i] = 2*7*24*60/a.timeStep; //this will be set to 0 at midnight, so need to check
                        if (i == a.tracked)
                            printf("\tDiagnosed, going into quarantine in home type 2 location %d for %d steps\n",
                                a.agentLocationsPtr[i],
                                a.stepsUntilMovePtr[i] - 1);
                    } else {
                        if (i == a.tracked)
                            printf("\tDiagnosed, going into home type 2 location %d for %d steps\n", 
                                a.agentLocationsPtr[i],
                                a.stepsUntilMovePtr[i]-1);
                    }
                    // Place home under quarantine
                    if (a.quarantinePolicy > 1 && a.locationQuarantineUntilPtr[a.agentLocationsPtr[i]] < a.timestamp) {
                        a.locationQuarantineUntilPtr[a.agentLocationsPtr[i]] = a.timestamp + 2 * 7 * 24 * 60 / a.timeStep;// TODO: quarantine period
                    }
                    if (a.quarantinePolicy > 2) {
                        unsigned school = RealMovementOps::findActualLocationForType(i, a.schoolType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr);
                        unsigned work = RealMovementOps::findActualLocationForType(i, a.workType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr);
                        unsigned toClose[2] = {school, work};
                        for (unsigned loc : toClose) {
                            if (loc != std::numeric_limits<unsigned>::max() &&
                                a.locationTypePtr[loc] != a.doctorType && a.locationTypePtr[loc] != a.hospitalType &&
                                a.locationQuarantineUntilPtr[loc] < a.timestamp) {
                                if (i == a.tracked)
                                    printf("\tFlagging work/school as quarantined: %d\n",loc);
                                a.locationQuarantineUntilPtr[loc] = a.timestamp + 2 * 7 * 24 * 60 / a.timeStep;// TODO: quarantine period
                            }
                        }
                    }
                }
            }
        }

        a.stepsUntilMovePtr[i]--;
    }

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    template<typename PPState, typename LocationType>
    __global__ void doMovementDriver(unsigned numberOfAgents,
        MovementArguments<PPState, LocationType> a) {
        unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < numberOfAgents) {
            RealMovementOps::doMovement(i, a);
        }
    }
#endif
}// namespace RealMovementOps

template<typename SimulationType>
class RealMovement {
    thrust::device_vector<unsigned> stepsUntilMove;
    unsigned publicSpace;
    unsigned home;
    unsigned hospital;
    unsigned cemeteryLoc;
    unsigned doctor;
    unsigned tracked;
    unsigned quarantinePolicy;
    unsigned school;
    unsigned work;

public:
    // add program parameters if we need any, this function got called already from Simulation
    static void addProgramParameters(cxxopts::Options& options) {
        options.add_options()(
            "trace", "Trace movements of agent", cxxopts::value<unsigned>()->default_value(std::to_string(std::numeric_limits<unsigned>::max())))
            ("quarantinePolicy", "Quarantine policy: 0 - None, 1 - Agent only, 2 - Agent and household, 3 - Agent, household, school/work", cxxopts::value<unsigned>()->default_value(std::to_string(unsigned(0))));
    }
    void initializeArgs(const cxxopts::ParseResult& result) { 
        tracked = result["trace"].as<unsigned>();
        quarantinePolicy = result["quarantinePolicy"].as<unsigned>();
    }
    void init(const parser::LocationTypes& data, unsigned cemeteryID) {
        publicSpace = data.publicSpace;
        home = data.home;
        hospital = data.hospital;
        cemeteryLoc = cemeteryID;
        doctor = data.doctor;
        school = data.school;
        work = data.work;
    }

    void planLocations() {
        PROFILE_FUNCTION();
        auto realThis = static_cast<SimulationType*>(this);
        thrust::device_vector<unsigned>& agentLocations = realThis->agents->location;
        unsigned numberOfAgents = agentLocations.size();

        if (stepsUntilMove.size() == 0) stepsUntilMove.resize(numberOfAgents);
        thrust::fill(stepsUntilMove.begin(), stepsUntilMove.end(), 0u);
    }

    void movement(Timehandler simTime, unsigned timeStep) {
        PROFILE_FUNCTION();
        auto realThis = static_cast<SimulationType*>(this);

        RealMovementOps::MovementArguments<typename SimulationType::PPState_t, typename SimulationType::TypeOfLocation_t> a;

        a.quarantinePolicy = quarantinePolicy;
        a.tracked = this->tracked;
        a.hospitalType = hospital;
        a.homeType = home;
        a.publicPlaceType = publicSpace;
        a.doctorType = doctor;
        a.timeStep = timeStep;
        a.simTime = TimeDay(simTime);
        a.cemeteryLoc = cemeteryLoc;
        a.schoolType = school;
        a.workType = work;

        // Location-based data
        thrust::device_vector<unsigned>& locationAgentList = realThis->locs->locationAgentList;
        unsigned *locationAgentListPtr = thrust::raw_pointer_cast(locationAgentList.data());
        thrust::device_vector<unsigned>& locationListOffsets = realThis->locs->locationListOffsets;
        unsigned *locationListOffsetsPtr = thrust::raw_pointer_cast(locationListOffsets.data());
        thrust::device_vector<unsigned>& locationIdsOfAgents = realThis->locs->locationIdsOfAgents;
        unsigned *locationIdsOfAgentsPtr = thrust::raw_pointer_cast(locationIdsOfAgents.data());
        thrust::device_vector<bool>& locationStates = realThis->locs->states;
        a.locationStatesPtr = thrust::raw_pointer_cast(locationStates.data());
        thrust::device_vector<unsigned>& locationCapacities = realThis->locs->capacity;
        a.locationCapacitiesPtr = thrust::raw_pointer_cast(locationCapacities.data());
        thrust::device_vector<unsigned>& locationQuarantineUntil = realThis->locs->quarantineUntil;
        a.locationQuarantineUntilPtr = thrust::raw_pointer_cast(locationQuarantineUntil.data());
        thrust::device_vector<typename SimulationType::TypeOfLocation_t>& locationTypes = realThis->locs->locType;
        a.locationTypePtr = thrust::raw_pointer_cast(locationTypes.data());

        // Agent-based data
        thrust::device_vector<unsigned>& agentLocations = realThis->agents->location;
        a.agentLocationsPtr = thrust::raw_pointer_cast(agentLocations.data());
        thrust::device_vector<unsigned>& agentTypes = realThis->agents->types;
        a.agentTypesPtr = thrust::raw_pointer_cast(agentTypes.data());
        thrust::device_vector<typename SimulationType::PPState_t>& agentStates = realThis->agents->PPValues;
        a.agentStatesPtr = thrust::raw_pointer_cast(agentStates.data());
        thrust::device_vector<bool>& diagnosed = realThis->agents->diagnosed;
        a.diagnosedPtr = thrust::raw_pointer_cast(diagnosed.data());
        thrust::device_vector<bool>& quarantined = realThis->agents->quarantined;
        a.quarantinedPtr = thrust::raw_pointer_cast(quarantined.data());
        thrust::device_vector<AgentStats>& agentStats = realThis->agents->agentStats;
        a.agentStatsPtr = thrust::raw_pointer_cast(agentStats.data());
        a.stepsUntilMovePtr = thrust::raw_pointer_cast(this->stepsUntilMove.data());

        // Arrays storing actual location IDs for each agent, for each location type
        thrust::device_vector<unsigned long>& locationOffset = realThis->agents->locationOffset;
        a.locationOffsetPtr = thrust::raw_pointer_cast(locationOffset.data());
        thrust::device_vector<unsigned>& possibleLocations = realThis->agents->possibleLocations;
        a.possibleLocationsPtr = thrust::raw_pointer_cast(possibleLocations.data());
        thrust::device_vector<unsigned>& possibleTypes = realThis->agents->possibleTypes;
        a.possibleTypesPtr = thrust::raw_pointer_cast(possibleTypes.data());

        // Arrays storing movement behaviour with general locationTypes - for each agent type, WB state, and day
        thrust::device_vector<unsigned>& eventOffset = realThis->agents->agentTypes.eventOffset;
        a.eventOffsetPtr = thrust::raw_pointer_cast(eventOffset.data());
        thrust::device_vector<AgentTypeList::Event>& events = realThis->agents->agentTypes.events;
        a.eventsPtr = thrust::raw_pointer_cast(events.data());

        unsigned numberOfAgents = agentLocations.size();
        unsigned numberOfLocations = locationListOffsets.size() - 1;

        a.day = simTime.getDay();
        a.timestamp = simTime.getTimestamp();

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP
#pragma omp parallel for
        for (unsigned i = 0; i < numberOfAgents; i++) {
            RealMovementOps::doMovement(i, a);
        }
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
        RealMovementOps::doMovementDriver<<<(numberOfAgents - 1) / 256 + 1, 256>>>(numberOfAgents, a);
        cudaDeviceSynchronize();
#endif
        Util::updatePerLocationAgentLists(agentLocations, locationIdsOfAgents, locationAgentList, locationListOffsets);
    }
};
