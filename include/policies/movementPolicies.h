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
    void init(const parser::LocationTypes& data) {}

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
    void init(const parser::LocationTypes& data) {}

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
[[nodiscard]] HD unsigned findActualLocationForType(unsigned agent, unsigned locType, unsigned long *locationOffsetPtr, unsigned *possibleLocationsPtr, unsigned *possibleTypesPtr) {
    for (unsigned i = locationOffsetPtr[agent]; i < locationOffsetPtr[agent+1]; i++) {
        if (locType == possibleTypesPtr[i])
            return possibleLocationsPtr[i];
    }
    printf("locType %d not found for angent %d\n", locType, agent );
    return 0;
}
template <typename PPState>
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
__device__
#endif
void doMovement(unsigned i, unsigned *stepsUntilMovePtr, PPState *agentStatesPtr,
                        unsigned *agentTypesPtr,  unsigned *eventOffsetPtr, AgentTypeList::Event *eventsPtr, unsigned *agentLocationsPtr,
                        unsigned long*locationOffsetPtr, unsigned *possibleLocationsPtr, unsigned *possibleTypesPtr, 
                        Days day, unsigned hospitalType, unsigned homeType, unsigned publicPlaceType, 
                        TimeDay simTime, unsigned timeStep, unsigned tracked) {
    if (stepsUntilMovePtr[i]>0) {
        stepsUntilMovePtr[i]--;
        return;
    }
    states::WBStates wBState = agentStatesPtr[i].getWBState();
    if (wBState == states::WBStates::D) { //If dead, do not go anywhere
        stepsUntilMovePtr[i] = UINT32_MAX;
        return;
    }
    //TODO: during disease progression, move people who just diesd to some specific place

    unsigned &agentType = agentTypesPtr[i];

    unsigned agentTypeOffset = AgentTypeList::getOffsetIndex(agentType,wBState, day);
    unsigned eventsBegin = eventOffsetPtr[agentTypeOffset];
    unsigned eventsEnd = eventOffsetPtr[agentTypeOffset+1];

    int activeEventsBegin=-1;
    int activeEventsEnd=-1;

    //Here we assume if multiple events are given for the same timeslot, they all start & end at the same time
    for (unsigned j = eventsBegin; j < eventsEnd; j++) {
        if (simTime >= eventsPtr[j].start && simTime < eventsPtr[j].end && activeEventsBegin==-1)
            activeEventsBegin = j;
        if (simTime < eventsPtr[j].start) {
            activeEventsEnd = j;
            break;
        }
    }
    if (i == tracked)
        printf("Agent %d of type %d day %d at %d:%d WBState %d activeEvents: %d-%d\n", i, agentType+1, (int)day, simTime.getMinutes()/60, simTime.getMinutes()%60, (int)wBState, activeEventsBegin, activeEventsEnd);

    //Possibilities:
    // 1 both are -1 -> no more events for that day. Should be home if wBState != S, or at hospital if S
    // 2 Begin != -1, End == -1 -> last event for the day. Move there (if needed pick randomly)
    // 3 Begin == -1, End != -1 -> no events right now, but there will be some later
    //      3a if less than 30 mins until next possible event, then stay here
    //      3b if 30-60 to next possible event, should go to public place (type 0)
    //      3c more than 60 mins, then go home
    // 4 neither -1, then pick randomly between one of the events

    //ISSUES:
    //do we forcibly finish at midnight?? What if the duration goes beyond that?

    //Case 1
    if (activeEventsBegin==-1 && activeEventsEnd == -1) {
        unsigned typeToGoTo = wBState == states::WBStates::S ? hospitalType : homeType; //Hostpital if sick, home otherwise
        unsigned myHome = RealMovementOps::findActualLocationForType(i, typeToGoTo, locationOffsetPtr, possibleLocationsPtr, possibleTypesPtr);
        agentLocationsPtr[i] = myHome;
        stepsUntilMovePtr[i] = simTime.getStepsUntilMidnight(timeStep);
        if (i == tracked)
            printf("\tCase 1- moving to locType %d location %d until midnight (for %d steps)\n", typeToGoTo, myHome, stepsUntilMovePtr[i]-1);
    }
    //Case 2 and 4
    if (activeEventsBegin!=-1) {
        unsigned numPotentialEvents = eventsEnd-activeEventsBegin;
        unsigned newLocationType = UINT32_MAX;
        TimeDayDuration basicDuration(0.0);
        if (numPotentialEvents == 1) {
            newLocationType = eventsPtr[activeEventsBegin].locationType;
            basicDuration = eventsPtr[activeEventsBegin].duration;
        } else {
            double rand = RandomGenerator::randomReal(1.0);
            double threshhold = eventsPtr[activeEventsBegin].chance;
            unsigned i = 0;
            while (rand > threshhold && i < numPotentialEvents) {
                i++;
                threshhold += eventsPtr[activeEventsBegin+i].chance;
            }
            newLocationType = eventsPtr[activeEventsBegin+i].locationType;
            basicDuration = eventsPtr[activeEventsBegin+i].duration;
        }
        unsigned newLocation = RealMovementOps::findActualLocationForType(i, newLocationType, locationOffsetPtr, possibleLocationsPtr, possibleTypesPtr);
        agentLocationsPtr[i] = newLocation;
        if (activeEventsEnd==-1) {
            //TODO: Do we truncate to midnight?
            if ((simTime + basicDuration).isOverMidnight()) { //TODO: is this right?
                stepsUntilMovePtr[i] = simTime.getStepsUntilMidnight(timeStep);
            } else {
                //does not last till midnight, but no events afterwards - spend full duration there
                stepsUntilMovePtr[i] = basicDuration.steps(timeStep);
            }
        } else {
            //If duration is less then the beginning of the next move window, then spend full duration here
            if (simTime + basicDuration < eventsPtr[activeEventsEnd].start)
                stepsUntilMovePtr[i] = basicDuration.steps(timeStep);
            else {
                //Otherwise I need to move again randomly between the end of this duration and the end of next movement window
                TimeDayDuration window = eventsPtr[activeEventsEnd].end - (simTime + basicDuration);
                unsigned randExtra = RandomGenerator::randomUnsigned(window.steps(timeStep));
                stepsUntilMovePtr[i] = basicDuration.steps(timeStep) + randExtra;
            }
        }
        if (i == tracked)
            printf("\tCase 2&4- moving to locType %d location %d for %d steps\n", newLocationType, newLocation, stepsUntilMovePtr[i]-1);

        
    }

    //Case 3
    if (activeEventsBegin==-1 && activeEventsEnd!=-1) {
        //Randomly decide when the move will occur in the next window:
        TimeDayDuration length = eventsPtr[activeEventsEnd].end-eventsPtr[activeEventsEnd].start;
        unsigned length_steps = length.steps(timeStep);
        unsigned randDelay = RandomGenerator::randomUnsigned(length_steps);
        stepsUntilMovePtr[i] = (eventsPtr[activeEventsEnd].start-simTime).steps(timeStep) + randDelay;
        unsigned timeLeft = stepsUntilMovePtr[i];
        //Case 3.a -- less than 30 mins -> stay here
        if (timeLeft < TimeDayDuration(0.3).steps(timeStep)) {
            if (i == tracked)
                printf("\tCase 3a- staying in place for %d steps\n", stepsUntilMovePtr[i]-1);
            //Do nothing - location stays the same
        } else if (timeLeft < TimeDayDuration(1.0).steps(timeStep)) {
            unsigned myPublicPlace = RealMovementOps::findActualLocationForType(i, publicPlaceType, locationOffsetPtr, possibleLocationsPtr, possibleTypesPtr);
            agentLocationsPtr[i] = myPublicPlace;
            if (i == tracked)
                printf("\tCase 3b- moving to public Place type 1 location %d for %d steps\n", myPublicPlace, stepsUntilMovePtr[i]-1);
        } else {
            unsigned myHome = RealMovementOps::findActualLocationForType(i, homeType, locationOffsetPtr, possibleLocationsPtr, possibleTypesPtr);
            agentLocationsPtr[i] = myHome;
            if (i == tracked)
                printf("\tCase 3c- moving to home type 2 location %d for %d steps\n", myHome, stepsUntilMovePtr[i]-1);
        }
    }
    stepsUntilMovePtr[i]--;
    //Movement should start at random within the movement period (i.e. between start and end)
    //->but here we need to determine the exact time of the next move. So for cases 3 and 4 we add a random on top of duration
}

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
template <typename PPState>
__global__ void doMovementDriver(unsigned numberOfAgents, unsigned *stepsUntilMovePtr, PPState *agentStatesPtr,
                        unsigned *agentTypesPtr,  unsigned *eventOffsetPtr, AgentTypeList::Event *eventsPtr, unsigned *agentLocationsPtr,
                        unsigned long*locationOffsetPtr, unsigned *possibleLocationsPtr, unsigned *possibleTypesPtr, 
                        Days day, unsigned hospitalType, unsigned homeType, unsigned publicPlaceType, 
                        TimeDay simTime, unsigned timeStep, unsigned tracked) {
    unsigned i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i < numberOfAgents) {
        RealMovementOps::doMovement(i, stepsUntilMovePtr, agentStatesPtr,
                    agentTypesPtr,  eventOffsetPtr, eventsPtr, agentLocationsPtr,
                    locationOffsetPtr, possibleLocationsPtr, possibleTypesPtr, 
                    day, hospitalType, homeType, publicPlaceType, simTime, timeStep, tracked);
    }
}
#endif
}

template<typename SimulationType>
class RealMovement {   
    thrust::device_vector<unsigned> stepsUntilMove;
    unsigned publicSpace;
    unsigned home;
    unsigned hospital;
    unsigned tracked;

    public:
    // add program parameters if we need any, this function got called already from Simulation
    static void addProgramParameters(cxxopts::Options& options) {
        options.add_options()("trace", "Trace movements of agent", cxxopts::value<unsigned>()->default_value(std::to_string(UINT32_MAX)));

    }
    void initializeArgs(const cxxopts::ParseResult& result) {
        tracked = result["trace"].as<unsigned>();
    }
    void init(const parser::LocationTypes& data) {
        publicSpace = data.publicSpace;
        home = data.home;
        hospital = data.hospital;
        printf("%d %d %d\n",publicSpace,home,hospital);
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
        thrust::device_vector<unsigned>& locationAgentList = realThis->locs->locationAgentList;
        unsigned *locationAgentListPtr = thrust::raw_pointer_cast(locationAgentList.data());
        thrust::device_vector<unsigned>& locationListOffsets = realThis->locs->locationListOffsets;
        unsigned *locationListOffsetsPtr = thrust::raw_pointer_cast(locationListOffsets.data());
        thrust::device_vector<unsigned>& locationIdsOfAgents = realThis->locs->locationIdsOfAgents;
        unsigned *locationIdsOfAgentsPtr = thrust::raw_pointer_cast(locationIdsOfAgents.data());
        thrust::device_vector<unsigned>& agentLocations = realThis->agents->location;
        unsigned *agentLocationsPtr = thrust::raw_pointer_cast(agentLocations.data());
        thrust::device_vector<unsigned>& agentTypes = realThis->agents->types;
        unsigned *agentTypesPtr = thrust::raw_pointer_cast(agentTypes.data());
        thrust::device_vector<typename SimulationType::PPState_t>& agentStates = realThis->agents->PPValues;
        typename SimulationType::PPState_t *agentStatesPtr = thrust::raw_pointer_cast(agentStates.data());
        unsigned *stepsUntilMovePtr = thrust::raw_pointer_cast(this->stepsUntilMove.data());

        //Arrays storing actual location IDs for each agent, for each location type
        thrust::device_vector<unsigned long> &locationOffset = realThis->agents->locationOffset;
        unsigned long*locationOffsetPtr = thrust::raw_pointer_cast(locationOffset.data());
        thrust::device_vector<unsigned> &possibleLocations = realThis->agents->possibleLocations;
        unsigned *possibleLocationsPtr = thrust::raw_pointer_cast(possibleLocations.data());
        thrust::device_vector<unsigned> &possibleTypes = realThis->agents->possibleTypes;
        unsigned *possibleTypesPtr = thrust::raw_pointer_cast(possibleTypes.data());

        //Arrays storing movement behaviour with general locationTypes - for each agent type, WB state, and day
        thrust::device_vector<unsigned> &eventOffset = realThis->agents->agentTypes.eventOffset;
        unsigned *eventOffsetPtr = thrust::raw_pointer_cast(eventOffset.data());
        thrust::device_vector<AgentTypeList::Event> &events = realThis->agents->agentTypes.events;
        AgentTypeList::Event *eventsPtr = thrust::raw_pointer_cast(events.data());

        unsigned numberOfAgents = agentLocations.size();
        unsigned numberOfLocations = locationListOffsets.size() - 1;

        Days day = simTime.getDay();

        #if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP
        #pragma omp parallel for
        for (unsigned i = 0; i < numberOfAgents; i++) {
            RealMovementOps::doMovement(i, stepsUntilMovePtr, agentStatesPtr,
                       agentTypesPtr,  eventOffsetPtr, eventsPtr, agentLocationsPtr,
                       locationOffsetPtr, possibleLocationsPtr, possibleTypesPtr, 
                       day, hospital, home, publicSpace, TimeDay(simTime), timeStep, this->tracked);
            
        }
        #elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
        RealMovementOps::doMovementDriver<<<(numberOfAgents-1)/256+1,256>>>(numberOfAgents, stepsUntilMovePtr, agentStatesPtr,
                       agentTypesPtr,  eventOffsetPtr, eventsPtr, agentLocationsPtr,
                       locationOffsetPtr, possibleLocationsPtr, possibleTypesPtr, 
                       day, hospital, home, publicSpace, TimeDay(simTime), timeStep, this->tracked);
        cudaDeviceSynchronize();
        #endif
        Util::updatePerLocationAgentLists(agentLocations, locationIdsOfAgents, locationAgentList, locationListOffsets);
    }
};
