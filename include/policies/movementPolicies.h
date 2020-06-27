#pragma once
#include <iostream>
#include "timeHandler.h"
#include "datatypes.h"
#include "cxxopts.hpp"

template<typename SimulationType>
class NoMovement {
protected:
    void planLocations() {}
    void movement(Timehandler simTime, unsigned timeStep) {}
};

template<typename SimulationType>
class DummyMovement {
    thrust::device_vector<unsigned> stepsUntilMove;

public:
    // add program parameters if we need any, this function got called already from Simulation
    static void addProgramParameters(cxxopts::Options& options) {}
    void initializeArgs(const cxxopts::ParseResult& result) {}

    void planLocations() {
        auto realThis = static_cast<SimulationType*>(this);
        thrust::device_vector<unsigned>& agentLocations = realThis->agents->location;
        unsigned numberOfAgents = agentLocations.size();

        if (stepsUntilMove.size() == 0) stepsUntilMove.resize(numberOfAgents);
        thrust::fill(stepsUntilMove.begin(), stepsUntilMove.end(), 0u);
    }

    void movement(Timehandler simTime, unsigned timeStep) {
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

template<typename SimulationType>
class RealMovement : public DummyMovement {
    [[nodiscard]] static unsigned findActualLocationForType(unsigned agent, unsigned locType, unsigned *locationOffsetPtr, unsigned *possibleLocationsPtr, unsigned *possibleTypesPtr) {
        for (unsigned i = locationOffsetPtr[agent]; i < locationOffsetPtr[agent+1]; i++) {
            if (locType == possibleTypesPtr[i])
                return possibleLocationsPtr[i];
        }
    }

    public:
    void movement(Timehandler simTime, unsigned timeStep) override {
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
        thrust::device_vector<SimulationType::PPState_t>& agentStates = realThis->agents->PPValues;
        SimulationType::PPState_t *agentStatesPtr = thrust::raw_pointer_cast(agentStates.data());
        unsigned *stepsUntilMovePtr = thrust::raw_pointer_cast(this->stepsUntilMove.data());

        //Arrays storing actual location IDs for each agent, for each location type
        thrust::device_vector<unsigned long> &locationOffset = realThis->agents->locationOffset;
        unsigned *locationOffsetPtr = thrust::raw_pointer_cast(locationOffset.data());
        thrust::device_vector<unsigned> &possibleLocations = realThis->agents->possibleLocations;
        unsigned *possibleLocationsPtr = thrust::raw_pointer_cast(possibleLocations.data());
        thrust::device_vector<unsigned> &possibleTypes = realThis->agents->possibleTypes;
        unsigned *possibleTypesPtr = thrust::raw_pointer_cast(possibleTypes.data());

        //Arrays storing movement behaviour with general locationTypes - for each agent type, WB state, and day
        thrust::device_vector<unsigned> &eventOffset = realThis->agents->agentTypes.eventOffset;
        unsigned *eventOffsetPtr = thrust::raw_pointer_cast(eventOffset.data());
        thrust::device_vector<Event> &events = realThis->agents->agentTypes.events;
        AgentTypeList::Event *eventsPtr = thrust::raw_pointer_cast(events.data());

        unsigned numberOfAgents = agentLocations.size();
        unsigned numberOfLocations = locationListOffsets.size() - 1;

        Days day = simTime.getDay();

        for (unsigned i = 0; i < numberOfAgents; i++) {
            if (stepsUntilMovePtr[i]>0) {
                stepsUntilMovePtr[i]--;
                continue;
            }
            states::WBStates wBState = agentTypesPtr[i].getWBState();
            if (wBState == states::WBStates::D) { //If dead, do not go anywhere
                stepsUntilMovePtr[i] = UINT32_MAX;
                continue;
            }

            unsigned &agentType = agentTypesPtr[i];

            unsigned agentTypeOffset = AgentTypeList::getOffsetIndex(agentType,wBState, day);
            unsigned eventsBegin = eventOffsetPtr[agentTypeOffset];
            unsigned eventsEnd = eventOffsetPtr[agentTypeOffset];

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
            //Possibilities:
            // 1 both are -1 -> no more events for that day. Should be home if wBState != S, or at hospital if S
            // 2 Begin != -1, End == -1 -> last event for the day. Move there (if needed pick randomly)
            // 3 Begin == -1, End != -1 -> no events right now, but there will be some later
            //      3a if less than 30 mins until next possible event, then stay here
            //      3b if 30-60 to next possible event, should go to public place (type 0)
            //      3c more than 60 mins, then go home
            // 4 neither -1, then pick randomly between one of the events

            //ISSUES:
            //do we forcibly finish at midnight?? WHat if the duration goes beyond that?

            //Case 1
            if (activeEventsBegin==-1 && activeEventsEnd == -1) {
                unsigned typeToGoTo = wBState == states::WBStates::S ? 12 : 2; //Hostpital if sick, home otherwise
                unsigned myHome = findActualLocationForType(i, typeToGoTo, locationOffsetPtr, possibleLocationsPtr, possibleTypesPtr);
                agentLocationsPtr[i] = myHome;
                stepsUntilMovePtr[i] = simTime.stepsUntilMidnight(); //TODO: Need to figure out how many timesteps until midnight (if we forcibly stop then)
            }
            //Case 2 and 4
            if (activeEventsBegin!=-1) {
                unsigned numPotentialEvents = eventsEnd-activeEventsBegin;
                unsigned newLocationType = -1;
                TimeDayDuration basicDuration(0);
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
                unsigned newLocation = findActualLocationForType(i, newLocationType, locationOffsetPtr, possibleLocationsPtr, possibleTypesPtr);
                agentLocationsPtr[i] = newLocation;
                if (activeEventsEnd==-1) {
                    //TODO: Do we truncate to midnight?
                    if (simTime + basicDuration > simTime.nextMidnight()) { //TODO: is this right?
                        stepsUntilMovePtr[i] = simTime.stepsUntilMidnight();
                    } else {
                        //does not last till midnight, but no events afterwards - spend full duration there
                        stepsUntilMovePtr[i] = basicDuration.steps(timeStep); //TODO: Need to be able to convert TimeDayDuration to number of timesteps
                    }
                } else {
                    //If duration is less then the beginning of the next move window, then spend full duration here
                    if (simtTime + basicDuration < eventsPtr[activeEventsEnd].start)
                        stepsUntilMovePtr[i] = basicDuration.steps(timeStep); //TODO: Need to be able to convert TimeDayDuration to number of timesteps
                    else {
                        //Otherwise I need to move again randomly between the end of this duration and the end of next movement window
                        TimeDayDuration window = eventsPtr[activeEventsEnd].end - (simtTime + basicDuration);
                        unsigned randExtra = RandomGenerator::randomUnsigned(window.steps(timeStep));
                        stepsUntilMovePtr[i] = basicDuration.steps(timeStep) + randExtra;
                    }
                }
                
            }

            //Movement should start at random within the movement period (i.e. between start and end)
            //->but here we need to determine the exact time of the next move
        }
    }
}