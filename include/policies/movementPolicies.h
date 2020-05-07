#pragma once
#include <iostream>
#include "timeHandler.h"

template<typename SimulationType>
class NoMovement {
protected:
    void planLocations() {}
    void movement(Timehandler simTime, unsigned timeStep) {}
};

template<typename SimulationType>
class DummyMovement {
    thrust::device_vector<unsigned> stepsUntilMove;
    
protected:
    void planLocations() {
        auto realThis = static_cast<SimulationType*>(this);
        thrust::device_vector<unsigned>& agentLocations = realThis->agents->location;
        unsigned numberOfAgents = agentLocations.size();

        if (stepsUntilMove.size()==0)
            stepsUntilMove.resize(numberOfAgents);
        thrust::fill(stepsUntilMove.begin(), stepsUntilMove.end(), 0u);


    }
    void movement(Timehandler simTime, unsigned timeStep) {
        auto realThis = static_cast<SimulationType*>(this);
        thrust::device_vector<unsigned>& locationAgentList = realThis->locationAgentList;
        thrust::device_vector<unsigned>& locationListOffsets = realThis->locationListOffsets;
        thrust::device_vector<unsigned>& locationIdsOfAgents = realThis->locationIdsOfAgents;
        thrust::device_vector<unsigned>& agentLocations = realThis->agents->location;
        unsigned numberOfAgents = agentLocations.size();
        unsigned numberOfLocations = locationListOffsets.size()-1;

        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(agentLocations.begin(),stepsUntilMove.begin())),
                         thrust::make_zip_iterator(thrust::make_tuple(agentLocations.end(),stepsUntilMove.end())),
                         [numberOfLocations](auto tuple){
                             auto &location = thrust::get<0>(tuple);
                             auto &stepsUntilMove = thrust::get<1>(tuple);
                             if (stepsUntilMove==0) {
                                 location = RandomGenerator::randomUnsigned(numberOfLocations-1);
                                 stepsUntilMove = RandomGenerator::randomUnsigned(144/4-1); //Move 4 times per day on average
                             }
                             stepsUntilMove--;
                         });
        Util::updatePerLocationAgentLists(agentLocations,locationIdsOfAgents,locationAgentList,locationListOffsets);

    }
};