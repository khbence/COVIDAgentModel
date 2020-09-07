#pragma once
#include <iostream>
#include "timeHandler.h"
#include "datatypes.h"
#include "cxxopts.hpp"
#include "operators.h"
#include "locationTypesFormat.h"

template<typename SimulationType>
class NoTesting {
public:
    // add program parameters if we need any, this function got called already from Simulation
    static void addProgramParameters(cxxopts::Options& options) {}
    void initializeArgs(const cxxopts::ParseResult& result) {}
    void init(const parser::LocationTypes& data) {}

    void performTests(Timehandler simTime, unsigned timeStep) {}
};



namespace DetailedTestingOps {
        template<typename PPState, typename LocationType>
    struct TestingArguments {
        TestingArguments() {}
        PPState *agentStatesPtr;
        AgentStats* agentStatsPtr;
        unsigned long* locationOffsetPtr;
        unsigned* possibleLocationsPtr;
        unsigned* possibleTypesPtr;
        unsigned* locationQuarantineUntilPtr;
        unsigned hospitalType;
        unsigned homeType;
        unsigned publicPlaceType;
        unsigned doctorType;
        unsigned schoolType;
        unsigned workType;
        unsigned timeStep;
        unsigned timestamp;
        unsigned tracked;
        LocationType* locationTypePtr;
        unsigned *lastTestPtr;
        bool *locationFlagsPtr;
        bool* diagnosedPtr;
    };

    template<typename PPState, typename LocationType>
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    __device__
#endif
    void
    flagLocations(unsigned i,  TestingArguments<PPState, LocationType> &a) {
        //If diagnosed in the last 24 hours
        if (a.agentStatsPtr[i].diagnosedTimestamp > a.timestamp - 24 * 60 / a.timeStep) {
            //Mark home
            unsigned home = RealMovementOps::findActualLocationForType(i, a.homeType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr);
            a.locationFlagsPtr[home] = true;
            //Mark work
            unsigned work = RealMovementOps::findActualLocationForType(i, a.workType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr);
            if (work != std::numeric_limits<unsigned>::max() && a.locationQuarantineUntilPtr[work] < a.timestamp)
                a.locationFlagsPtr[work] = true;
            //Mark school
            unsigned school = RealMovementOps::findActualLocationForType(i, a.schoolType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr);
            if (school != std::numeric_limits<unsigned>::max() && a.locationQuarantineUntilPtr[school] < a.timestamp)
                a.locationFlagsPtr[school] = true;

            if (a.tracked == i) {
                printf("Testing: Agent %d was diagnosed in last 24 hours, marking home %d, work %d school %d\n",
                    i, home, 
                    work==std::numeric_limits<unsigned>::max()?-1:(int)work,
                    school==std::numeric_limits<unsigned>::max()?-1:(int)school);
            }
        }

    }
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    template<typename PPState, typename LocationType>
    __global__ void flagLocationsDriver(TestingArguments<PPState, LocationType> &a, unsigned numberOfAgents ) {
        unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < numberOfAgents) { RealMovementOps::flagLocations(i, TestingArguments<PPState, LocationType> &a); }
    }
#endif

template<typename PPState, typename LocationType>
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    __device__
#endif
    void
    doTesting(unsigned i,  TestingArguments<PPState, LocationType> &a) {
        //if recently tested, don't test again
        if (a.timestamp > 5*24*60/a.timeStep && a.lastTestPtr[i] != std::numeric_limits<unsigned>::max() &&
            a.lastTestPtr[i] > a.timestamp - 5*24*60/a.timeStep) return; //TODO: how many days?
    
        //Check home
        unsigned home = RealMovementOps::findActualLocationForType(i, a.homeType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr);
        bool homeFlag = a.locationFlagsPtr[home];
        //Check work
        unsigned work = RealMovementOps::findActualLocationForType(i, a.workType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr);
        bool workFlag = false;
        if (work != std::numeric_limits<unsigned>::max())
            workFlag = a.locationFlagsPtr[work];
        //Check school
        unsigned school = RealMovementOps::findActualLocationForType(i, a.schoolType, a.locationOffsetPtr, a.possibleLocationsPtr, a.possibleTypesPtr);
        bool schoolFlag = false;
        if (school != std::numeric_limits<unsigned>::max())
            schoolFlag = a.locationFlagsPtr[school];

        double testingProbability = 0.0;
        testingProbability += homeFlag * 0.2;
        testingProbability += workFlag * 0.1;
        testingProbability += schoolFlag * 0.1;

        //If agent works in hospital or doctor's office
        if (work != std::numeric_limits<unsigned>::max() &&
            (a.locationTypePtr[work] ==  a.doctorType || 
            a.locationTypePtr[work] ==  a.hospitalType)) {
            testingProbability += 0.2;
        }

        if (a.tracked == i && testingProbability>0.0) 
            printf("Testing: Agent %d testing probability: %g\n",
                    i, testingProbability);
        
        //Do the test
        if (testingProbability>1.0 ||
            RandomGenerator::randomReal(1.0) < testingProbability) { 
            a.lastTestPtr[i] = a.timestamp;
#warning Should check for infected, not infectious!
            if (a.agentStatesPtr[i].isInfectious()) {
                a.diagnosedPtr[i] = true;
                a.agentStatsPtr[i].diagnosedTimestamp = a.timestamp;
                if (a.tracked == i) 
                    printf("\t Agent %d tested positive\n", i);
            } else {
                if (a.tracked == i) 
                    printf("\t Agent %d tested negative\n", i);
            }
            
        } else if (testingProbability>0.0) {
            if (a.tracked == i) 
                printf("\t Agent %d was not tested\n");
        }

        
    }
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    template<typename PPState, typename LocationType>
    __global__ void doTestingDriver(TestingArguments<PPState, LocationType> &a, unsigned numberOfAgents ) {
        unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < numberOfAgents) { RealMovementOps::doTesting(i, TestingArguments<PPState, LocationType> &a); }
    }
#endif
}// namespace DetailedTestingOps


template<typename SimulationType>
class DetailedTesting {
public:
    // add program parameters if we need any, this function got called already from Simulation
    static void addProgramParameters(cxxopts::Options& options) {}
    void initializeArgs(const cxxopts::ParseResult& result) {}

    unsigned publicSpace;
    unsigned home;
    unsigned hospital;
    unsigned doctor;
    unsigned tracked;
    unsigned quarantinePolicy;
    unsigned school;
    unsigned work;


    void init(const parser::LocationTypes& data) {
        publicSpace = data.publicSpace;
        home = data.home;
        hospital = data.hospital;
        doctor = data.doctor;
        school = data.school;
        work = data.work;
    }

    thrust::device_vector<unsigned> lastTest;
    thrust::device_vector<bool> locationFlags;

    void performTests(Timehandler simTime, unsigned timeStep) {
        PROFILE_FUNCTION();
        auto realThis = static_cast<SimulationType*>(this);
        DetailedTestingOps::TestingArguments<typename SimulationType::PPState_t, typename SimulationType::TypeOfLocation_t> a;

        thrust::device_vector<unsigned>& agentLocations = realThis->agents->location;
        unsigned numberOfLocations = realThis->locs->locType.size();
        unsigned numberOfAgents = agentLocations.size();
        a.timestamp = simTime.getTimestamp();

        //Running for the first time - initialize arrays
        if (lastTest.size()==0) {
            lastTest.resize(numberOfAgents);
            thrust::fill(lastTest.begin(), lastTest.end(), std::numeric_limits<unsigned>::max());
            locationFlags.resize(numberOfLocations);
            tracked = realThis->locs->tracked;
        }
        //Set all flags of all locations to false (no recent diagnoses)
        thrust::fill(locationFlags.begin(), locationFlags.end(), false);

        a.tracked = tracked;
        a.locationFlagsPtr = thrust::raw_pointer_cast(locationFlags.data());
        a.lastTestPtr = thrust::raw_pointer_cast(lastTest.data());
        a.hospitalType = hospital;
        a.homeType = home;
        a.publicPlaceType = publicSpace;
        a.doctorType = doctor;
        a.timeStep = timeStep;
        a.schoolType = school;
        a.workType = work;

        //agent data
        thrust::device_vector<AgentStats>& agentStats = realThis->agents->agentStats;
        a.agentStatsPtr = thrust::raw_pointer_cast(agentStats.data());
        thrust::device_vector<typename SimulationType::PPState_t>& agentStates = realThis->agents->PPValues;
        a.agentStatesPtr = thrust::raw_pointer_cast(agentStates.data());
        thrust::device_vector<bool>& diagnosed = realThis->agents->diagnosed;
        a.diagnosedPtr = thrust::raw_pointer_cast(diagnosed.data());
        //primary location types
        thrust::device_vector<typename SimulationType::TypeOfLocation_t>& locationTypes = realThis->locs->locType;
        a.locationTypePtr = thrust::raw_pointer_cast(locationTypes.data());
        // Arrays storing actual location IDs for each agent, for each location type
        thrust::device_vector<unsigned long>& locationOffset = realThis->agents->locationOffset;
        a.locationOffsetPtr = thrust::raw_pointer_cast(locationOffset.data());
        thrust::device_vector<unsigned>& possibleLocations = realThis->agents->possibleLocations;
        a.possibleLocationsPtr = thrust::raw_pointer_cast(possibleLocations.data());
        thrust::device_vector<unsigned>& possibleTypes = realThis->agents->possibleTypes;
        a.possibleTypesPtr = thrust::raw_pointer_cast(possibleTypes.data());
        thrust::device_vector<unsigned>& locationQuarantineUntil = realThis->locs->quarantineUntil;
        a.locationQuarantineUntilPtr = thrust::raw_pointer_cast(locationQuarantineUntil.data());

        //
        //Step 1 - flag locations of anyone diagnosed yesterday
        //

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP
#pragma omp parallel for
        for (unsigned i = 0; i < numberOfAgents; i++) { DetailedTestingOps::flagLocations(i, a); }
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
        DetailedTestingOps::flagLocationsDriver<<<(numberOfAgents - 1) / 256 + 1, 256>>>(a, numberOfAgents);
        cudaDeviceSynchronize();
#endif

        //
        //Step 2 - do the testing
        //

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP
#pragma omp parallel for
        for (unsigned i = 0; i < numberOfAgents; i++) { DetailedTestingOps::doTesting(i, a); }
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
        DetailedTestingOps::doTestingDriver<<<(numberOfAgents - 1) / 256 + 1, 256>>>(a, numberOfAgents);
        cudaDeviceSynchronize();
#endif

    }
};
