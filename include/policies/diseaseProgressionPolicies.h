#pragma once
#include "PPStateTypes.h"

template<typename SimulationType>
class ExtendedProgression {
protected:
    using PPStateType = PPStateSIRextended;

    void updateDiseaseStates() {
        auto realThis = static_cast<SimulationType*>(this);
        auto& ppstates = realThis->agents->PPValues;
        auto& agentMeta = realThis->agents->agentMetaData;
        //Update states
        for_each(make_zip_iterator(make_tuple(ppstates.begin(), agentMeta.begin())),
                 make_zip_iterator(make_tuple(ppstates.end(), agentMeta.end())),
                 [](auto tup){
                     auto &ppstate = get<0>(tup);
                     auto &meta = get<1>(tup);
                     ppstate.update(meta.getScalingSymptoms());
                 });

        //Extract Idxs
        device_vector<char> idxs(ppstates.size());
        transform(ppstates.begin(), ppstates.end(), idxs.begin(),[](auto &ppstate){return ppstate.getIdx();});
        //Sort them
        sort(idxs.begin(),idxs.end());

        //We need to find the positions in the sorted idxs array where the following values begin
        std::array<char,10> h_states{0, 1,2,3,4,5,6,7, 10, 11};
        device_vector<char> states(h_states.begin(), h_states.end());
        device_vector<unsigned int> offsets(10);
        lower_bound(idxs.begin(), idxs.end(), states.begin(), states.end(), offsets.begin());

        //The number of S,I,R,D is simply the difference in positions of the relevant idxs
        host_vector<unsigned int> h_offsets(offsets);
        //The number of patients in various I subStates is the difference in positions of relevant idxs
        unsigned int stats[4] = {h_offsets[1]-h_offsets[0],h_offsets[7]-h_offsets[1],h_offsets[8]-h_offsets[7],h_offsets[9]-h_offsets[8]};
        unsigned int subStates[6];
        for (int i = 0; i < 6; i++)
            subStates[i] = h_offsets[i+2]-h_offsets[i+1];
       
        std::cout << stats[0] << ", " << stats[1] << ", " << stats[2] << ", " << stats[3];
        std::cout << ", " << subStates[0] << ", " << subStates[1] << ", " << subStates[2] 
                  << ", " << subStates[3] << ", " << subStates[4] << ", " << subStates[5] << std::endl;
    }
};