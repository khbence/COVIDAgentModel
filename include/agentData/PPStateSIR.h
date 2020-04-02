#pragma once

class PPStateSIR {
    enum class PP { S = 0, I, R, D };
    
    //just to show this is also possible
    friend PP& operator++(PP& e) {
        return e = static_cast<PP>(static_cast<int>(e)+1);
    }

    PP state = PP::S;
    //static MarkovChain mc; //specific for this, but uses only indexes not the enum type

public:
    void update(/*elapsed time step?*/) {
        ++state;
        // of course the increment doesn't make much sense here, probably we'll have a Markov chain here
        //if(state == PP::I) state = static_cast<PP>(mc(static_cast<int>(state)));
    }

    void infect() {
        state = PP::I;
    }
};