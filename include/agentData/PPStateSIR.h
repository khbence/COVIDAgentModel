#pragma once

//TODO create more specialized SIRD++ class, makes this tempated of that
//this will be kinda an interface for that
class PPStateSIR {
    enum class PP { S = 0, I, R, D };

    PP state = PP::S;
    unsigned counter = 0;
    //static MarkovChain mc; //specific for this, but uses only indexes not the enum type

public:
    void update(/*elapsed time step?*/);

    void gotInfected();

    //WBState getWBState() const;

};