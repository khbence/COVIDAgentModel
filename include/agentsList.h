#pragma once
#include <vector>
/*
template<typename T>
concept PPStateType = requires (T x) { x.update(); x.gotInfected(); };
*/

// singleton

template<typename PPState, typename Location>
class AgentList {
    std::vector<PPState> PPValues;
    std::vector<bool> diagnosed;
    std::vector<Location*> localizations;

    AgentList() = default;

public:
    [[nodiscard]] static AgentList* getInstance() {
        static AgentList instance;
        return &instance;
    }

    [[nodiscard]] bool checkConsistency() const;// if all vector are of the same lengths
};
