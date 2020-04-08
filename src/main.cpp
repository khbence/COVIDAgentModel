#include <iostream>
#include "PPStateSIRBasic.h"
#include "simulation.h"
#include <random>

int main(int argc, char const *argv[]) {
	int numAgents = 1000;
    if (argc>1) numAgents = atoi(argv[1]);
    double initial_infected_ratio = 0.01;
    if (argc>1) initial_infected_ratio = atof(argv[2]);

    //Basic config - PPStateSIRBasic for agent states, int PositionType, int TypeOfLocation
    Simulation<PPStateSIRBasic, int, int> s;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    //Create basic location for everyone
    s.locationList().push_back(Location<int,int>(0,0));
    Location<int, int> *location = &(s.locationList()[0]);

    //Populate agent list
    for (int i = 0; i < numAgents; i++) {
    	s.agentList()->addAgent(PPStateSIRBasic(dis(gen)<initial_infected_ratio?states::SIRD::I : states::SIRD::S),
    							false, location);
    }
    
    return 0;
}
