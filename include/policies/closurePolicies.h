#pragma once
#include <iostream>
#include "timeHandler.h"
#include "datatypes.h"
#include "cxxopts.hpp"
#include "operators.h"
#include "locationTypesFormat.h"
namespace ClosureHelpers {
    std::vector<std::string> splitHeader(std::string &header) {
        std::stringstream ss(header);
        std::string arg;
        std::vector<std::string> params;
        for (char i; ss >> i;) {
            arg.push_back(i);    
            if (ss.peek() == '\t') {
                if (arg.length()>0) {
                    params.push_back(arg);
                    arg.clear();
                }
                ss.ignore();
            }
        }
        if (arg.length()>0) {
            params.push_back(arg);
            arg.clear();
        }
        return params;
    }
}

template<typename SimulationType>
class NoClosure {
public:
    // add program parameters if we need any, this function got called already from Simulation
    static void addProgramParameters(cxxopts::Options& options) {}
    void initializeArgs(const cxxopts::ParseResult& result) {}
    void init(const parser::LocationTypes& data, std::string header) {}

    void midnight(Timehandler simTime, unsigned timeStep, std::vector<unsigned> &stats) {}
    void step(Timehandler simTime, unsigned timeStep) {}
};

template<typename SimulationType>
class RuleClosure {
    public:
    std::vector<std::string> header;
    class GlobalCondition {
        public:
        std::vector<unsigned> headerPos;
        std::vector<double> history;
        bool active;
        std::function<bool(GlobalCondition *, std::vector<unsigned>&)> condition;
        GlobalCondition(std::vector<unsigned> h, bool a, std::function<bool(GlobalCondition *, std::vector<unsigned>&)> r) : headerPos(h), active(a), condition(r) {}
    };
    std::vector<GlobalCondition> globalConditions;
    class Rule {
        public:
        std::vector<GlobalCondition *> conditions;
        std::function<void(Rule *)> rule;
        bool previousOpenState;
        Rule(std::vector<GlobalCondition *> c, std::function<void(Rule *)> r) : conditions(c), rule(r), previousOpenState(true) {}
    };
    std::vector<Rule> rules;
    // add program parameters if we need any, this function got called already from Simulation
    static void addProgramParameters(cxxopts::Options& options) {
        options.add_options()("closureParams",
            "Ratio of people in hospital when schools close",
            cxxopts::value<double>()->default_value("0.005"));
    }
    double closureParams;
    void initializeArgs(const cxxopts::ParseResult& result) {
        closureParams = result["closureParams"].as<double>();
    }
    void init(const parser::LocationTypes& data, std::string header) {
        this->header = ClosureHelpers::splitHeader(header);

        auto realThis = static_cast<SimulationType*>(this);
        thrust::device_vector<unsigned>& agentLocations = realThis->agents->location;
        unsigned numberOfAgents = agentLocations.size();

        //Condition for cases in hospital > 0.5% of population
        std::vector<unsigned> hospitalStates;
        double closureParams2 = closureParams;
        for (unsigned i = 0; i < this->header.size(); i++)
            if (this->header[i].find("_h")!=std::string::npos) hospitalStates.push_back(i);
        globalConditions.emplace_back(hospitalStates, false, [numberOfAgents,closureParams2](GlobalCondition* c, std::vector<unsigned>& stats){
            double value = (double)(stats[c->headerPos[0]]+stats[c->headerPos[1]]+stats[c->headerPos[2]])/(double)numberOfAgents;
            //c->history.push_back(value);
            return value > closureParams2;
            });
        
        unsigned schoolType = data.school;
        thrust::device_vector<typename SimulationType::TypeOfLocation_t>& locTypes = realThis->locs->locType;
        thrust::device_vector<bool>& locationOpenState = realThis->locs->states;
        //Rule for closing schools
        std::vector<GlobalCondition*> conds2 = {&globalConditions[0]};
        rules.emplace_back(conds2, [&,schoolType](Rule *r) {
            bool close = true;
            for (GlobalCondition *c : r->conditions) close = close && c->active;
            bool shouldBeOpen = !close;
            if (r->previousOpenState != shouldBeOpen) {
                thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(locTypes.begin(), locationOpenState.begin())),
                                 thrust::make_zip_iterator(thrust::make_tuple(locTypes.end(), locationOpenState.end())),
                                 [schoolType,shouldBeOpen]HD(thrust::tuple<typename SimulationType::TypeOfLocation_t&, bool&> tup)
                                 {
                                     auto& type = thrust::get<0>(tup);
                                     auto& isOpen = thrust::get<1>(tup);
                                     if (type == schoolType) isOpen = shouldBeOpen;
                                 });
                r->previousOpenState = shouldBeOpen;
                printf("Schools switched %d\n", (int)shouldBeOpen);
            }
        });

        unsigned workType = data.work;
        //Rule for closing workplaces
        std::vector<GlobalCondition*> conds3 = {&globalConditions[0]};
        rules.emplace_back(conds3, [&,workType](Rule *r) {
            bool close = true;
            for (GlobalCondition *c : r->conditions) close = close && c->active;
            bool shouldBeOpen = !close;
            if (r->previousOpenState != shouldBeOpen) {
                thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(locTypes.begin(), locationOpenState.begin())),
                                 thrust::make_zip_iterator(thrust::make_tuple(locTypes.end(), locationOpenState.end())),
                                 [workType,shouldBeOpen]HD(thrust::tuple<typename SimulationType::TypeOfLocation_t&, bool&> tup)
                                 {
                                     auto& type = thrust::get<0>(tup);
                                     auto& isOpen = thrust::get<1>(tup);
                                     if (type == workType) isOpen = shouldBeOpen;
                                 });
                r->previousOpenState = shouldBeOpen;
                printf("Workplaces switched %d\n", (int)shouldBeOpen);
            }
        });

        //Rule for closing pubs, churches, etc.
        std::vector<GlobalCondition*> conds = {&globalConditions[0]};
        rules.emplace_back(conds, [&](Rule *r) {
            bool close = true;
            for (GlobalCondition *c : r->conditions) close = close && c->active;
            bool shouldBeOpen = !close;
            if (r->previousOpenState != shouldBeOpen) {
                thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(locTypes.begin(), locationOpenState.begin())),
                                 thrust::make_zip_iterator(thrust::make_tuple(locTypes.end(), locationOpenState.end())),
                                 [shouldBeOpen]HD(thrust::tuple<typename SimulationType::TypeOfLocation_t&, bool&> tup)
                                 {
                                     auto& type = thrust::get<0>(tup);
                                     auto& isOpen = thrust::get<1>(tup);
                                     if (type == 5 || type == 6 || type == 9) isOpen = shouldBeOpen;
                                 });
                r->previousOpenState = shouldBeOpen;
                printf("LocTypes 5,6,9 switched %d\n", (int)shouldBeOpen);
            }
        });
    }

    void midnight(Timehandler simTime, unsigned timeStep, std::vector<unsigned> &stats) {
        for (GlobalCondition &c : globalConditions) {
            c.active = c.condition(&c,stats);
        }
        for (Rule &r : rules) {
            r.rule(&r);
        }
        
    }
    void step(Timehandler simTime, unsigned timeStep) {}
};