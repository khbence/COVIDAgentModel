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
    int mod(int a, int b)
    {
        int r = a % b;
        return r < 0 ? r + b : r;
    }
}

template<typename SimulationType>
class NoClosure {
public:
    // add program parameters if we need any, this function got called already from Simulation
    static void addProgramParameters(cxxopts::Options& options) {}
    void initializeArgs(const cxxopts::ParseResult& result) {}
    void init(const parser::LocationTypes& data, const parser::ClosureRules& rules, std::string header) {}

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
        int pos;
        bool active;
        std::function<bool(GlobalCondition *, std::vector<unsigned>&)> condition;
        GlobalCondition(std::vector<unsigned> h, bool a, std::function<bool(GlobalCondition *, std::vector<unsigned>&)> r) : headerPos(h), active(a), condition(r), pos(0) {}
    };
    std::vector<GlobalCondition> globalConditions;
    class Rule {
        public:
        std::string name;
        std::vector<GlobalCondition *> conditions;
        std::function<void(Rule *)> rule;
        bool previousOpenState;
        Rule(std::string n, std::vector<GlobalCondition *> c, std::function<void(Rule *)> r) : name (n), conditions(c), rule(r), previousOpenState(true) {}
    };
    std::vector<Rule> rules;
    // add program parameters if we need any, this function got called already from Simulation
    static void addProgramParameters(cxxopts::Options& options) {
        options.add_options()("enableClosures",
            "Enable(1)/disable(0) closure rules defined in closureRules.json",
            cxxopts::value<unsigned>()->default_value("0"));
        options.add_options()("maskCoefficient",
            "0.0-1.0 multiplier on infectiousness at non-home locations",
            cxxopts::value<double>()->default_value("1.0"))
            ("holidayMode",
            "enable/disable holiday mode - in cojunction with a HolidayMode closure policy");
    }
    unsigned enableClosures;
    bool curfewExists;
    double maskCoefficient;
    bool holidayModeExists;
    unsigned diagnosticLevel=0;
    void initializeArgs(const cxxopts::ParseResult& result) {
        enableClosures = result["enableClosures"].as<unsigned>();
        diagnosticLevel = result["diags"].as<unsigned>();
        maskCoefficient = result["maskCoefficient"].as<double>();
        try {
            curfewExists = result["curfew"].as<std::string>().length()>0;
        } catch (std::exception &e) {
            curfewExists = false;
        }
        holidayModeExists = result["holidayMode"].as<bool>();
    }
    void init(const parser::LocationTypes& data, const parser::ClosureRules& rules, std::string header) {
        this->header = ClosureHelpers::splitHeader(header);

        auto realThis = static_cast<SimulationType*>(this);
        thrust::device_vector<unsigned>& agentLocations = realThis->agents->location;
        unsigned numberOfAgents = agentLocations.size();
        unsigned diags = diagnosticLevel;

        std::vector<unsigned> hospitalStates;
        for (unsigned i = 0; i < this->header.size(); i++)
            if (this->header[i].find("_h")!=std::string::npos) hospitalStates.push_back(i);

        globalConditions.reserve(rules.rules.size());
        this->rules.reserve(rules.rules.size());

        for (const parser::ClosureRules::Rule& rule : rules.rules) {

            //check if masks/closures are enabled
            if (!enableClosures && !(maskCoefficient<1.0 && rule.name.compare("Masks")==0) && !rule.name.compare("Curfew")==0 && !rule.name.compare("HolidayMode")==0) continue;
            if (maskCoefficient==1.0 && rule.name.compare("Masks")==0) continue;

            //Create condition

            int closeAfter = rule.closeAfter; 
            int openAfter = rule.openAfter;
            double threshold = rule.threshold;
            if (rule.conditionType.compare("afterDays")==0) {
                std::vector<unsigned> none;
                if (closeAfter >=0 || openAfter<=0) throw CustomErrors("For closure rule 'afterDays', closeAfter must be -1, openAfter must be >0");
                globalConditions.emplace_back(none, false, [openAfter,threshold](GlobalCondition* c, std::vector<unsigned>& stats){
                     c->history[0]++; return (c->history[0]>=threshold && c->history[0] <= threshold + openAfter); });
                globalConditions.back().history.resize(1,0.0);
            } else if (rule.conditionType.compare("hospitalizedFraction")==0) {
                if (closeAfter <=0 || openAfter<=0) throw CustomErrors("For closure rule 'hospitalizedFraction', closeAfter and openAfter must be >0");
                globalConditions.emplace_back(hospitalStates, 0, [threshold, openAfter, closeAfter,numberOfAgents](GlobalCondition* c, std::vector<unsigned>& stats){
                    //calculate fraction
                    double accum = 0.0;
                    for (unsigned i = 0; i < c->headerPos.size(); i++) accum += stats[c->headerPos[i]];
                    double value = accum/(double)numberOfAgents;
                    //insert into history
                    c->history[c->pos] = value;
                    c->pos = (c->pos+1)%(c->history.size());
                    //check if above threshold, and if so has it been for the last closeAfter days
                    if (value > threshold) {
                        bool wasBelow = false;
                        for (unsigned i = ClosureHelpers::mod(c->pos-closeAfter,c->history.size()); i != c->pos; i=(i+1)%c->history.size()) {
                            wasBelow |= c->history[i] < threshold;
                        }
                        if (!wasBelow) return true;
                        else return c->active;
                    } else {
                        //below threshold for openAfter days
                        bool wasAbove = false;
                        for (unsigned i = ClosureHelpers::mod(c->pos-openAfter,c->history.size()); i != c->pos; i=(i+1)%c->history.size()) {
                            wasAbove |= c->history[i] >= threshold;
                        }
                        if (!wasAbove) return false;
                        else return c->active;
                    }
                });
                globalConditions.back().history.resize(std::max(closeAfter, openAfter)+2,0.0);
            } else if (rule.conditionType.compare("newDeadFraction")==0) {
                if (closeAfter <=0 || openAfter<=0) throw CustomErrors("For closure rule 'newDeadFraction', closeAfter and openAfter must be >0");
                std::vector<unsigned> deadState;
                for (unsigned i = 0; i < this->header.size(); i++)
                    if (this->header[i].compare("D1")==0) deadState.push_back(i);
                deadState.push_back(0);
                globalConditions.emplace_back(deadState, 0, [threshold, openAfter, closeAfter,numberOfAgents](GlobalCondition* c, std::vector<unsigned>& stats){
                    double value =  (stats[c->headerPos[0]]-c->headerPos[1])/(double)numberOfAgents; //new dead is number of dead - last time
                    c->headerPos[1] = stats[c->headerPos[0]]; //save previous value
                    //insert into history
                    c->history[c->pos] = value;
                    c->pos = (c->pos+1)%(c->history.size());
                    //check if above threshold, and if so has it been for the last closeAfter days
                    if (value > threshold) {
                        bool wasBelow = false;
                        for (unsigned i = ClosureHelpers::mod(c->pos-closeAfter,c->history.size()); i != c->pos; i=(i+1)%c->history.size()) {
                            wasBelow |= c->history[i] < threshold;
                        }
                        if (!wasBelow) return true;
                        else return c->active;
                    } else {
                        //below threshold for openAfter days
                        bool wasAbove = false;
                        for (unsigned i = ClosureHelpers::mod(c->pos-openAfter,c->history.size()); i != c->pos; i=(i+1)%c->history.size()) {
                            wasAbove |= c->history[i] >= threshold;
                        }
                        if (!wasAbove) return false;
                        else return c->active;
                    }
                });
                globalConditions.back().history.resize(std::max(closeAfter, openAfter)+2,0.0);
            } else {
                throw CustomErrors("Unknown closure type "+rule.conditionType);
            }

            if (rule.name.compare("Masks")!=0 && rule.name.compare("Curfew")!=0 && rule.name.compare("HolidayMode")!=0) { //Not masks or curfew
                //Create rule
                thrust::device_vector<typename SimulationType::TypeOfLocation_t>& locTypes = realThis->locs->locType;
                thrust::device_vector<bool>& locationOpenState = realThis->locs->states;
                thrust::device_vector<uint8_t>& locationEssential = realThis->locs->essential;
                const std::vector<int> &locationTypesToClose = rule.locationTypesToClose;

                //Create small fixed size array for listing types to close that can be captured properly 
                //typename SimulationType::TypeOfLocation_t fixListArr[10];
                std::array<unsigned, 10> fixListArr;
                if (locationTypesToClose.size()>=10) throw CustomErrors("Error, Closure Rule " + rule.name+ " has over 10 location types to close, please increase implementation limit");
                for (unsigned i = 0; i < locationTypesToClose.size(); i++) {
                    fixListArr[i] = locationTypesToClose[i];
                }
                for (unsigned i = locationTypesToClose.size(); i < 10; i++) fixListArr[i] = (typename SimulationType::TypeOfLocation_t)-1;

                //printf("cond %p\n",&globalConditions[globalConditions.size()-1]);
                std::vector<GlobalCondition*> conds = {&globalConditions[globalConditions.size()-1]};
                this->rules.emplace_back(rule.name, conds, [&,fixListArr,diags](Rule *r) {
                    bool close = true;
                    for (GlobalCondition *c : r->conditions) {close = close && c->active; /*printf("rule %s cond %p\n", r->name.c_str(), c);*/}
                    //printf("Rule %s %d\n", r->name.c_str(), close ? 1 : 0);
                    bool shouldBeOpen = !close;
                    if (r->previousOpenState != shouldBeOpen) {
                        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(locTypes.begin(), locationOpenState.begin(), locationEssential.begin())),
                                        thrust::make_zip_iterator(thrust::make_tuple(locTypes.end(), locationOpenState.end(), locationEssential.end())),
                                        [fixListArr,shouldBeOpen]HD(thrust::tuple<typename SimulationType::TypeOfLocation_t&, bool&, uint8_t&> tup)
                                        {
                                            auto& type = thrust::get<0>(tup);
                                            auto& isOpen = thrust::get<1>(tup);
                                            auto& isEssential = thrust::get<2>(tup);
                                            if (isEssential==1) return;
                                            for (unsigned i = 0; i < 10; i++)
                                                if (type == fixListArr[i])
                                                    isOpen = shouldBeOpen;
                                                else if ((typename SimulationType::TypeOfLocation_t)-1 == fixListArr[i]) break;
                                        });
                        r->previousOpenState = shouldBeOpen;
                        if (diags>0) printf("Rule %s %s\n", r->name.c_str(), (int)shouldBeOpen ? "disabled": "enabled");
                    }
                });
            } else if (rule.name.compare("Masks")==0) {
                //Masks
                thrust::device_vector<double>& locInfectiousness = realThis->locs->infectiousness;
                thrust::device_vector<typename SimulationType::TypeOfLocation_t>& locTypes = realThis->locs->locType;
                unsigned homeType = data.home;
                double maskCoefficient2 = maskCoefficient;
                std::vector<GlobalCondition*> conds = {&globalConditions[globalConditions.size()-1]};
                this->rules.emplace_back(rule.name, conds, [&,homeType,maskCoefficient2,diags](Rule *r) {
                    bool close = true;
                    for (GlobalCondition *c : r->conditions) {close = close && c->active;}
                    bool shouldBeOpen = !close;
                    if (r->previousOpenState != shouldBeOpen) {
                        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(locTypes.begin(), locInfectiousness.begin())),
                                        thrust::make_zip_iterator(thrust::make_tuple(locTypes.end(), locInfectiousness.end())),
                                        [maskCoefficient2,homeType,shouldBeOpen]HD(thrust::tuple<typename SimulationType::TypeOfLocation_t&, double&> tup)
                                        {
                                            auto& type = thrust::get<0>(tup);
                                            auto& infectiousness = thrust::get<1>(tup);
                                            if (type != homeType) {
                                                if (shouldBeOpen) infectiousness = infectiousness / maskCoefficient2;
                                                else infectiousness = infectiousness * maskCoefficient2;
                                            }
                                        });
                        r->previousOpenState = shouldBeOpen;
                        if (diags>0) printf("Masks %s with %g multiplier\n", (int)shouldBeOpen ? "off": "on", maskCoefficient2);
                    }
                });
            } else if (rule.name.compare("Curfew")==0) {
                if (!curfewExists) continue;
                //Curfew
                std::vector<GlobalCondition*> conds = {&globalConditions[globalConditions.size()-1]};
                auto realThis = static_cast<SimulationType*>(this);
                this->rules.emplace_back(rule.name, conds, [&, realThis,diags](Rule *r) {
                    bool close = true;
                    for (GlobalCondition *c : r->conditions) {close = close && c->active;}
                    bool shouldBeOpen = !close;
                    if (r->previousOpenState != shouldBeOpen) {
                        realThis->toggleCurfew(close);
                        r->previousOpenState = shouldBeOpen;
                        if (diags>0) printf("Curfew %s\n", (int)shouldBeOpen ? "disabled": "enabled");
                    }
                });
            } else if (rule.name.compare("HolidayMode")==0) {
                if (!holidayModeExists) continue;
                //Curfew
                std::vector<GlobalCondition*> conds = {&globalConditions[globalConditions.size()-1]};
                auto realThis = static_cast<SimulationType*>(this);
                this->rules.emplace_back(rule.name, conds, [&, realThis,diags](Rule *r) {
                    bool close = true;
                    for (GlobalCondition *c : r->conditions) {close = close && c->active;}
                    bool shouldBeOpen = !close;
                    if (r->previousOpenState != shouldBeOpen) {
                        realThis->toggleHolidayMode(close);
                        r->previousOpenState = shouldBeOpen;
                        if (diags>0) printf("Holiday mode %s\n", (int)shouldBeOpen ? "disabled": "enabled");
                    }
                });
            }
        }
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
