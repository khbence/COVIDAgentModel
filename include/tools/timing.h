#pragma once
#include <chrono>
#include <map>
#include <vector>
#include <iostream>

class Timing {
    struct LoopData {
    int index = 0;
    int parent = 0;
    double time = 0.0;
    std::chrono::time_point<std::chrono::system_clock> current;
    };
    
    static std::map<std::string, LoopData> loops;
    static std::vector<int> stack;
    static int counter;
    
    static void reportWithParent(int parent, std::string indentation) {
        for (std::pair<std::string, LoopData> element : loops) {
            LoopData& l = element.second;
            if (l.parent == parent) {
                std::cout << indentation + element.first + ": " + std::to_string(l.time) + " seconds\n";
                reportWithParent(l.index, indentation+"  ");
            }
        }
    }

    public:
    static void startTimer(std::string _name) {
        auto now = std::chrono::system_clock::now();
        if (loops.size()==0) counter = 0;
        int parent = stack.size()==0 ? -1 : stack.back();
        std::string fullname = _name + "(" + std::to_string(parent) + ")";
        int index;
        if (loops.find(fullname) != loops.end()) {
            loops[fullname].current = now;
            index = loops[fullname].index;
        } else {
            loops[fullname] = {counter++, parent, 0.0, now};
            index = counter-1;
        }
        stack.push_back(index);
    }

    static void stopTimer(std::string _name) {
        int my_index = stack.back();
        stack.pop_back();
        int parent = stack.size()==0 ? -1 : stack.back();
        std::string fullname = _name + "(" + std::to_string(parent) + ")";
        auto now = std::chrono::system_clock::now();
        loops[fullname].time += std::chrono::duration<double>(now-loops[fullname].current).count();
    }

    static void report() {
        std::vector<int> loopstack;
        int parent = -1;
        std::string indentation = "  ";
        for (std::pair<std::string, LoopData> element : loops) {
            LoopData& l = element.second;
            if (l.parent == parent) {
                std::cout << indentation + element.first + ": " + std::to_string(l.time) + " seconds\n";
                reportWithParent(l.index, indentation+"  ");
            }
        }
    }
};