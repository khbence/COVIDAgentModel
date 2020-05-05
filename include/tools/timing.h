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

    static void reportWithParent(int parent, const std::string& indentation);

public:
    static void startTimer(const std::string& _name);
    static void stopTimer(const std::string& _name);
    static void report();
};