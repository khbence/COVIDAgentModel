#pragma once
#include "rapidjson/rapidjson.h"
#include "rapidjson/document.h"
#include <string>

class JSONWriter {
protected:
    rapidjson::Document d;
    rapidjson::Document::AllocatorType& allocator;

    JSONWriter();

public:
    void writeFile(const std::string& fileName) const;
};